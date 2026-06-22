#!/usr/bin/env python3
import sys
import os
import subprocess
import threading
from collections import defaultdict

# ============================================================
# 인터페이스 목록 - 빼고 싶으면 주석 처리, 추가하고 싶으면 줄 추가
# 같은 서버의 서로 다른 NIC 주소 (각 200Gbps)
# ============================================================
INTERFACES = [
    "10.0.1.2",
    "10.0.2.2",
    "10.0.3.2",
    "10.0.4.2",
]

REMOTE_USER = "user"
REMOTE_BASE = "/home/user/Models"
SCP_OPTS = ["-T", "-c", "aes128-gcm@openssh.com", "-o", "Compression=no", "-o", "StrictHostKeyChecking=no"]
SSH_OPTS = ["-o", "StrictHostKeyChecking=no"]
CHECK_BYTES = 512
SPLIT_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1GB 이상이면 분할 전송


def get_local_info(src_abs, file_list):
    """로컬 파일의 크기, 앞/중간/뒤 512바이트"""
    info = {}
    for rel in file_list:
        path = os.path.join(src_abs, rel)
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(CHECK_BYTES).hex()
            mid_offset = max(0, size // 2 - CHECK_BYTES // 2)
            f.seek(mid_offset)
            mid = f.read(CHECK_BYTES).hex()
            if size > CHECK_BYTES:
                f.seek(-CHECK_BYTES, 2)
            tail = f.read(CHECK_BYTES).hex()
        info[rel] = (size, head, mid, tail)
    return info


def get_remote_info(host, remote_dir, file_list):
    """원격 파일의 앞/중간/뒤 512바이트"""
    if not file_list:
        return {}
    script_lines = []
    for rel in file_list:
        fp = f"{remote_dir}/{rel}"
        script_lines.append(
            f'F="{fp}"; '
            f'if [ -f "$F" ]; then '
            f'S=$(stat -c%s "$F"); '
            f'H=$(head -c {CHECK_BYTES} "$F" | xxd -p | tr -d "\\n"); '
            f'MO=$(( S / 2 - {CHECK_BYTES // 2} )); '
            f'[ $MO -lt 0 ] && MO=0; '
            f'M=$(dd if="$F" bs=1 skip=$MO count={CHECK_BYTES} 2>/dev/null | xxd -p | tr -d "\\n"); '
            f'T=$(tail -c {CHECK_BYTES} "$F" | xxd -p | tr -d "\\n"); '
            f'echo "{rel}|$S|$H|$M|$T"; '
            f'fi'
        )
    script = "\n".join(script_lines)
    cmd = ["ssh"] + SSH_OPTS + [f"{REMOTE_USER}@{host}", f"bash -c '{script}'"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = {}
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            if not line or "|" not in line:
                continue
            parts = line.split("|", 4)
            if len(parts) == 5:
                rel, size, head, mid, tail = parts
                info[rel] = (int(size), head, mid, tail)
    return info


def create_remote_dirs(host, remote_dir, dir_set):
    if not dir_set:
        return
    dirs = " ".join(f"'{remote_dir}/{d}'" for d in sorted(dir_set))
    cmd = ["ssh"] + SSH_OPTS + [f"{REMOTE_USER}@{host}", f"mkdir -p {dirs}"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def collect_files(src_dir):
    all_files = []
    src_abs = os.path.abspath(src_dir)
    for root, dirs, files in os.walk(src_abs):
        for f in sorted(files):
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, src_abs)
            all_files.append(rel_path)
    return all_files


def fmt_size(n):
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.2f}GB"
    elif n >= 1024 ** 2:
        return f"{n / 1024**2:.1f}MB"
    elif n >= 1024:
        return f"{n / 1024:.0f}KB"
    return f"{n}B"


# =================================================================
# 소형 파일 전송 (라운드 로빈)
# =================================================================
def worker(iface, file_list, src_abs, remote_base_dir, total, results, lock):
    for idx, rel_path in file_list:
        local_path = os.path.join(src_abs, rel_path)
        remote_path = f"{REMOTE_USER}@{iface}:{remote_base_dir}/{rel_path}"
        scp_cmd = ["scp"] + SCP_OPTS + [local_path, remote_path]
        result = subprocess.run(scp_cmd)
        status = "OK" if result.returncode == 0 else "FAIL"
        with lock:
            print(f"[{idx}/{total}] {rel_path} -> {iface} : {status}", flush=True)
            results.append((rel_path, iface, result.returncode == 0))


# =================================================================
# 대형 파일 분할 전송
# =================================================================
def send_large_file(rel_path, src_abs, remote_base_dir, file_num, large_total, lock):
    local_path = os.path.join(src_abs, rel_path)
    remote_path = f"{remote_base_dir}/{rel_path}"
    file_size = os.path.getsize(local_path)
    num_parts = len(INTERFACES)
    part_size = file_size // num_parts

    with lock:
        print(f"\n[LARGE {file_num}/{large_total}] {rel_path} ({fmt_size(file_size)}) -> {num_parts} parts", flush=True)

    # --- 1) 원격 잔여 .part 파일 정리 ---
    parts_remote = [f"{remote_path}.part{i+1}" for i in range(num_parts)]
    cleanup = " ".join(f"'{p}'" for p in parts_remote)
    subprocess.run(
        ["ssh"] + SSH_OPTS + [f"{REMOTE_USER}@{INTERFACES[0]}", f"rm -f {cleanup}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # --- 2) 로컬에서 dd로 병렬 분할 ---
    part_paths = [f"{local_path}.part{i+1}" for i in range(num_parts)]
    split_ok = [False] * num_parts

    with lock:
        print(f"  Splitting locally ({num_parts} parts)...", flush=True)

    def _split(idx):
        offset = idx * part_size
        length = file_size - offset if idx == num_parts - 1 else part_size
        dd_cmd = [
            "dd", f"if={local_path}", f"of={part_paths[idx]}",
            "bs=8M", f"skip={offset}", f"count={length}",
            "iflag=skip_bytes,count_bytes", "status=none"
        ]
        r = subprocess.run(dd_cmd)
        split_ok[idx] = (r.returncode == 0)

    threads = [threading.Thread(target=_split, args=(i,)) for i in range(num_parts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not all(split_ok):
        failed = [i+1 for i, ok in enumerate(split_ok) if not ok]
        for pp in part_paths:
            try: os.remove(pp)
            except OSError: pass
        with lock:
            print(f"  [FAIL] Local split failed at part{failed}", flush=True)
        return False

    # --- 3) 각 파트를 인터페이스별 병렬 scp ---
    send_ok = [False] * num_parts

    def _send(idx):
        iface = INTERFACES[idx]
        part_remote = f"{REMOTE_USER}@{iface}:{remote_path}.part{idx+1}"
        scp_cmd = ["scp"] + SCP_OPTS + [part_paths[idx], part_remote]
        r = subprocess.run(scp_cmd)
        send_ok[idx] = (r.returncode == 0)
        ps = os.path.getsize(part_paths[idx])
        with lock:
            print(f"  part{idx+1} ({fmt_size(ps)}) -> {iface} : {'OK' if send_ok[idx] else 'FAIL'}", flush=True)

    threads = [threading.Thread(target=_send, args=(i,)) for i in range(num_parts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # --- 4) 로컬 파트 정리 (성공이든 실패든) ---
    for pp in part_paths:
        try: os.remove(pp)
        except OSError: pass

    # --- 5) 전송 실패 체크 ---
    if not all(send_ok):
        failed = [i+1 for i, ok in enumerate(send_ok) if not ok]
        with lock:
            print(f"  [FAIL] part{failed} transfer failed.", flush=True)
            print(f"         .part files remain on remote (retry will clean up)", flush=True)
        return False

    # --- 6) 원격 합체 ---
    with lock:
        print(f"  Assembling on remote...", flush=True)

    parts_str = " ".join(f"'{p}'" for p in parts_remote)
    assemble = subprocess.run(
        ["ssh"] + SSH_OPTS + [
            f"{REMOTE_USER}@{INTERFACES[0]}",
            f"cat {parts_str} > '{remote_path}' && rm {parts_str}"
        ]
    )

    if assemble.returncode != 0:
        with lock:
            print(f"  [FAIL] Assembly failed. .part files remain on remote.", flush=True)
        return False

    # --- 7) 크기 검증 ---
    verify = subprocess.run(
        ["ssh"] + SSH_OPTS + [
            f"{REMOTE_USER}@{INTERFACES[0]}",
            f"stat -c%s '{remote_path}'"
        ],
        capture_output=True, text=True
    )

    if verify.returncode != 0:
        with lock:
            print(f"  [FAIL] Cannot verify remote file size", flush=True)
        return False

    remote_size = int(verify.stdout.strip())
    if remote_size != file_size:
        with lock:
            print(f"  [FAIL] Size mismatch! local={file_size} remote={remote_size}", flush=True)
        # 깨진 파일 삭제
        subprocess.run(
            ["ssh"] + SSH_OPTS + [f"{REMOTE_USER}@{INTERFACES[0]}", f"rm -f '{remote_path}'"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return False

    with lock:
        print(f"  [DONE] {rel_path} verified ({fmt_size(file_size)})", flush=True)
    return True


# =================================================================
# main
# =================================================================
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)

    src_dir = sys.argv[1]
    if not os.path.isdir(src_dir):
        print(f"Error: '{src_dir}' is not a directory")
        sys.exit(1)

    src_abs = os.path.abspath(src_dir)
    # Mirror the source's path RELATIVE TO REMOTE_BASE so nested layouts (e.g. Models/Org/Model)
    # are reproduced on the remote instead of being flattened to just the leaf dir name.
    real_base = os.path.realpath(REMOTE_BASE)
    real_src  = os.path.realpath(src_abs)
    if real_src == real_base or real_src.startswith(real_base + "/"):
        dir_name = os.path.relpath(real_src, real_base)   # e.g. "Org/Model"
    else:
        dir_name = os.path.basename(src_abs)              # outside Models: fall back to leaf name
    remote_base_dir = f"{REMOTE_BASE}/{dir_name}"
    print(f"Remote target: {remote_base_dir}", flush=True)

    # 1) 파일 수집
    print("Collecting local files...", flush=True)
    all_files = collect_files(src_dir)
    total = len(all_files)
    if total == 0:
        print("No files found.")
        sys.exit(0)

    # 2) 원격 크기 비교
    print(f"Fetching remote file sizes...", flush=True)
    size_cmd = ["ssh"] + SSH_OPTS + [
        f"{REMOTE_USER}@{INTERFACES[0]}",
        f"find {remote_base_dir} -type f ! -name '*.part[0-9]*' -printf '%s %P\\n' 2>/dev/null"
    ]
    size_result = subprocess.run(size_cmd, capture_output=True, text=True)
    remote_sizes = {}
    if size_result.returncode == 0:
        for line in size_result.stdout.strip().splitlines():
            if not line:
                continue
            s, p = line.split(None, 1)
            remote_sizes[p] = int(s)

    # 잔여 .part 파일 경고
    part_cmd = ["ssh"] + SSH_OPTS + [
        f"{REMOTE_USER}@{INTERFACES[0]}",
        f"find {remote_base_dir} -name '*.part[0-9]*' -type f 2>/dev/null | head -5"
    ]
    part_result = subprocess.run(part_cmd, capture_output=True, text=True)
    if part_result.returncode == 0 and part_result.stdout.strip():
        print(f"  WARNING: leftover .part files found on remote (previous failed transfer):", flush=True)
        for line in part_result.stdout.strip().splitlines():
            print(f"    {line}", flush=True)

    to_send = []
    need_check = []
    for rel in all_files:
        local_size = os.path.getsize(os.path.join(src_abs, rel))
        if rel not in remote_sizes:
            to_send.append(rel)
        elif remote_sizes[rel] != local_size:
            to_send.append(rel)
        else:
            need_check.append(rel)

    print(f"  New/size mismatch: {len(to_send)} -> send", flush=True)
    print(f"  Size match:        {len(need_check)} -> check", flush=True)

    # 3) 크기 같은 파일: 앞/중간/뒤 512B 비교
    skipped = 0
    if need_check:
        print(f"Checking head/mid/tail {CHECK_BYTES}B for {len(need_check)} files...", flush=True)

        remote_info = {}
        def fetch_remote():
            nonlocal remote_info
            remote_info = get_remote_info(INTERFACES[0], remote_base_dir, need_check)

        remote_thread = threading.Thread(target=fetch_remote)
        remote_thread.start()
        local_info = get_local_info(src_abs, need_check)
        remote_thread.join()

        for rel in need_check:
            local = local_info[rel]
            remote = remote_info.get(rel)
            if remote and local == remote:
                skipped += 1
            else:
                to_send.append(rel)

    send_total = len(to_send)
    print(f"\n  Total: {total}, Skip: {skipped}, To send: {send_total}", flush=True)

    if send_total == 0:
        print("All files already synced. Nothing to do.")
        sys.exit(0)

    # 4) 디렉토리 생성
    needed_dirs = set(".")
    for rel in to_send:
        d = os.path.dirname(rel)
        if d:
            needed_dirs.add(d)
    print(f"Creating {len(needed_dirs)} remote directories...", flush=True)
    create_remote_dirs(INTERFACES[0], remote_base_dir, needed_dirs)

    # 5) 대형/소형 분류
    large_files = []
    small_files = []
    for rel in to_send:
        size = os.path.getsize(os.path.join(src_abs, rel))
        if size >= SPLIT_THRESHOLD:
            large_files.append(rel)
        else:
            small_files.append(rel)

    print(f"\n  Small files (< {fmt_size(SPLIT_THRESHOLD)}): {len(small_files)}")
    print(f"  Large files (>= {fmt_size(SPLIT_THRESHOLD)}): {len(large_files)} -> split transfer", flush=True)

    # 6) 소형 파일: 라운드 로빈
    lock = threading.Lock()
    results = []

    if small_files:
        small_total = len(small_files)
        iface_queues = defaultdict(list)
        for i, rel in enumerate(small_files):
            iface = INTERFACES[i % len(INTERFACES)]
            iface_queues[iface].append((i + 1, rel))

        print(f"\n--- Small files ---")
        for iface in INTERFACES:
            count = len(iface_queues[iface])
            if count:
                print(f"  {iface}: {count} files")
        print(flush=True)

        threads = []
        for iface in INTERFACES:
            if not iface_queues[iface]:
                continue
            t = threading.Thread(
                target=worker,
                args=(iface, iface_queues[iface], src_abs, remote_base_dir, small_total, results, lock),
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    # 7) 대형 파일: 분할 전송 (하나씩, 각각 모든 인터페이스 사용)
    large_results = []
    if large_files:
        print(f"\n--- Large files (split transfer) ---", flush=True)
        for i, rel in enumerate(large_files):
            ok = send_large_file(rel, src_abs, remote_base_dir, i + 1, len(large_files), lock)
            large_results.append((rel, ok))

    # 결과 요약
    small_ok = sum(1 for _, _, s in results if s)
    small_fail = sum(1 for _, _, s in results if not s)
    large_ok = sum(1 for _, ok in large_results if ok)
    large_fail = sum(1 for _, ok in large_results if not ok)

    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}")
    print(f"  Skipped:      {skipped}")
    print(f"  Small sent:   {small_ok} OK, {small_fail} FAIL")
    print(f"  Large sent:   {large_ok} OK, {large_fail} FAIL")
    print(f"{'='*50}")

    if small_fail or large_fail:
        print("\nFailed files can be retried by running again.")
        sys.exit(1)
    else:
        print("\nAll files synced successfully.")


if __name__ == "__main__":
    main()
