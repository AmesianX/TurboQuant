#!/usr/bin/env python3
"""
Multi-turn KV-cache reuse judge for the DSV4 chat server.

Replays a llama.cpp web-UI conversation export turn-by-turn through
/v1/chat/completions, rebuilding the full message history each turn exactly as
the web UI does, and reports per-turn prompt_n / cache_n / prompt_ms.

PASS  = cache_n grows with the conversation (prior context reused) and prompt_n
        stays ~= just the new user message (prompt_ms stays low).
FAIL  = cache_n stuck near 1 and prompt_n == full growing context (re-prefill).

Usage:
  python3 cachetest_replay.py <conv.json> [--base http://127.0.0.1:8080] [--key tbq-dsv4]

The conv file may be a web-UI export ({"conv":..., "messages":[...tree...]}) or a
simple {"messages":[{"role","content"},...]} list. Assistant contents from the
file are used verbatim so the rendered prompt is byte-identical to the user's.
"""
import json, sys, time, argparse, urllib.request

def load_linear(path):
    d = json.load(open(path))
    msgs = d["messages"] if isinstance(d, dict) else d
    # simple flat list of {role, content}
    if msgs and "id" not in msgs[0]:
        return [{"role": m["role"], "content": m["content"]} for m in msgs]
    # web-UI tree: walk from currNode up via parent
    by_id = {m["id"]: m for m in msgs}
    leaf = d.get("conv", {}).get("currNode") or msgs[-1]["id"]
    chain = []
    cur = leaf
    while cur in by_id:
        m = by_id[cur]
        chain.append({"role": m["role"], "content": m["content"]})
        cur = m.get("parent")
    chain.reverse()
    return chain

def post(base, key, messages):
    body = json.dumps({"model": "x", "messages": messages,
                       "max_tokens": 1, "temperature": 0,
                       "cache_prompt": True}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conv")
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="tbq-dsv4")
    a = ap.parse_args()

    chain = load_linear(a.conv)
    # turns = indices of user messages, in order
    user_idx = [i for i, m in enumerate(chain) if m["role"] == "user"]
    print(f"conversation: {len(chain)} msgs, {len(user_idx)} user turns  ->  {a.base}\n")
    print(f"{'turn':>4} {'new_usr':>8} {'prompt_n':>9} {'cache_n':>8} {'prompt_ms':>10} {'t/s':>7}  reuse")
    print("-" * 64)

    rows = []
    for t, ui in enumerate(user_idx, 1):
        msgs = [{"role": m["role"], "content": m["content"]} for m in chain[:ui + 1]]
        new_usr = len(chain[ui]["content"])
        try:
            resp = post(a.base, a.key, msgs)
        except Exception as e:
            print(f"{t:>4}  ERROR: {str(e)[:50]}")
            continue
        tm = resp.get("timings", {}) or {}
        pn = int(tm.get("prompt_n", -1)); cn = int(tm.get("cache_n", -1))
        pms = float(tm.get("prompt_ms", 0)); tps = tm.get("prompt_per_second", 0) or 0
        reuse = "YES" if (cn > 2 and pn < max(8, cn)) else ("--" if t == 1 else "NO")
        print(f"{t:>4} {new_usr:>8} {pn:>9} {cn:>8} {pms:>10.0f} {tps:>7.1f}  {reuse}")
        rows.append((t, pn, cn, pms))
        time.sleep(0.3)

    # verdict: after turn 1, every turn should reuse (cache_n grows, prompt_n small)
    judged = rows[1:]
    reused = [r for r in judged if r[2] > 2 and r[1] < max(8, r[2])]
    worst_ms = max((r[3] for r in rows), default=0)
    print("-" * 64)
    print(f"reused turns: {len(reused)}/{len(judged)}   worst prompt_ms: {worst_ms:.0f}")
    ok = len(judged) > 0 and len(reused) == len(judged)
    print("VERDICT:", "PASS  (KV cache reused across turns)" if ok
          else "FAIL  (re-prefilling each turn -- cache not reused)")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
