#!/usr/bin/env python3
"""
TurboQuant Math Accuracy Test (llama-server compatible)

Based on eullm/eullm bench/turboquant_math_accuracy.py by primoco.
Modified for llama-server OpenAI-compatible API (/v1/chat/completions).

Usage:
  python math_bench.py collect --url http://127.0.0.1:8889 --label tbq3_sub32 --filler 200,500,1000
  python math_bench.py collect --url http://127.0.0.1:8889 --label tbq3_sub32 --filler 0
  python math_bench.py compare results1.json results2.json
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)


# ── Filler ────────────────────────────────────────────────────────────────────

FILLER_PARAGRAPHS = [
    "The European Union's digital strategy aims to make this transformation work for people and businesses, while helping to achieve its target of a climate-neutral Europe by 2050. The Commission has outlined key policies and frameworks to guide this transition.",
    "Recent advances in semiconductor manufacturing have enabled the production of chips at the 3nm node, dramatically increasing transistor density while reducing power consumption. TSMC, Samsung, and Intel are competing to deliver these next-generation processors.",
    "The Mediterranean basin is home to approximately 25,000 plant species, of which about 50% are endemic. This biodiversity hotspot faces threats from urbanisation, agricultural intensification, and climate change.",
    "Cloud computing has fundamentally changed how organisations deploy and manage IT infrastructure. The shift from capital expenditure to operational expenditure models has enabled startups and enterprises alike to scale their operations globally.",
    "Italian Renaissance art underwent several distinct phases, from the early experiments of Giotto in the 13th century to the High Renaissance works of Leonardo, Michelangelo, and Raphael.",
    "The carbon cycle involves the exchange of carbon between the atmosphere, oceans, terrestrial biosphere, and geological formations. Anthropogenic emissions have disrupted this cycle, leading to an increase in atmospheric CO2 concentrations.",
    "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations that would be impractical for classical computers.",
    "The logistics industry handles approximately 65 billion parcels annually worldwide, with e-commerce driving a significant portion of this volume. Last-mile delivery remains the most expensive segment.",
    "Volcanic activity along tectonic plate boundaries shapes the Earth's surface through both constructive and destructive processes. The Ring of Fire contains approximately 75% of the world's active volcanoes.",
    "Machine learning models for natural language processing have grown exponentially in size. This scaling has been accompanied by emergent capabilities including in-context learning and chain-of-thought reasoning.",
]


def generate_filler(target_tokens: int) -> str:
    target_chars = target_tokens * 4
    paragraphs = []
    total = 0
    while total < target_chars:
        p = random.choice(FILLER_PARAGRAPHS)
        paragraphs.append(p)
        total += len(p)
    return "\n\n".join(paragraphs)


# ── Answer checking ───────────────────────────────────────────────────────────

def strip_thinking(s: str) -> str:
    return re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()


def normalize(s: str) -> str:
    s = strip_thinking(s)
    s = s.replace(" ", "").replace("\n", "").replace("`", "").replace("*", "")
    return s.strip().lower()


def extract_last_line(s: str) -> str:
    s = strip_thinking(s)
    lines = [l.strip() for l in s.strip().split('\n') if l.strip()]
    return lines[-1] if lines else s


def _extract_matrix_from_latex(s: str) -> str:
    s = re.sub(r'\\boxed\{(.*?)\}', r'\1', s, flags=re.DOTALL)
    m = re.findall(
        r'\\begin\{[pvb]?matrix\*?\}(.*?)\\end\{[pvb]?matrix\*?\}',
        s, re.DOTALL
    )
    if not m:
        return ""
    inner = m[-1]
    rows = [r.strip() for r in re.split(r'\\\\', inner) if r.strip()]
    result = []
    for row in rows:
        nums = [int(x) for x in re.findall(r'-?\d+', row)]
        if nums:
            result.append(nums)
    if not result:
        return ""
    return "[" + ",".join("[" + ",".join(str(x) for x in r) + "]" for r in result) + "]"


def check_answer(test: dict, response: str) -> tuple:
    mode = test["check"]
    resp = strip_thinking(response).strip()
    resp_last = extract_last_line(response)

    if mode == "exact_normalized":
        expected = normalize(test["expected"])
        if expected in normalize(resp) or expected in normalize(resp_last):
            return True, f"expected={test['expected']} last_line={resp_last[:80]}"
        latex_result = _extract_matrix_from_latex(resp)
        if latex_result and normalize(latex_result) == expected:
            return True, f"expected={test['expected']} latex={latex_result}"
        return False, f"expected={test['expected']} last_line={resp_last[:80]}"

    elif mode == "contains_number":
        expected = test["expected"]
        boxed = re.findall(r'\\boxed\{([^}]+)\}', resp)
        boxed_str = " ".join(boxed)
        ok = expected in resp or expected in resp_last or expected in boxed_str
        return ok, f"expected={expected} in response={resp_last[:80]}"

    return False, "unknown check mode"


# ── Matrix helpers ────────────────────────────────────────────────────────────

def _mm(A, B):
    rows, cols, inner = len(A), len(B[0]), len(B)
    return [[sum(A[r][k] * B[k][c] for k in range(inner)) for c in range(cols)]
            for r in range(rows)]


def _fmt(rows):
    return "[" + ",".join("[" + ",".join(str(x) for x in r) + "]" for r in rows) + "]"


# ── Test data (seeded generators) ────────────────────────────────────────────

def _gen_matrix(rng, n, lo, hi):
    return [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]


def gen_matrix_2x2(seed):
    rng = random.Random(seed ^ 0xA5A5_2222)
    out = []
    for i in range(10):
        A = _gen_matrix(rng, 2, 0, 9)
        B = _gen_matrix(rng, 2, 0, 9)
        out.append((f"E{i+1:02d}", A, B))
    return out


def gen_matrix_3x3(seed):
    rng = random.Random(seed ^ 0xA5A5_3333)
    out = []
    for i in range(10):
        A = _gen_matrix(rng, 3, 0, 9)
        B = _gen_matrix(rng, 3, 0, 9)
        out.append((f"M{i+1:02d}", A, B))
    return out


def gen_scalar(seed):
    rng = random.Random(seed ^ 0xA5A5_5555)
    out = []
    # 5 easy: a*b + c
    for i in range(5):
        a = rng.randint(11, 29); b = rng.randint(11, 29); c = rng.randint(5, 99)
        expr = f"({a} * {b}) + {c}"
        out.append((f"S{i+1:02d}", expr, str(a*b + c)))
    # 5 medium: a*b - c*d
    for i in range(5, 10):
        a = rng.randint(12, 40); b = rng.randint(12, 40)
        c = rng.randint(3, 20); d = rng.randint(20, 60)
        expr = f"{a} * {b} - {c} * {d}"
        out.append((f"S{i+1:02d}", expr, str(a*b - c*d)))
    # 5 hard: a^p - b^q + c^r
    for i in range(10, 15):
        a = rng.randint(2, 9); p = rng.randint(3, 5)
        b = rng.randint(2, 9); q = rng.randint(3, 5)
        c = rng.randint(2, 9); r = rng.randint(3, 5)
        expr = f"{a}^{p} - {b}^{q} + {c}^{r}"
        out.append((f"S{i+1:02d}", expr, str(a**p - b**q + c**r)))
    return out


def build_tests(filler_levels, skip_3x3=False, skip_scalar=False, seed=0):
    tests = []
    MATRIX_2X2 = gen_matrix_2x2(seed)
    MATRIX_3X3 = gen_matrix_3x3(seed)
    SCALAR = gen_scalar(seed)

    for label, A, B in MATRIX_2X2:
        C = _mm(A, B)
        tests.append({
            "id": f"direct_mat2x2_{label}", "category": "direct_math",
            "subtype": "matrix_2x2", "filler_tokens": 0,
            "prompt": f"Compute {_fmt(A)} × {_fmt(B)}. Return ONLY [[a,b],[c,d]].",
            "expected": _fmt(C), "check": "exact_normalized",
        })

    if not skip_3x3:
        for label, A, B in MATRIX_3X3:
            C = _mm(A, B)
            tests.append({
                "id": f"direct_mat3x3_{label}", "category": "direct_math",
                "subtype": "matrix_3x3", "filler_tokens": 0,
                "prompt": f"Compute {_fmt(A)} × {_fmt(B)}. Return ONLY [[a,b,c],[d,e,f],[g,h,i]].",
                "expected": _fmt(C), "check": "exact_normalized",
            })

    if not skip_scalar:
        for label, expr, expected in SCALAR:
            tests.append({
                "id": f"direct_scalar_{label}", "category": "direct_math",
                "subtype": "scalar", "filler_tokens": 0,
                "prompt": f"Compute {expr}. Return ONLY the integer result.",
                "expected": expected, "check": "contains_number",
            })

    for label, A, B in MATRIX_2X2:
        C = _mm(A, B)
        for fl in filler_levels:
            filler = generate_filler(fl)
            tests.append({
                "id": f"delayed_mat2x2_{label}_{fl}t", "category": "delayed_math",
                "subtype": "matrix_2x2", "filler_tokens": fl,
                "prompt": (
                    f"Memorize these matrices. Do NOT compute yet.\n\n"
                    f"A = {_fmt(A)}\nB = {_fmt(B)}\n\n"
                    f"{filler}\n\n"
                    f"Compute A × B. Return ONLY [[a,b],[c,d]], no explanation."
                ),
                "expected": _fmt(C), "check": "exact_normalized",
            })

    if not skip_3x3:
        for label, A, B in MATRIX_3X3:
            C = _mm(A, B)
            for fl in filler_levels:
                filler = generate_filler(fl)
                tests.append({
                    "id": f"delayed_mat3x3_{label}_{fl}t", "category": "delayed_math",
                    "subtype": "matrix_3x3", "filler_tokens": fl,
                    "prompt": (
                        f"Memorize these matrices. Do NOT compute yet.\n\n"
                        f"A = {_fmt(A)}\nB = {_fmt(B)}\n\n"
                        f"{filler}\n\n"
                        f"Compute A × B. Return ONLY [[a,b,c],[d,e,f],[g,h,i]], no explanation."
                    ),
                    "expected": _fmt(C), "check": "exact_normalized",
                })

    if not skip_scalar:
        for label, expr, expected in SCALAR:
            for fl in filler_levels:
                filler = generate_filler(fl)
                tests.append({
                    "id": f"delayed_scalar_{label}_{fl}t", "category": "delayed_math",
                    "subtype": "scalar", "filler_tokens": fl,
                    "prompt": (
                        f"Remember this expression. Do NOT compute it yet.\n\n"
                        f"EXPRESSION: {expr}\n\n"
                        f"{filler}\n\n"
                        f"Compute the expression you memorized. Return ONLY the integer result."
                    ),
                    "expected": expected, "check": "contains_number",
                })

    return tests


# ── API client (llama-server OpenAI-compatible) ──────────────────────────────

async def send_prompt(session: aiohttp.ClientSession, url: str, model: str,
                      prompt: str, temperature: float = 0.0,
                      num_predict: int = 2048, api_key: str = "") -> tuple:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": num_predict,
        "temperature": temperature,
        "stream": False,
    }
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return f"[ERROR: HTTP {resp.status} {body[:100]}]", {}
            data = await resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "[no content]")
            usage = data.get("usage", {})
            timing = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
            return content, timing
    except Exception as e:
        return f"[ERROR: {e}]", {}


# ── Collect ───────────────────────────────────────────────────────────────────

async def collect(args):
    random.seed(args.seed)
    filler_levels = [int(x) for x in args.filler.split(",") if x and int(x) > 0]
    tests = build_tests(filler_levels, skip_3x3=args.skip_3x3, skip_scalar=args.skip_scalar, seed=args.seed)

    print(f"TurboQuant Math Accuracy Test (llama-server)")
    print(f"  URL:   {args.url}")
    print(f"  Label: {args.label}")
    print(f"  Tests: {len(tests)}  filler={filler_levels}")
    print()

    results = []
    cats = {}
    run_start = time.time()

    async with aiohttp.ClientSession() as session:
        for test in tests:
            t0 = time.time()
            response, timing = await send_prompt(session, args.url, "gemma4",
                                                 test["prompt"], args.temperature,
                                                 args.num_predict, args.api_key)
            latency_s = time.time() - t0
            passed, detail = check_answer(test, response)

            status = "PASS" if passed else "FAIL"
            cat = test["category"]
            cats.setdefault(cat, {"pass": 0, "total": 0})
            cats[cat]["total"] += 1
            if passed:
                cats[cat]["pass"] += 1

            fl = test["filler_tokens"]
            fl_s = f"filler={fl:>5}t" if fl else "direct       "
            ptoks = timing.get("prompt_tokens", 0)
            print(f"  [{status}] {test['id']:<42} {fl_s} ctx={ptoks:>5}t  {detail[:50]}")

            results.append({
                "id": test["id"], "category": cat, "subtype": test["subtype"],
                "filler_tokens": fl, "passed": passed, "expected": test["expected"],
                "detail": detail, "response_preview": strip_thinking(response)[:300],
                "latency_s": latency_s, "prompt_tokens": ptoks,
                "completion_tokens": timing.get("completion_tokens", 0),
            })

    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)
    pct = total_pass / total * 100 if total else 0

    print(f"\n{'='*65}")
    print(f"  {args.label}: {total_pass}/{total} passed ({pct:.1f}%)")
    print(f"{'='*65}")

    print(f"\n  Category breakdown:")
    for cat, s in sorted(cats.items()):
        p = s['pass'] / s['total'] * 100 if s['total'] else 0
        print(f"    {cat:<22} {s['pass']:>4}/{s['total']:<4}  {p:>5.1f}%")

    total_wall = time.time() - run_start
    print(f"\n  Wall time: {total_wall:.1f}s")

    output = {
        "label": args.label, "url": args.url,
        "temperature": args.temperature, "filler_levels": filler_levels,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "score": {"passed": total_pass, "total": total, "percent": pct},
        "categories": cats, "results": results,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {args.output}")


# ── Compare ───────────────────────────────────────────────────────────────────

def compare(args):
    data = []
    for f in args.files:
        with open(f) as fh:
            data.append(json.load(fh))

    print(f"\n{'='*65}")
    print(f"  TurboQuant Math Accuracy Comparison")
    print(f"{'='*65}\n")
    print(f"  {'Label':<20} {'Pass/Total':>12}   {'%':>6}")
    print(f"  {'-'*42}")
    for d in data:
        s = d["score"]
        print(f"  {d['label']:<20} {s['passed']:>4}/{s['total']:<4}        {s['percent']:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Math Accuracy Test")
    sub = parser.add_subparsers(dest="command")

    c = sub.add_parser("collect")
    c.add_argument("--url", default="http://127.0.0.1:8889")
    c.add_argument("--label", required=True)
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--filler", default="200,500,1000")
    c.add_argument("--skip-3x3", action="store_true")
    c.add_argument("--skip-scalar", action="store_true")
    c.add_argument("--num-predict", type=int, default=2048)
    c.add_argument("--api-key", default="")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--output", "-o")
    c.add_argument("--verbose", "-v", action="store_true")

    p = sub.add_parser("compare")
    p.add_argument("files", nargs="+")

    args = parser.parse_args()
    if args.command == "collect":
        asyncio.run(collect(args))
    elif args.command == "compare":
        compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
