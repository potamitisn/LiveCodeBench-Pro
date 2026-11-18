"""
Compare conditions on the problems that every condition has a judged result for.

    python compare.py            # prints the comparison and the list of complete problem ids
    python compare.py --ids      # prints only the comma-separated ids that have a cached response in every condition
"""
import json
import os
import sys
from collections import Counter

CONDITIONS = ["baseline", "sr_compact", "sr_explicit"]


def cached_ids(cond):
    path = os.path.join("results", cond, "llm_responses.jsonl")
    return {json.loads(l)["problem_id"] for l in open(path)} if os.path.exists(path) else set()


def main():
    complete = set.intersection(*(cached_ids(c) for c in CONDITIONS))
    if "--ids" in sys.argv:
        print(",".join(sorted(complete)))
        return

    results = {}
    for c in CONDITIONS:
        rows = json.load(open(os.path.join("results", c, "benchmark_result.json")))
        results[c] = {r["problem_id"]: r for r in rows}
    ids = sorted(complete & set.intersection(*(set(results[c]) for c in CONDITIONS)))
    print(f"Problems with a judged result in all {len(CONDITIONS)} conditions: {len(ids)}\n")

    print(f"{'condition':12s} {'accepted':>10s} {'rate':>7s} {'mean out tok':>13s} {'mean think tok':>15s} {'cost $':>8s}  verdicts")
    for c in CONDITIONS:
        rs = [results[c][p] for p in ids]
        acc = sum(r["judge_result"] == "Accepted" for r in rs)
        usage = [r["response_meta"]["usage"] for r in rs]
        out = sum(u["output_tokens"] for u in usage) / len(rs)
        think = sum((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0 for u in usage) / len(rs)
        cost = sum(u["input_tokens"] * 3e-6 + u["output_tokens"] * 15e-6 for u in usage)
        verdicts = dict(Counter(r["judge_result"] for r in rs).most_common())
        print(f"{c:12s} {acc:5d}/{len(rs):<4d} {acc / len(rs):7.1%} {out:13.0f} {think:15.0f} {cost:8.2f}  {verdicts}")

    print("\nPer-problem agreement with baseline:")
    base = {p: results["baseline"][p]["judge_result"] == "Accepted" for p in ids}
    for c in CONDITIONS[1:]:
        cur = {p: results[c][p]["judge_result"] == "Accepted" for p in ids}
        only_base = [p for p in ids if base[p] and not cur[p]]
        only_cur = [p for p in ids if cur[p] and not base[p]]
        both = sum(base[p] and cur[p] for p in ids)
        print(f"  {c}: both solved {both}, only baseline {len(only_base)} {only_base}, only {c} {len(only_cur)} {only_cur}")


if __name__ == "__main__":
    main()
