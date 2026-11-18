"""
Generate LLM responses for several conditions in parallel, under a hard dollar budget and a hard deadline.
No judging happens here: responses are appended to results/<condition>/llm_responses.jsonl, and benchmark.py
judges them later from that cache (with LCB_NO_API=1 it makes no API calls).

Tasks are ordered problem-major (all conditions of problem 1, then problem 2, ...), so whatever is finished
when the budget or time runs out consists mostly of complete sets.

Usage:
    python generate.py --budget 140 --launch-cutoff 00:35 --hard-stop 00:50 --concurrency 60
"""
import argparse
import datetime as dt
import json
import logging
import os
import queue
import threading
import time

from datasets import load_dataset
from huggingface_hub import HfApi

import api_interface
from judge import LightCPVerifierJudge

CONDITIONS = {
    "baseline": None,
    "sr_compact": "prompts/sr_compact_v1.txt",
    "sr_explicit": "prompts/sr_explicit_v1.txt",
}
# Newest first, to keep problems as recent as possible. quater_2025_7_9 has no test data.
SPLIT_ORDER = ["quater_2025_4_6", "quater_2025_1_3", "quater_2024_10_12", "biannual_2024_7_12", "biannual_2025_1_6"]

# Claude Sonnet 4.5 pricing (USD per token), prompts under 200K tokens.
PRICE_IN, PRICE_OUT = 3e-6, 15e-6
WORST_CASE_CALL = 2_000 * PRICE_IN + api_interface.ClaudeSonnet45.MAX_TOKENS * PRICE_OUT  # ~$0.97

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("generate")


def at_today_or_tomorrow(hhmm: str) -> dt.datetime:
    now = dt.datetime.now()
    t = dt.datetime.combine(now.date(), dt.time.fromisoformat(hhmm))
    return t if t > now else t + dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, required=True, help="hard ceiling in USD, counting in-flight calls at worst case")
    ap.add_argument("--launch-cutoff", required=True, help="HH:MM, no new calls start after this")
    ap.add_argument("--hard-stop", required=True, help="HH:MM, the process exits at this time no matter what")
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--difficulty", default="easy")
    args = ap.parse_args()

    launch_cutoff = at_today_or_tomorrow(args.launch_cutoff)
    hard_stop = at_today_or_tomorrow(args.hard_stop)
    assert launch_cutoff < hard_stop

    # Hard stop: exit the whole process (cancelling in-flight requests) at the deadline.
    def killer():
        time.sleep(max(0, (hard_stop - dt.datetime.now()).total_seconds()))
        log.error("HARD STOP reached, exiting now.")
        os._exit(2)
    threading.Thread(target=killer, daemon=True).start()

    judgeable = {f[:-4] for f in HfApi().list_repo_files(LightCPVerifierJudge.DATASET_HF_REPO, repo_type="dataset") if f.endswith(".zip")}
    dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro")
    # --difficulty may list several levels (e.g. "medium,hard"); problems then alternate between them,
    # so whatever finishes before the budget or deadline is split evenly across the levels.
    levels = [d.strip() for d in args.difficulty.split(",")]
    per_level, seen = {d: [] for d in levels}, set()
    for split in SPLIT_ORDER:
        for row in dataset[split]:
            pid = row["problem_id"]
            if row["difficulty"] in per_level and pid in judgeable and pid not in seen:
                seen.add(pid)
                per_level[row["difficulty"]].append((pid, row["problem_statement"]))
    problems = []
    for i in range(max(len(v) for v in per_level.values())):
        for d in levels:
            if i < len(per_level[d]):
                problems.append(per_level[d][i])
    log.info(f"{len(problems)} problems with test data: " + ", ".join(f"{len(v)} {d}" for d, v in per_level.items()))

    llms, cached = {}, {}
    for cond, path in CONDITIONS.items():
        os.makedirs(os.path.join("results", cond), exist_ok=True)
        llms[cond] = api_interface.ClaudeSonnet45(system_prompt=open(path).read().strip() if path else None)
        cache_path = os.path.join("results", cond, os.environ.get("LCB_CACHE_NAME", "llm_responses.jsonl"))
        cached[cond] = set()
        # Skip problems cached in this run's file or in the main response file.
        for path in {cache_path} | {os.path.join("results", cond, f) for f in os.listdir(os.path.join("results", cond)) if f.startswith("llm_responses") and f.endswith(".jsonl")}:
            if os.path.exists(path):
                cached[cond] |= {json.loads(l)["problem_id"] for l in open(path)}

    tasks = queue.Queue()
    for pid, stmt in problems:
        for cond in CONDITIONS:
            if pid not in cached[cond]:
                tasks.put((pid, stmt, cond))
    log.info(f"{tasks.qsize()} calls to make; budget ${args.budget}, launch cutoff {launch_cutoff:%H:%M}, hard stop {hard_stop:%H:%M}")

    lock = threading.Lock()
    state = {"spent": 0.0, "in_flight": 0, "done": 0, "failed": 0, "stop": False}
    status_path = os.path.join("results", os.environ.get("LCB_STATUS_NAME", "generate_status.json"))

    def write_status():
        with open(status_path, "w") as f:
            json.dump({**state, "time": f"{dt.datetime.now():%H:%M:%S}", "remaining_tasks": tasks.qsize()}, f)

    def reserve() -> bool:
        """Wait until a new call fits in the budget; False means stop launching."""
        while True:
            with lock:
                if state["stop"] or dt.datetime.now() >= launch_cutoff:
                    state["stop"] = True
                    return False
                if state["spent"] + (state["in_flight"] + 1) * WORST_CASE_CALL <= args.budget:
                    state["in_flight"] += 1
                    return True
                if state["in_flight"] == 0:
                    state["stop"] = True
                    return False
            time.sleep(2)

    def worker():
        while True:
            try:
                pid, stmt, cond = tasks.get_nowait()
            except queue.Empty:
                return
            if not reserve():
                return
            cost = WORST_CASE_CALL  # if the call errors we cannot see its usage; count it at worst case
            try:
                resp, meta = llms[cond].generate_solution(stmt)
                u = meta["usage"]
                cost = (u["input_tokens"] + (u.get("cache_creation_input_tokens") or 0)) * PRICE_IN + u["output_tokens"] * PRICE_OUT
                meta["cost_usd"] = round(cost, 4)
                with lock:
                    with open(os.path.join("results", cond, os.environ.get("LCB_CACHE_NAME", "llm_responses.jsonl")), "a") as f:
                        f.write(json.dumps({"problem_id": pid, "text_response": resp, "response_meta": meta}) + "\n")
                    state["done"] += 1
                log.info(f"{pid} {cond}: {u['output_tokens']} out tokens, ${cost:.3f}, {meta['latency_s']}s, {meta['stop_reason']}")
            except Exception as e:
                with lock:
                    state["failed"] += 1
                log.error(f"{pid} {cond} failed: {type(e).__name__}: {e}")
            finally:
                with lock:
                    state["in_flight"] -= 1
                    state["spent"] += cost
                    write_status()

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        with lock:
            write_status()
            log.info(f"STATUS spent=${state['spent']:.2f} in_flight={state['in_flight']} done={state['done']} failed={state['failed']} queued={tasks.qsize()}")
        time.sleep(60)
    write_status()
    log.info(f"FINISHED spent=${state['spent']:.2f} done={state['done']} failed={state['failed']}")


if __name__ == "__main__":
    main()
