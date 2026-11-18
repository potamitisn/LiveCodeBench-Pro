from datasets import load_dataset, DatasetDict
import pydantic
import typing
import api_interface
import json
import tqdm
import time
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from judge import LightCPVerifierJudge, SupportedLanguage, ProblemNotFoundError
from util import extract_longest_cpp_code
from huggingface_hub import HfApi

# *************************** Change this before use ****************************

# Experimental condition, chosen with the LCB_CONDITION environment variable.
# Each condition differs ONLY in the system prompt; the user prompt, model and thinking settings are identical.
#   baseline     - no system prompt (the original benchmark behavior)
#   sr_compact   - structured-reasoning instructions from prompts/sr_compact_v1.txt
#   sr_explicit  - structured-reasoning instructions from prompts/sr_explicit_v1.txt
CONDITIONS = {
    "baseline": None,
    "sr_compact": "prompts/sr_compact_v1.txt",
    "sr_explicit": "prompts/sr_explicit_v1.txt",
}
condition = os.environ.get("LCB_CONDITION", "baseline")
if condition not in CONDITIONS:
    raise SystemExit(f"Unknown LCB_CONDITION '{condition}'. Choose one of: {', '.join(CONDITIONS)}")
system_prompt_path = CONDITIONS[condition]
system_prompt = open(system_prompt_path).read().strip() if system_prompt_path else None

llm_instance = (
    api_interface.ClaudeSonnet45(system_prompt=system_prompt)
)  # change this to the LLM class you want to benchmark on

# change this to the number of workers you want to use in LightCPVerifier
# recommended to be <= number of CPU physical cores
worker = 4

# number of problems sent to the LLM at the same time
llm_concurrency = 12

# Every condition writes to its own folder, so runs never share cached responses or results.
output_dir = os.path.join("results", condition)
os.makedirs(output_dir, exist_ok=True)
# LLM responses are appended here as they arrive; rerunning skips problems already in it
response_cache_path = os.path.join(output_dir, "llm_responses.jsonl")
result_path = os.path.join(output_dir, "benchmark_result.json")

# *******************************************************************************


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkResult(pydantic.BaseModel):
    problem_id: str
    problem_title: str
    difficulty: str
    platform: str
    text_response: str
    code: str | None
    judge_result: str
    response_meta: typing.Any

class ProblemTestState(pydantic.BaseModel):
    problem_id: str
    problem_title: str
    difficulty: str
    platform: str
    problem_statement: str
    text_response: str | None = None
    code: str | None = None
    submission_id: int | None = None
    judge_result: str = "Judging"
    response_meta: typing.Any = None

def get_problem_set(dataset: DatasetDict) -> dict[str, ProblemTestState]:
    problem_set = {}
    for split in dataset.values():
        for row in split:
            if row["problem_id"] not in problem_set:
                problem_set[row["problem_id"]] = ProblemTestState(**row)
    return problem_set

def print_stats(dataset: DatasetDict, problem_set: dict[str, ProblemTestState]):
    print("=" * 80)
    print("BENCHMARK STATISTICS")
    print("=" * 80)

    split_difficulty_stats = {}

    for split_name, split in dataset.items():
        split_difficulty_stats[split_name] = {}
        
        for row in split:
            problem_id = row["problem_id"]
            difficulty = row.get("difficulty", "unknown")

            if problem_id in problem_set:
                judge_result = problem_set[problem_id].judge_result
            else:
                judge_result = "Not Tested"

            if difficulty not in split_difficulty_stats[split_name]:
                split_difficulty_stats[split_name][difficulty] = {
                    "total": 0, 
                    "accepted": 0, 
                    "judge_results": {}
                }
            
            split_difficulty_stats[split_name][difficulty]["total"] += 1
            if judge_result == "Accepted":
                split_difficulty_stats[split_name][difficulty]["accepted"] += 1

            if judge_result not in split_difficulty_stats[split_name][difficulty]["judge_results"]:
                split_difficulty_stats[split_name][difficulty]["judge_results"][judge_result] = []
            split_difficulty_stats[split_name][difficulty]["judge_results"][judge_result].append(problem_id)

    for split_name in split_difficulty_stats:
        print(f"\n[SPLIT: {split_name.upper()}]")
        print("-" * 60)
        
        total_problems_in_split = 0
        total_accepted_in_split = 0
        
        for difficulty, stats in sorted(split_difficulty_stats[split_name].items()):
            total = stats["total"]
            accepted = stats["accepted"]
            accuracy = (accepted / total * 100) if total > 0 else 0.0
            
            print(f"\n{difficulty.upper()} Difficulty: {accepted}/{total} ({accuracy:.1f}%)")

            for judge_result, problem_ids in sorted(stats["judge_results"].items()):
                count = len(problem_ids)
                percentage = (count / total * 100) if total > 0 else 0.0
                print(f"  {judge_result:20s}: {count:3d} ({percentage:5.1f}%) - {', '.join(sorted(problem_ids))}")
            
            total_problems_in_split += total
            total_accepted_in_split += accepted

        overall_accuracy = (total_accepted_in_split / total_problems_in_split * 100) if total_problems_in_split > 0 else 0.0
        print(f"\nOVERALL for {split_name}: {total_accepted_in_split}/{total_problems_in_split} ({overall_accuracy:.1f}%)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro")
    problem_set = get_problem_set(dataset)
    if wanted := os.environ.get("LCB_PROBLEMS"):
        wanted_ids = [p.strip() for p in wanted.split(",") if p.strip()]
        missing = [p for p in wanted_ids if p not in problem_set]
        if missing:
            raise SystemExit(f"Problem ids not in dataset: {', '.join(missing)}")
        problem_set = {p: problem_set[p] for p in wanted_ids}
        logger.info(f"LCB_PROBLEMS set: running {len(problem_set)} selected problems.")
    # Skip problems the judge has no test data for, before paying for an LLM call on them.
    judgeable = {
        f.removesuffix(".zip")
        for f in HfApi().list_repo_files(LightCPVerifierJudge.DATASET_HF_REPO, repo_type="dataset")
        if f.endswith(".zip")
    }
    no_test_data = [p for p in problem_set if p not in judgeable]
    if no_test_data:
        logger.warning(f"Skipping {len(no_test_data)} problems without test data: {', '.join(no_test_data)}")
        problem_set = {p: s for p, s in problem_set.items() if p in judgeable}
    if limit := int(os.environ.get("LCB_LIMIT", 0)):
        problem_set = dict(list(problem_set.items())[:limit])
        logger.info(f"LCB_LIMIT set: running only {limit} problems.")

    # Reuse LLM responses from earlier (possibly interrupted) runs so they are not paid for twice.
    cache: dict[str, dict] = {}
    if os.path.exists(response_cache_path):
        with open(response_cache_path) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["problem_id"]] = entry
        logger.info(f"Loaded {len(cache)} cached responses from {response_cache_path}.")
    cache_lock = threading.Lock()

    def generate(problem: ProblemTestState):
        if problem.problem_id in cache:
            entry = cache[problem.problem_id]
            return entry["text_response"], entry["response_meta"]
        if os.environ.get("LCB_NO_API"):
            raise RuntimeError("LCB_NO_API is set and this problem has no cached response")
        response, meta = llm_instance.generate_solution(problem.problem_statement)
        with cache_lock, open(response_cache_path, "a") as f:
            f.write(json.dumps({"problem_id": problem.problem_id, "text_response": response, "response_meta": meta}) + "\n")
        return response, meta

    with LightCPVerifierJudge(worker=worker) as judge:
        def solve_and_submit(problem: ProblemTestState):
            try:
                response, meta = generate(problem)
            except Exception as e:
                logger.error(f"LLM call failed for problem {problem.problem_id}: {e}")
                return
            problem.text_response = response
            problem.code = extract_longest_cpp_code(response)
            problem.response_meta = meta
            if problem.code:
                try:
                    problem.submission_id = judge.submit(problem.problem_id, SupportedLanguage.CPP, problem.code)
                except ProblemNotFoundError:
                    logger.warning(f"Problem {problem.problem_id} not found in judge dataset.")
                except Exception as e:
                    logger.error(f"Error submitting problem {problem.problem_id}: {e}")

        with ThreadPoolExecutor(max_workers=llm_concurrency) as pool:
            list(tqdm.tqdm(pool.map(solve_and_submit, problem_set.values()), total=len(problem_set), desc="Generating solutions"))
        for problem in tqdm.tqdm(problem_set.values(), desc="Fetching results"):
            if not problem.submission_id:
                if (problem.response_meta or {}).get("stop_reason") == "refusal":
                    problem.judge_result = "Refusal"
                elif problem.text_response is not None and not problem.code:
                    problem.judge_result = "No Code"
                else:
                    problem.judge_result = "Judge Failed"
                continue
            while True:
                problem.judge_result = judge.get_result(problem.submission_id)
                if problem.judge_result != "Judging":
                    break
                time.sleep(1)
    
    results = []
    for problem in problem_set.values():
        results.append(BenchmarkResult(**problem.model_dump()).model_dump())
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump({
            "condition": condition,
            "system_prompt_file": system_prompt_path,
            "system_prompt": system_prompt,
            "llm_class": type(llm_instance).__name__,
            "model": getattr(llm_instance, "MODEL", None),
            "max_tokens": getattr(llm_instance, "MAX_TOKENS", None),
            "thinking_budget": getattr(llm_instance, "THINKING_BUDGET", None),
            "user_prompt_prefix": llm_instance.prompt,
            "problem_ids": list(problem_set),
            "judge_workers": worker,
            "llm_concurrency": llm_concurrency,
        }, f, indent=4)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    print_stats(dataset, problem_set)
