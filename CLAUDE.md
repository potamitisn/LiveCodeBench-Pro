# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LiveCodeBench Pro is a toolkit for benchmarking LLMs on competitive programming problems. An LLM generates a C++ solution for each problem. A local judge (LightCPVerifier, running in Docker) checks each solution against the hidden test cases.

## Setup & Commands

- Python >= 3.12. Install with `uv sync` (or `pip install -r requirements.txt`).
- `LightCPVerifier/` is a git submodule. Run `git submodule update --init` before the first run, because the Docker image is built from it.
- Docker must be running. The judge container runs with `privileged=True`, and go-judge needs Linux cgroups (the README targets Ubuntu 20.04+), so judging may not work on macOS.
- Full benchmark: `python benchmark.py`. Writes `benchmark_result.json` and prints accuracy by split and difficulty.
- Judge smoke test: `python judge.py`. Submits one known solution for problem `2000A`.
- LLM smoke test: `python api_interface.py`. Needs `OPENAI_API_KEY` for `ExampleLLM`.
- There is no test suite, linter or build step.
- Local run on macOS (Intel): Docker comes from Colima (`colima start`); export `DOCKER_HOST=unix://$HOME/.colima/default/docker.sock`. The API key is in `~/projects/.env` (`set -a; source ~/projects/.env; set +a`).
- Experiment knobs (env vars for `benchmark.py`): `LCB_CONDITION` = `baseline` | `sr_compact` | `sr_explicit` (only the system prompt differs; prompts live in `prompts/`), `LCB_PROBLEMS=id1,id2` and `LCB_LIMIT=N` to run a subset. Outputs go to `results/<condition>/` (`llm_responses.jsonl` is a response cache: reruns skip cached problems and make no API call).
- `LightCPVerifier/` carries local patches: npm pinned to 10, go-judge pinned to v1.9.6, compile CPU limits raised to 60 s. The judge image is only built when missing, so after editing the submodule run `docker build -t lightcpverifier LightCPVerifier`.
- Never start a run that calls the API without the user's explicit confirmation.

## Architecture

The pipeline is `benchmark.py` → `api_interface.py` → `util.py` → `judge.py`:

1. **Dataset**: `benchmark.py` loads the HF dataset `QAQAQAQAQ/LiveCodeBench-Pro`. Its splits can share problems, so `get_problem_set` dedupes by `problem_id`. Each problem is solved once, but stats are reported per split.
2. **LLM**: subclasses of `LLMInterface` implement `call_llm(prompt) -> (text, metadata)`. `generate_solution` adds the shared C++ system prompt in front of the problem statement. The model under test is set by editing the `llm_instance` and `worker` globals at the top of `benchmark.py`. There is no CLI or config file for these.
3. **Code extraction**: `util.extract_longest_cpp_code` takes the last ```` ```cpp ```` block that contains `#include`. If there is none, it brace-matches from the last `int main(` and looks upward for `#include` lines. It returns `None` if both fail, and then the problem is never submitted.
4. **Judging**: `LightCPVerifierJudge` is a context manager. On entry it builds the `lightcpverifier` image (only if missing), replaces any existing container with that name, and maps port 8081 to a random host port. It also mounts `LightCPVerifier/{problems,submissions,data}`. Test data is downloaded lazily for each problem from the HF dataset `QAQAQAQAQ/LiveCodeBench-Pro-Testcase` (`{problem_id}.zip`) and unpacked to `LightCPVerifier/problems/<id>/`. A problem counts as cached if its `config.yaml` exists. The REST API is `POST /submit` → `sid`, then `GET /result/{sid}`. `get_result` maps both a 404 and `queued` to `"Judging"`, and `error` to `"Judge Failed"`.
5. **Two phases**: every problem is generated and submitted first. Only then are results polled, once per second. Problems with no extracted code, or whose submission failed, are recorded as `"Judge Failed"`.

The `Judge` ABC and the `SupportedLanguage` enum (cpp/python3/pypy3) allow other judges or languages, but `benchmark.py` only submits C++.

Results can be emailed to zz4242@nyu.edu for the leaderboard (see the README).
