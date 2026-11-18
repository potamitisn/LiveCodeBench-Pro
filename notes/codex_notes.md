# Structured Reasoning Prompting — LiveCodeBench-Pro

## 1. Objective

Implement a simple, prompting-only Structured Reasoning baseline for **Claude Sonnet 4 with extended thinking enabled** in the existing LiveCodeBench-Pro evaluation pipeline.

Call the baseline **SR-Prompt** (Structured Reasoning Prompting), or follow the repository's existing naming conventions. The goal is to investigate whether instructing an existing reasoning model to reason structurally improves coding performance and/or computational efficiency, without modifying its architecture, decoding algorithm, or inference-time orchestration.

The agent should inspect the existing repository and its Claude integration before making changes. The instructions below describe the intended experimental behavior, not a prescribed repository architecture.

## 2. Research context

The Structured Reasoning framework (Section 2 of the accompanying manuscript, *Structured Reasoning*) defines structure as a graded property of a reasoning process, characterized by:

- **Units:** Distinct, explicit, addressable intermediate reasoning units (e.g., hypotheses, candidate solutions, partial plans, subproblems).
- **Relations:** Defined relationships between units that constrain dependencies and information flow (e.g., parent-of, alternative-to, refinement-of, evidence-for).
- **Operators:** Assessment, expansion, control, and memory operations applied over those units.

The operator vocabulary includes:

| Family | Operators |
| --- | --- |
| Expansion | Expand, Branch, Decompose |
| Assessment | Evaluate, Verify, Reflect |
| Control | Select, Backtrack, Merge, Refine, Halt, Respond |
| Memory | Store, Retrieve |

Branching creates competing alternatives (OR); decomposition creates complementary subproblems (AND). Structure is not synonymous with a tree, graph, or parallel execution. A sequential process may exhibit structure, and a parallel process may lack it.

The manuscript distinguishes **executed** structure from **narrated** structure. A model saying that it backtracked does not establish that it restored a previous computational state or prevented the abandoned branch from influencing future reasoning. The prompting baseline must not be presented as guaranteeing execution of any structural operator.

## 3. Proposed system prompt

Use the following as a starting point. The coding agent may refine the wording if justified, but should preserve the framework, remain task-agnostic, and avoid imposing a fixed algorithm. Prefer this relatively compact version for the first experiment.

```text
Solve the problem using Structured Reasoning.

Organize your thinking into distinct, explicitly identifiable and addressable intermediate reasoning units. These may be thoughts, hypotheses, candidate solutions, partial solutions, subproblems, or plans.

Maintain defined relationships between units, such as dependencies, competing alternatives, refinements, and supporting evidence. Respect these relationships when using information from different units.

Dynamically apply reasoning operators as appropriate:
- Expansion: extend existing units, branch into competing alternatives, or decompose a problem into complementary subproblems.
- Assessment: evaluate the promise of units, verify their validity, and reflect on their weaknesses.
- Control: selectively pursue promising units, prune unpromising ones, backtrack, merge useful information, and refine units based on feedback.
- Memory: retain and retrieve useful intermediate information when needed.

Keep competing alternatives distinct. When abandoning an approach, avoid carrying its invalid assumptions into new attempts, while retaining verified information and useful lessons. Preserve valid intermediate results and revise dependent units when necessary.

Choose which operations to apply, when to apply them, and which units to target based on the problem and evidence accumulated during reasoning. Allocate reasoning effort adaptively and stop when sufficient evidence supports an answer.

Do not follow a predetermined reasoning topology or fixed sequence of operations. Use only the structure appropriate for the problem. Apply these instructions during your thinking process, and follow the user's required final-answer format.
```

### Prompt design constraints

- Do not force branching, decomposition, backtracking, or any other operation on every problem.
- Do not impose a fixed number of candidates, a tree depth, a beam width, or a hand-designed search policy.
- Do not require JSON, XML, explicit operator logs, or reasoning transcripts in the final answer. The benchmark's existing answer/code format must remain intact.
- Do not add coding-specific hints, benchmark-specific tricks, examples of solved benchmark problems, or additional problem information to the SR prompt.
- Do not introduce extra model calls, tools, verification passes, execution feedback, or external search as part of SR-Prompt. Preserve whatever the original evaluation protocol already uses.
- Do not claim the prompt can enforce computational state isolation, restoration, or actual internal execution.

## 4. Where to put the prompt

Use the **system prompt**, not the user prompt, for the SR instructions. The benchmark problem and its existing formatting should remain unchanged in the user message.

If the repository already uses a system prompt (for example, to enforce a code-only answer format), preserve it in both conditions and append the SR instructions only in the treatment condition. Check for conflicts and do not silently replace existing benchmark instructions. The vanilla condition should retain its original behavior.

Illustrative Anthropic Messages API example (adapt to the existing provider wrapper; do not replace the benchmark integration with this standalone code):

```python
import anthropic

client = anthropic.Anthropic()

# Keep these settings identical between vanilla and SR-Prompt.
# The values are illustrative, not prescribed experimental budgets.
model = "claude-sonnet-4-20250514"
thinking_budget = 12000
max_tokens = 16384

# Read the proposed SR instructions from a versioned prompt file or constant.
sr_prompt = STRUCTURED_REASONING_PROMPT
base_system = EXISTING_SYSTEM_PROMPT  # Use None if the original has none.

def run(problem, use_sr=False):
    system = base_system
    if use_sr:
        system = "\n\n".join(filter(None, [base_system, sr_prompt]))

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "enabled", "budget_tokens": thinking_budget},
        messages=[{"role": "user", "content": problem}],
    )
    if system is not None:
        kwargs["system"] = system
    return client.messages.create(**kwargs)
```

Anthropic's native Messages API uses a top-level `system` parameter rather than a message with `role="system"`. If the repository uses a gateway, an OpenAI-compatible API, or another provider abstraction, inspect its system-message and thinking configuration and ensure both are actually forwarded. Confirm the exact supported model identifier and thinking parameters against the configured provider. Do not silently substitute another model. Official documentation: https://docs.anthropic.com/ (Messages API and extended thinking).

## 5. Experimental conditions

The primary comparison should be:

| Condition | System prompt | User prompt | Extended thinking |
| --- | --- | --- | --- |
| Vanilla Thinking | Original system prompt (or none) | Original benchmark problem | Enabled |
| SR-Prompt | Original system prompt + SR instructions | Same benchmark problem | Enabled |

Keep model version/provider, configured thinking and output budgets, supported sampling parameters, benchmark split, problem set, number of runs, timeout/retry policy, tooling, answer extraction, and scoring identical across conditions. Do not tune the prompt on held-out evaluation results.

Optional future ablations (not required for this implementation): a one-line instruction (`Think using Structured Reasoning.`), a more detailed operator-by-operator prompt, and a user-message placement variant. Keep these separate from the primary baseline.

### Evaluation and logging

- Use the existing LiveCodeBench-Pro evaluator and its official correctness metrics. Report comparable metrics (e.g., pass@1 or pass@k where supported) using the same protocol for both conditions.
- Record the exact prompt/version, resolved model ID, thinking configuration, token limits, and run settings needed to reproduce the experiment.
- Record actual input/output token usage, provider-reported reasoning/thinking usage if separately available, latency, and cost if reliably available. Do not estimate hidden thinking tokens from the length of returned thinking summaries.
- Equal configured thinking budgets do not necessarily imply equal actual compute. Report actual token usage and, if feasible, compare performance across matched budget levels or comparable observed token consumption.
- Log errors, truncated responses, and stop reasons separately from ordinary incorrect solutions. Use the same retry policy in both conditions.
- If the existing integration exposes thinking summaries, keep them separate from the final answer passed to the evaluator. Treat them as observable summaries, not complete internal reasoning traces or proof of operator execution.

## 6. Implementation checklist for the coding agent

1. Locate the benchmark entry point, model/provider configuration, input construction, Claude thinking settings, response parsing, and result logging.
2. Identify the existing system instructions and the mechanism by which they reach the provider.
3. Add SR-Prompt as an opt-in configuration/method, reusing the existing pipeline rather than duplicating it. Keep the default/vanilla path unchanged.
4. Store the exact SR system prompt in a versioned, reusable location consistent with repository conventions.
5. Confirm that enabling SR-Prompt changes only the intended system instructions, not the benchmark user message, thinking configuration, evaluation protocol, or answer format.
6. Check that final-answer extraction excludes thinking blocks where applicable and still yields the expected code submission.
7. Add a small test or mocked request inspection for vanilla versus SR-Prompt. If the environment and credentials permit, run a small smoke test on the same problem(s) under both conditions; avoid launching a full expensive benchmark run without authorization.
8. Document how to run both conditions using the repository's actual commands/configuration options, along with affected files, tests performed, and limitations. Do not invent command-line flags or report tests that were not run.

## 7. Interpretation

SR-Prompt tests whether prompting alone can elicit useful structured reasoning behavior in an existing thinking model. An improvement in benchmark accuracy or efficiency is evidence of the utility of the prompting intervention, not by itself evidence that the model executes structural operators internally. Conversely, no improvement does not establish that structured reasoning is ineffective in general.

This is intended as the simplest prompting intervention, prior to methods that provide explicit operational support for addressable units, state restoration, branch-local state, controlled information transfer, or adaptive control.

## 8. Scope and assumptions

These notes are based on Section 2 of the manuscript and the accompanying discussion about a system-prompt-only baseline. The repository has not been inspected in preparing this handoff. API code and budget values are illustrative and have not been executed or independently verified against the repository's provider. The coding agent should resolve implementation-specific details from the actual codebase and report any blockers rather than changing the experimental design silently.
