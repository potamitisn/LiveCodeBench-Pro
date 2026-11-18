# Notes for the agent: Structured Reasoning baseline on LiveCodeBench Pro

## What you are asked to do

Run LiveCodeBench Pro **twice** with the same model (Claude Sonnet 4 with extended thinking, `claude-sonnet-4-20250514`):

1. **Baseline:** the model as is, with the benchmark's normal prompt.
2. **Structured:** identical, except that a *structured reasoning* instruction block is added to the **system prompt**.

The only difference between the two runs must be that system-prompt block. Model, thinking budget, max tokens, user prompt, problem set, number of samples, and code extraction stay the same.

This is meant as the most basic baseline: we only *instruct* the model to reason in a structured way. We are not building a search algorithm or scaffold around it.

---

## Background: what "structured reasoning" means here

This comes from a draft paper ("Structured Reasoning", ICLR 2027 submission, §2). A reasoning process is structured to the degree that:

1. **Units:** it keeps explicit, addressable intermediate units (an approach, a subproblem, a code draft, a test result), each with an id that later steps can point to.
2. **Relations:** units are explicitly related (child-of, alternative-to, part-of, refinement-of). Sibling alternatives share their parent's state, not each other's history.
3. **Operators:** control flow is made explicit through operators applied to units:
   - *Expansion:* Branch (competing alternatives, at most one needs to work), Decompose (parts that must all be solved)
   - *Assessment:* Evaluate (how promising), Verify (is it valid), Reflect
   - *Control:* Select/Prune, Backtrack, Merge, Refine, Halt

A prompt can only make the model *narrate* this structure in its text. Nothing guarantees the model actually executes it. That's acceptable for us, because this is the floor that more serious methods should beat.

---

## Implementation notes

### Where the prompt goes
- Put it in the **system prompt**. Append the block below to whatever system message the harness already uses (or use it with a one-line role sentence if there is none). Leave the user message (the problem) byte-identical between the two runs.
- The LiveCodeBench Pro toolkit's LLM interface (`call_llm(self, user_prompt)`) only receives the user prompt. Your LLM class therefore has to pass the system prompt itself, e.g. `client.messages.create(system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user_prompt}], ...)`. Make the system prompt a constructor argument so the same class serves both runs: an empty or plain role sentence for the baseline, role plus structured block for the structured run.

### Thinking
- Use extended thinking in **both** runs with the same `budget_tokens` and `max_tokens` (for example `budget_tokens=24000`, `max_tokens=32000`). Temperature cannot be changed while thinking is on, so don't set it.
- The structured reasoning should happen **inside the thinking**. The visible answer should contain only the final code block, so code extraction behaves identically in both runs.
- Claude 4 models return only a *summary* of their thinking through the API. You cannot fully verify that the model followed the structure. Spot-check a handful of thinking summaries for unit ids / `OP` lines and report roughly how often they appear.

### Language and extraction
- LiveCodeBench Pro judges **C++**. The toolkit extracts the C++ code from the response automatically. The prompt below therefore ends with a single ```` ```cpp ```` block. Check how the toolkit's extractor picks the block (first or last fenced block) and make sure the structured run's visible output contains only one code block, so the extractor can't pick the wrong one.

### Fairness checks before the full run
- Run 3–5 problems in both modes. Confirm the two user prompts are identical, the system prompts differ only by the block, and code is extracted correctly in both modes.
- Log `response.usage` (input/output tokens) for every call. The structured run will probably spend more thinking tokens, and that needs to be reported next to accuracy.
- Use several samples per problem if the budget allows (sampling runs at temperature 1 with thinking on), and compare the two runs per problem, not only on the headline pass rate.
- Restrict to problems released after the model's training cutoff (early 2025) where possible.

---

## The system-prompt block

Append this to the system prompt for the structured run (adjust only if the harness requires a different final-answer format):

````text
Reason about the problem using structured reasoning. In your thinking, organize your work as explicit UNITS and OPERATIONS rather than one free-flowing chain.

UNITS: every intermediate result is a unit with a unique id (U1, U2, ...), its parent, its relation, and a self-contained state. A unit can be an observation about the constraints, a candidate approach or algorithm, a subproblem, a code draft, or a test result.
  [U<n> | parent=<U-id or ROOT> | rel=child-of|alternative-to:<U>|part-of:<U>|refinement-of:<U> | state: ...]
A unit's state must be self-contained: reading it together with its parent chain must be enough to continue from it.

OPERATIONS, always naming the units they act on:
  [OP BRANCH <U> -> <U-ids>]        propose competing approaches; each must rest on a genuinely different idea
  [OP DECOMPOSE <U> -> <U-ids>]     split into parts that must all be solved
  [OP EVALUATE <U> = <0-1> : ...]   judge promise: correctness of the idea and time/memory complexity against the constraints
  [OP VERIFY <U> = PASS|FAIL : ...] check validity: trace the sample tests, edge cases (min/max sizes, equal values, overflow), and the complexity bound
  [OP PRUNE <U> : ...]              abandon a unit permanently
  [OP BACKTRACK -> <U>]             resume from that unit's state only; reuse nothing from pruned descendants
  [OP MERGE <U-ids> -> <U> : ...]   combine results from several units
  [OP REFINE <U> -> <U'> : ...]     revise a unit using feedback from EVALUATE or VERIFY
  [OP HALT : ...]

After every BRANCH, PRUNE, BACKTRACK or MERGE, list the open units: [FRONTIER: ...]
Never re-expand a state you have already explored.
Halt as soon as a code unit passes VERIFY on the samples and edge cases; do not reopen a verified unit.

Your visible answer must contain only the final C++17 program from the verified unit, in a single ```cpp code block, and nothing else.
````

---

## Example of the intended reasoning

This example is for **your understanding** of what the instruction asks for. Do **not** put it in the prompt by default: a worked example biases which strategies the model tries. If format compliance turns out to be very poor in the pilot, adding a short example is an option, but report it as a separate condition.

Problem (toy): *Given n ≤ 2·10^5 integers a_i (|a_i| ≤ 10^9) and K, count pairs i < j with a_i + a_j = K.*

```text
[U1 | parent=ROOT | rel=child-of | state: n up to 2e5 -> need ~O(n log n); sums up to 2e9 -> use long long; answer up to ~2e10 -> long long]
[OP BRANCH U1 -> U2, U3, U4]
[U2 | parent=U1 | rel=alternative-to:U3 | state: brute force over all pairs, O(n^2)]
[U3 | parent=U1 | rel=alternative-to:U2 | state: hash map of counts; for each a_j add cnt[K - a_j], then cnt[a_j]++]
[U4 | parent=U1 | rel=alternative-to:U3 | state: sort + two pointers, careful counting of equal runs]
[FRONTIER: U2, U3, U4]
[OP EVALUATE U2 = 0.05 : 4e10 operations, too slow]
[OP PRUNE U2 : violates time limit]
[OP EVALUATE U3 = 0.9 : O(n) expected, simple; anti-hash tests possible -> use custom hash]
[OP EVALUATE U4 = 0.7 : O(n log n), equal-run counting is error-prone]
[FRONTIER: U3, U4]
[U5 | parent=U3 | rel=child-of | state: code draft using unordered_map<long long,long long> with splitmix64 hash]
[OP VERIFY U5 = FAIL : sample 2 (all equal values, K = 2x) gives 0; bug: cnt updated before lookup]
[OP REFINE U5 -> U6 : swap order, lookup first then increment]
[U6 | parent=U5 | rel=refinement-of:U5 | state: corrected code draft]
[OP VERIFY U6 = PASS : samples 1-2 match; n=1 -> 0; all equal -> n(n-1)/2; overflow handled]
[OP HALT : verified unit U6]
```

Visible answer: only the ```` ```cpp ```` block with U6's program.

---

## What to report back

- Pass rate per run (overall and by difficulty tier), with the number of samples per problem.
- Per-problem comparison: problems solved only by baseline, only by structured, by both, by neither.
- Mean input/output tokens per problem for each run.
- Format-compliance spot check: in what fraction of sampled thinking summaries the unit/operator structure is visible.
- Any deviations from this plan (prompt changes, extraction fixes, problems skipped).
