# agent/eval/golden/

Twenty hand-authored tasks with a real checker each, run through the full
pipeline by `otto eval` (`agent/eval/runner.py`). One JSON file per item:

```json
{
  "id": "nphard_math_binpacking_01",
  "domain": "math",
  "prompt": "Items of sizes [...] must be packed into bins of ...",
  "checker": "# 5 is what first-fit-decreasing needs here; the optimum is 4.\nanswer_is(4.0, decoys=[5.0])"
}
```

| group | items | how they are checked |
| --- | --- | --- |
| `code_*` | 6 | the candidate's code is run and its output compared |
| `math_*` | 6 | the value must be present and no decoy may be |
| `nphard_*` | 8 | four code items (graph colouring, knapsack, two TSPs) and four questions with one number for an answer (subset sum, bin packing, set cover, maximum clique) |

Every NP-hard optimum was verified by exhaustive search before the task was
written. A decoy is what a named wrong method produces on that instance: the
bin count first-fit-decreasing needs where fewer suffice, the set count a
greedy cover takes, the clique size an obvious sub-optimal choice gives, the
target read straight off a subset-sum question when it is not achievable. A
checker that only rejects nonsense is not checking optimality, and a checker
that passes on the right number appearing anywhere in the text lets a model
enumerate possibilities and conclude the wrong one; `answer_is(expected,
decoys=[...])` refuses both.

`debug_pipeline.py` at the repository root runs one or more items and prints
the board, the action record, the checklist the judge held the answer to,
the final answer, and the checker's verdict.
