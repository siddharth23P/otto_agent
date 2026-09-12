"""Smoke coverage for the golden-dataset harness (agent/eval/): the golden
set itself loads and covers both domains, and check_output() actually runs
candidates/checkers for real (no LLM calls involved here -- run_golden()'s
LLM-driving path is exercised manually via `otto eval`, not in the offline
suite).
"""
import pytest

from agent.eval.runner import GoldenItem, check_output, load_golden


def _golden_item(item_id: str) -> GoldenItem:
    """The real item from agent/eval/golden, by id -- so these tests run the
    shipped checker rather than a copy of it that can drift."""
    for item in load_golden():
        if item.id == item_id:
            return item
    raise AssertionError(f"no golden item {item_id!r}")


def test_golden_set_loads_and_has_both_domains():
    items = load_golden()
    domains = {i.domain for i in items}
    assert domains == {"code", "math"}
    assert len(items) >= 10


def test_every_golden_checker_accepts_a_correct_candidate_and_rejects_a_wrong_one():
    # Hand-written correct/wrong solutions for each golden item, run through
    # the real checker -- this validates the golden set's own checkers, not
    # just the harness plumbing.
    solutions = {
        "code_01": ("def is_prime(n):\n    if n < 2: return False\n"
                    "    return all(n % d for d in range(2, int(n**0.5) + 1))\n",
                    "def is_prime(n):\n    return True\n"),
        "code_02": ("def fibonacci(n):\n    a, b = 0, 1\n"
                    "    for _ in range(n): a, b = b, a + b\n    return a\n",
                    "def fibonacci(n):\n    return 0\n"),
        "code_03": ("def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n",
                    "def reverse_words(s):\n    return s\n"),
        "code_04": ("def binary_search(arr, target):\n"
                    "    lo, hi = 0, len(arr) - 1\n"
                    "    while lo <= hi:\n"
                    "        mid = (lo + hi) // 2\n"
                    "        if arr[mid] == target: return mid\n"
                    "        if arr[mid] < target: lo = mid + 1\n"
                    "        else: hi = mid - 1\n"
                    "    return -1\n",
                    "def binary_search(arr, target):\n    return -1\n"),
        "code_05": ("def is_palindrome(s):\n"
                    "    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n",
                    "def is_palindrome(s):\n    return False\n"),
        "code_06": ("def merge_intervals(intervals):\n"
                    "    if not intervals: return []\n"
                    "    ivs = sorted(intervals)\n    out = [ivs[0]]\n"
                    "    for s, e in ivs[1:]:\n"
                    "        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)\n"
                    "        else: out.append([s, e])\n    return out\n",
                    "def merge_intervals(intervals):\n    return intervals\n"),
        "math_01": ("The answer is 1275.", "The answer is 42."),
        "math_02": ("120", "42"),
        "math_03": ("12", "7"),
        "math_04": ("x = 5", "x = 999"),
        "math_05": ("0.375", "0.9"),
        "math_06": ("153.94", "1.0"),
        # NP-hard MATH items. These are the same class of problem as the
        # nphard code items, asked as a question rather than as a function to
        # write, so the "wrong" answer is what a greedy or first-fit pass
        # actually produces rather than an arbitrary number -- a checker that
        # only rejects nonsense is not checking optimality.
        #
        # Subset sum: the target is 33333 and the true best is 33308, so
        # "33333" is exactly the mistake of reading the target off the
        # question and calling it an answer.
        "nphard_math_subsetsum_01": ("the best achievable sum is 33308", "33333"),
        # Bin packing: first-fit-decreasing needs 5 bins here, the optimum is 4.
        "nphard_math_binpacking_01": ("4 bins suffice", "you need 5 bins"),
        # Set cover: greedy takes 4 sets, the minimum is 3.
        "nphard_math_setcover_01": ("3 sets are enough", "4 sets"),
        # Max clique: the obvious 4-clique on 1,2,3,4 is not the largest --
        # adding vertex 8 makes 5.
        "nphard_math_clique_01": ("the largest clique has 5 vertices", "4"),
        # NP-hard items (pulled from NPHardEval instances, ground truth
        # independently verified by brute force -- see agent/eval/golden/
        # nphard_*.json for the embedded instance data and how the true
        # optimum was computed). "Wrong" here means genuinely, deliberately
        # suboptimal -- a couple of naive heuristics (nearest-neighbor TSP,
        # ratio-greedy knapsack) turned out to coincidentally hit the true
        # optimum on these small instances during authoring, which is not a
        # checker bug, just evidence the instances are easy for a decent
        # heuristic; these wrong solutions are chosen to fail unambiguously.
        "nphard_tsp_01": (
            "import itertools\n"
            "def solve_tsp(dist):\n"
            "    n = len(dist)\n"
            "    best = None; best_tour = None\n"
            "    for perm in itertools.permutations(range(1, n)):\n"
            "        t = [0] + list(perm) + [0]\n"
            "        d = sum(dist[t[i]][t[i+1]] for i in range(len(t)-1))\n"
            "        if best is None or d < best:\n"
            "            best, best_tour = d, t\n"
            "    return best_tour, best\n",
            "def solve_tsp(dist):\n"
            "    n = len(dist)\n"
            "    tour = list(range(n)) + [0]\n"
            "    total = sum(dist[tour[i]][tour[i+1]] for i in range(len(tour)-1))\n"
            "    return tour, total\n",
        ),
        "nphard_tsp_02": (
            "def solve_tsp(dist):\n"
            "    n = len(dist)\n"
            "    FULL = (1 << n) - 1\n"
            "    dp = {(1 << 0, 0): (0, [0])}\n"
            "    from itertools import combinations\n"
            "    for size in range(2, n + 1):\n"
            "        for subset_cities in combinations(range(1, n), size - 1):\n"
            "            subset = (1 << 0) | sum(1 << c for c in subset_cities)\n"
            "            for last in subset_cities:\n"
            "                prev_subset = subset & ~(1 << last)\n"
            "                best = None\n"
            "                for k in subset_cities:\n"
            "                    if k == last:\n"
            "                        continue\n"
            "                    if (prev_subset, k) not in dp:\n"
            "                        continue\n"
            "                    cost, path = dp[(prev_subset, k)]\n"
            "                    cand = cost + dist[k][last]\n"
            "                    if best is None or cand < best[0]:\n"
            "                        best = (cand, path + [last])\n"
            "                if best is None and prev_subset == (1 << 0):\n"
            "                    cost, path = dp[(prev_subset, 0)]\n"
            "                    best = (cost + dist[0][last], path + [last])\n"
            "                if best is not None:\n"
            "                    dp[(subset, last)] = best\n"
            "    best_total = None; best_path = None\n"
            "    for last in range(1, n):\n"
            "        if (FULL, last) in dp:\n"
            "            cost, path = dp[(FULL, last)]\n"
            "            cand = cost + dist[last][0]\n"
            "            if best_total is None or cand < best_total:\n"
            "                best_total = cand\n"
            "                best_path = path + [0]\n"
            "    return best_path, best_total\n",
            "def solve_tsp(dist):\n"
            "    n = len(dist)\n"
            "    tour = list(range(n)) + [0]\n"
            "    total = sum(dist[tour[i]][tour[i+1]] for i in range(len(tour)-1))\n"
            "    return tour, total\n",
        ),
        "nphard_gcp_01": (
            "import itertools\n"
            "def color_graph(n, edges, k):\n"
            "    for coloring in itertools.product(range(k), repeat=n):\n"
            "        if all(coloring[a] != coloring[b] for a, b in edges):\n"
            "            return {i: coloring[i] for i in range(n)}\n"
            "    return None\n",
            "def color_graph(n, edges, k):\n"
            "    colors = {}\n"
            "    for v in range(n):\n"
            "        used = {colors[u] for (a, b) in edges for u in (a, b) if (a == v or b == v) and u in colors}\n"
            "        c = 0\n"
            "        while c in used:\n"
            "            c += 1\n"
            "        colors[v] = c % k\n"  # wraps around instead of reporting infeasible
            "    return colors\n",
        ),
        "nphard_ksp_01": (
            "def solve_knapsack(items, capacity):\n"
            "    n = len(items)\n"
            "    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n"
            "    for i in range(1, n + 1):\n"
            "        w, v = items[i - 1]\n"
            "        for c in range(capacity + 1):\n"
            "            dp[i][c] = dp[i - 1][c]\n"
            "            if w <= c:\n"
            "                dp[i][c] = max(dp[i][c], dp[i - 1][c - w] + v)\n"
            "    chosen = []\n"
            "    c = capacity\n"
            "    for i in range(n, 0, -1):\n"
            "        if dp[i][c] != dp[i - 1][c]:\n"
            "            chosen.append(i - 1)\n"
            "            c -= items[i - 1][0]\n"
            "    return chosen, dp[n][capacity]\n",
            "def solve_knapsack(items, capacity):\n"
            "    return [0], items[0][1]\n",
        ),
    }
    by_id = {i.id: i for i in load_golden()}
    assert set(solutions) == set(by_id), "every golden item needs a hand-checked solution pair here"

    for item_id, (correct, wrong) in solutions.items():
        item = by_id[item_id]
        passed, evidence = check_output(item, correct)
        assert passed, f"{item_id}: correct solution rejected -- {evidence}"
        passed, _ = check_output(item, wrong)
        assert not passed, f"{item_id}: wrong solution accepted"


def test_check_output_math_domain_accepts_the_right_number_anywhere_in_text():
    item = GoldenItem(
        id="t", domain="math", prompt="what is 2+2",
        checker=(
            "import re\n"
            "nums = [float(x) for x in re.findall(r'-?\\d+\\.?\\d*', CANDIDATE_OUTPUT)]\n"
            "assert any(abs(n - 4.0) < 1e-6 for n in nums)\n"
            "print('OK')\n"
        ),
    )
    passed, _ = check_output(item, "After computing carefully, the answer is 4.")
    assert passed is True


# --------------------------------------------------------------------------
# The hole the math checkers used to have: they asserted the right number
# appeared ANYWHERE in the candidate's text, so a model that enumerated
# possibilities and concluded the wrong one still passed, as long as the
# right value went by at some point. It was not being exploited when this was
# found -- the point is that it could not have been caught.
# --------------------------------------------------------------------------

_SHOWED_WORKING = {
    # Each pair is (a candidate that reasons aloud and lands on the WRONG
    # answer, the name of the wrong method it used). Every decoy here is the
    # `wrong` half of a pair that already existed in the test above.
    "nphard_math_subsetsum_01": (
        "Let me try. The target is 33333. Greedy from the largest: "
        "10453 + 9161 + 8353 + 3541 = 31508. Trying other subsets I can reach "
        "33308. But the cap is 33333, so the answer is 33333.",
        "reading the target off the question",
    ),
    "nphard_math_binpacking_01": (
        "First-fit-decreasing packs these into 5 bins. A smarter arrangement "
        "might manage 4, but I will go with 5 bins.",
        "first-fit-decreasing",
    ),
    "nphard_math_setcover_01": (
        "Greedy picks 4 sets. It is conceivable that 3 suffice; my answer is "
        "4 sets.",
        "greedy",
    ),
    "nphard_math_clique_01": (
        "Vertices 1,2,3,4 are mutually adjacent, so there is a clique of 4. "
        "Adding 8 might give 5 but I cannot confirm it. The answer is 4.",
        "stopping at the obvious clique",
    ),
}


@pytest.mark.parametrize("item_id", sorted(_SHOWED_WORKING))
def test_mentioning_the_right_number_while_concluding_a_wrong_one_fails(item_id):
    candidate, method = _SHOWED_WORKING[item_id]
    item = _golden_item(item_id)

    passed, evidence = check_output(item, candidate)

    assert not passed, (
        f"{item_id}: a candidate that concluded by {method} was accepted "
        "because the right number appeared earlier in its working"
    )
    assert "wrong answer" in evidence


@pytest.mark.parametrize("item_id", sorted(_SHOWED_WORKING))
def test_the_same_items_still_accept_real_working(item_id):
    # The guard must not reject a candidate that reasons aloud and gets it
    # RIGHT -- otherwise it trades a false pass for a false fail, which is
    # worse because nobody goes looking for it.
    item = _golden_item(item_id)
    correct = {
        "nphard_math_subsetsum_01":
            "Enumerating all 1024 subsets, the largest sum not exceeding the "
            "cap is 33308.",
        "nphard_math_binpacking_01":
            "First-fit-decreasing uses more bins than necessary here; an "
            "exhaustive check shows 4 bins suffice.",
        "nphard_math_setcover_01":
            "Greedy is not optimal here. The minimum cover uses 3 sets.",
        "nphard_math_clique_01":
            "The obvious mutually-adjacent group is not maximal; including "
            "vertex 8 gives a clique of 5.",
    }[item_id]

    passed, evidence = check_output(item, correct)

    assert passed, f"{item_id}: correct working rejected -- {evidence}"


def test_a_decoy_equal_to_the_answer_cannot_reject_every_candidate():
    # Defensive: a mis-authored item whose decoy IS the answer would
    # otherwise fail every candidate including the right one, and the report
    # would look like a model regression rather than a bad item.
    item = GoldenItem(id="x", domain="math", prompt="",
                      checker="answer_is(7.0, decoys=[7.0])\nprint('OK')\n")

    passed, evidence = check_output(item, "the answer is 7")

    assert passed, evidence


def test_plain_math_items_do_not_need_decoys_to_work():
    item = _golden_item("math_01")

    assert check_output(item, "The answer is 1275.")[0]
    assert not check_output(item, "The answer is 42.")[0]
