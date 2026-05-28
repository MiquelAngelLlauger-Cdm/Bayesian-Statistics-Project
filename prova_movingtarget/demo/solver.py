"""
solver.py
=========
Eagle (1984) constrained-search dynamic programming solver.

Extracted and lightly refactored from `reproduce_eagle.ipynb`. Builds the
non-dominated vector sets A(n, i) backward in time and exposes a helper to
read off the optimal searcher path for a given prior and starting cell.

Conventions
-----------
* Cells are labelled 1..9 (1-indexed) for the user-facing API, laid out as

      1  2  3
      4  5  6
      7  8  9

  Internally we work 0-indexed.
* Detection is certain (q_j = 1 for all j), matching the notebook.
"""

import numpy as np
from scipy.optimize import linprog

# --------------------------------------------------------------------------
# Grid topology
# --------------------------------------------------------------------------

# Searcher action sets: where the searcher may go next (includes staying put).
C_1indexed = {
    1: [1, 2, 4],
    2: [1, 2, 3, 5],
    3: [2, 3, 6],
    4: [1, 4, 5, 7],
    5: [2, 4, 5, 6, 8],
    6: [3, 5, 6, 9],
    7: [4, 7, 8],
    8: [5, 7, 8, 9],
    9: [6, 8, 9],
}
C_0indexed = {i - 1: [x - 1 for x in nb] for i, nb in C_1indexed.items()}

# Target adjacency (no self-loops); used to build the transition matrix.
TARGET_NEIGHBORS = {
    1: [2, 4],
    2: [1, 3, 5],
    3: [2, 6],
    4: [1, 5, 7],
    5: [2, 4, 6, 8],
    6: [3, 5, 9],
    7: [4, 8],
    8: [5, 7, 9],
    9: [6, 8],
}


def build_transition_matrix(p_stay=0.4):
    """
    Build the 9x9 target Markov transition matrix P.

    The target stays with probability `p_stay` and otherwise spreads the
    remaining mass uniformly over its side-adjacent neighbours.
    """
    p_move = 1.0 - p_stay
    P = np.zeros((9, 9))
    for i in range(1, 10):
        P[i - 1, i - 1] = p_stay
        neighs = TARGET_NEIGHBORS[i]
        for n in neighs:
            P[i - 1, n - 1] = p_move / len(neighs)
    return P


# --------------------------------------------------------------------------
# Vector pruning (dominance)
# --------------------------------------------------------------------------

def prune_simple(vectors, tol=1e-9):
    """
    Discard vectors element-wise dominated by another vector.

    First deduplicates identical vectors (rounding to a tolerance grid), then
    runs an O(k^2) pairwise dominance check over the survivors. Deduplication
    is what keeps the survivor count small; without it the candidate sets blow
    up into the thousands and the pairwise pass becomes the bottleneck.
    """
    if len(vectors) <= 1:
        return vectors

    # --- Deduplicate identical vectors (keep first occurrence) ---
    seen = {}
    deduped = []
    for vec, path in vectors:
        key = tuple(np.round(vec / tol).astype(np.int64))
        if key not in seen:
            seen[key] = True
            deduped.append((vec, path))
    vectors = deduped
    if len(vectors) <= 1:
        return vectors

    arrs = np.array([v[0] for v in vectors])
    n_vecs = len(arrs)
    keep = np.ones(n_vecs, dtype=bool)

    # Vector i is dominated if some distinct j has arrs[j] >= arrs[i] (all comps).
    # Since duplicates are gone, ">=" everywhere with j != i means strict
    # domination, so we can safely discard i.
    for i in range(n_vecs):
        if not keep[i]:
            continue
        diff = arrs - arrs[i]          # (n, 9)
        dominates_i = np.all(diff >= -tol, axis=1)
        dominates_i[i] = False
        if np.any(dominates_i & keep):
            keep[i] = False
    return [vectors[i] for i in range(n_vecs) if keep[i]]


def _is_dominated_lp(a, other_arrs):
    """LP test: is `a` dominated by a convex combination of `other_arrs`?"""
    N = len(a)
    K = len(other_arrs)
    if K == 0:
        return False
    c = np.zeros(N + 1)
    c[:N] = -a
    c[N] = 1.0
    A_ub = np.zeros((K, N + 1))
    A_ub[:, :N] = other_arrs
    A_ub[:, N] = -1.0
    b_ub = np.zeros(K)
    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * N + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    return bool(res.success and res.fun >= -1e-9)


def prune_combined(vectors):
    """Simple pruning followed by LP convex-combination pruning."""
    pruned = prune_simple(vectors)
    if len(pruned) <= 2:
        return pruned
    arrs = np.array([v[0] for v in pruned])
    n_vecs = len(arrs)
    keep = np.ones(n_vecs, dtype=bool)
    for i in range(n_vecs):
        others = [arrs[j] for j in range(n_vecs) if j != i and keep[j]]
        if others and _is_dominated_lp(arrs[i], np.array(others)):
            keep[i] = False
    return [pruned[i] for i in range(n_vecs) if keep[i]]


# --------------------------------------------------------------------------
# Dynamic programming
# --------------------------------------------------------------------------

def run_dynamic_programming(P, T=10, pruning_mode="simple"):
    """
    Build the vector sets A(n, i) for n = 0..T.

    Returns a dict keyed by (n, i) -> list of (vector, path) tuples, where
    `path` is the 1-indexed searcher path of length n associated with that
    vector (ordered from first search to last).
    """
    q = np.ones(9)
    A = {(0, i): [(np.zeros(9), [])] for i in range(9)}

    for n in range(1, T + 1):
        for i in range(9):
            candidates = []
            for j in C_0indexed[i]:
                e_j = np.zeros(9)
                e_j[j] = 1.0
                for arr_prev, path_prev in A[(n - 1, j)]:
                    u = P @ arr_prev
                    u_scaled = u.copy()
                    u_scaled[j] *= (1.0 - q[j])
                    arr = q[j] * e_j + u_scaled
                    candidates.append((arr, [j + 1] + path_prev))
            if pruning_mode == "none":
                A[(n, i)] = candidates
            elif pruning_mode == "simple":
                A[(n, i)] = prune_simple(candidates)
            elif pruning_mode == "combined":
                A[(n, i)] = prune_combined(candidates)
            else:
                raise ValueError(f"unknown pruning_mode: {pruning_mode}")
    return A


def normalize_prior(prior):
    prior = np.asarray(prior, dtype=float)
    if prior.shape != (9,):
        raise ValueError("prior must have length 9.")
    total = prior.sum()
    if total <= 0:
        raise ValueError("prior must have positive total mass.")
    return prior / total


def best_path_for_prior(A, prior, start_cell, horizon):
    """
    Given the precomputed vector sets `A`, return (best_pd, best_path) for a
    searcher constrained to begin at `start_cell` (1-indexed) over `horizon`
    steps, under the target belief `prior`.

    Ties are broken deterministically (first path encountered).
    """
    prior = normalize_prior(prior)
    i = start_cell - 1
    best_pd = -np.inf
    best_path = None
    for arr, path in A[(horizon, i)]:
        pd = float(np.dot(prior, arr))
        if pd > best_pd + 1e-9:
            best_pd = pd
            best_path = path
    return best_pd, best_path


def optimal_path(prior, start_cell, horizon, P=None, p_stay=0.4,
                 first_search_is_start=False):
    """
    Fast exact optimal searcher path for a *fixed* prior.

    Because detection is certain (q = 1), the belief after a failed search is a
    deterministic function of the path (zero out the searched cell, renormalise
    implicitly by carrying unnormalised mass, then push through P). We therefore
    maximise the cumulative detection probability with a memoised DFS over
    (step, current_cell, belief-signature).

    This reproduces Eagle (1984) Table I exactly and runs in well under a
    second, unlike the full vector-set DP.

    Parameters
    ----------
    start_cell : int (1-indexed)
        The searcher's position *before* the first search.
    first_search_is_start : bool
        If False (Eagle's convention), the first searched cell is a neighbour
        of `start_cell`. If True, the first searched cell *is* `start_cell`.

    Returns
    -------
    (best_pd, best_path) with best_path 1-indexed, length == horizon.
    """
    if P is None:
        P = build_transition_matrix(p_stay=p_stay)
    prior = normalize_prior(prior)

    memo = {}

    def rec(cell, belief, steps_left):
        # Search `cell`: detect the mass currently sitting there.
        detect = belief[cell - 1]
        b = belief.copy()
        b[cell - 1] = 0.0
        b = b @ P
        if steps_left == 1:
            return detect, [cell]

        key = (cell, steps_left, tuple(np.round(b, 9)))
        if key in memo:
            sub_pd, sub_path = memo[key]
            return detect + sub_pd, [cell] + sub_path

        best_pd = -1.0
        best_tail = None
        for nb in C_1indexed[cell]:
            pd, tail = rec(nb, b, steps_left - 1)
            if pd > best_pd:
                best_pd = pd
                best_tail = tail
        memo[key] = (best_pd, best_tail)
        return detect + best_pd, [cell] + best_tail

    first_cells = [start_cell] if first_search_is_start else C_1indexed[start_cell]
    best_pd = -1.0
    best_path = None
    for first in first_cells:
        pd, path = rec(first, prior.copy(), horizon)
        if pd > best_pd:
            best_pd = pd
            best_path = path
    return best_pd, best_path


def optimal_path_global(prior, horizon, P=None, p_stay=0.4):
    """Best optimal path over all 9 possible starting cells."""
    best_pd = -1.0
    best_path = None
    best_start = None
    for s in range(1, 10):
        pd, path = optimal_path(prior, s, horizon, P=P, p_stay=p_stay)
        if pd > best_pd:
            best_pd, best_path, best_start = pd, path, s
    return best_pd, best_path, best_start


def best_path_global(A, prior, horizon):
    """Best path over *all* starting cells for the given prior."""
    prior = normalize_prior(prior)
    best_pd = -np.inf
    best_path = None
    best_start = None
    for i in range(9):
        for arr, path in A[(horizon, i)]:
            pd = float(np.dot(prior, arr))
            if pd > best_pd + 1e-9:
                best_pd = pd
                best_path = path
                best_start = i + 1
    return best_pd, best_path, best_start
