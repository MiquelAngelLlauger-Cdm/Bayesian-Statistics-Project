"""
simulation.py
=============
Live simulation of the constrained search.

The DP solver (solver.py) gives us an *optimal searcher path*. Here we run an
actual episode: the submarine has a real (hidden) cell, the searcher walks the
optimal path, and after each searcher move the submarine takes one Markov step
according to P.

Timeline of one "round" k (0-indexed), matching the requested behaviour:
    1. The searcher moves to optimal_path[k] and searches that cell.
       -> If the submarine is currently there, it is FOUND. Episode ends.
    2. One step later, the submarine performs its Markov transition (unless it
       was just found, or the path is exhausted).

The episode ends when the submarine is found, or when the searcher has walked
the entire optimal path without a detection.
"""

import numpy as np

import solver


class SearchSimulation:
    """
    Stateful, step-driven simulation. The pygame layer calls `searcher_step()`
    and `submarine_step()` on a timer and reads the public attributes to draw.
    """

    # Episode outcome states
    RUNNING = "running"
    FOUND = "found"
    ESCAPED = "escaped"

    def __init__(self, p_stay=0.4, prior=None, start_cell=None,
                 horizon=10, sub_start=None, rng=None,
                 first_search_is_start=False):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.horizon = horizon
        self.p_stay = p_stay

        # Target belief / prior. Default matches the notebook: certain in cell 9.
        if prior is None:
            prior = np.zeros(9)
            prior[8] = 1.0
        self.prior = solver.normalize_prior(prior)

        # Build the model and solve for the optimal path (fast exact solver).
        self.P = solver.build_transition_matrix(p_stay=p_stay)

        if start_cell is not None:
            self.opt_pd, self.optimal_path = solver.optimal_path(
                self.prior, start_cell, horizon, P=self.P,
                first_search_is_start=first_search_is_start)
            self.start_cell = start_cell
        else:
            self.opt_pd, self.optimal_path, self.start_cell = \
                solver.optimal_path_global(self.prior, horizon, P=self.P)

        # Sample the submarine's true starting cell from the prior (1-indexed),
        # unless explicitly pinned.
        if sub_start is None:
            sub_start = int(self.rng.choice(9, p=self.prior)) + 1
        self.sub_cell = sub_start

        # Searcher hasn't moved yet; index points to the *next* cell to search.
        self.searcher_cell = None
        self.path_index = -1

        self.status = self.RUNNING
        self.log = []  # human-readable event log

        self._log(f"Optimal path (start cell {self.start_cell}): "
                  f"{'-'.join(map(str, self.optimal_path))}")
        self._log(f"Optimal detection probability P_d = {self.opt_pd:.4f}")
        self._log(f"Submarine secretly starts in cell {self.sub_cell}.")

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def searcher_step(self):
        """
        Advance the searcher to the next cell on the optimal path and check
        for detection. Returns True if the episode is still running afterward.
        """
        if self.status != self.RUNNING:
            return False

        self.path_index += 1
        if self.path_index >= len(self.optimal_path):
            # Path exhausted with no detection.
            self.status = self.ESCAPED
            self._log("Optimal path exhausted -- submarine escaped.")
            return False

        self.searcher_cell = self.optimal_path[self.path_index]
        t = self.path_index + 1
        self._log(f"t={t}: searcher searches cell {self.searcher_cell}.")

        if self.searcher_cell == self.sub_cell:
            self.status = self.FOUND
            self._log(f"t={t}: FOUND the submarine in cell "
                      f"{self.searcher_cell}!")
            return False
        return True

    def submarine_step(self):
        """
        Perform one Markov transition of the submarine. Called a beat after the
        searcher moves, only while the episode is running and the searcher has
        more moves to make.
        """
        if self.status != self.RUNNING:
            return
        if self.path_index >= len(self.optimal_path) - 1:
            # No further searcher move will follow, so the sub need not move.
            return

        probs = self.P[self.sub_cell - 1]
        new_cell = int(self.rng.choice(9, p=probs)) + 1
        if new_cell != self.sub_cell:
            self._log(f"   submarine moves {self.sub_cell} -> {new_cell}.")
        else:
            self._log(f"   submarine stays in cell {self.sub_cell}.")
        self.sub_cell = new_cell

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_running(self):
        return self.status == self.RUNNING

    def _log(self, msg):
        self.log.append(msg)

    @staticmethod
    def cell_to_rowcol(cell):
        """1-indexed cell -> (row, col), row 0 at top."""
        idx = cell - 1
        return idx // 3, idx % 3
