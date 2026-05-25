"""
utils.py
========
Core utilities for the Bayesian spatial search problem.

Modules:
- Grid and constants
- Prior construction (Monte Carlo over alpha, tau, perpendicular noise)
- Detection model (cloglog with depth/roughness terrain effects)
- Bayesian posterior update
- Mission selection (efficiency-based search over rectangles)
- Visualization helpers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Constants (fixed by the problem)
# ---------------------------------------------------------------------------
NX, NY = 50, 35
X_E = np.array([7.0, 20.0])
V_PLANE = np.array([6.0, -3.5])
V_WIND = np.array([-1.0, -1.5])
V_DRIFT = np.array([0.5, -1.5])
BUDGET_TOTAL = 530


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def load_grid(path="grid_dataset.csv"):
    """Load the grid dataset and return as a DataFrame indexed by cell_id."""
    df = pd.read_csv(path)
    return df.sort_values("cell_id").reset_index(drop=True)


def grid_to_arrays(grid_df):
    """
    Convert the grid DataFrame to 2D arrays of shape (NY, NX) for depth and
    roughness, plus the (x, y) coordinate meshgrid of cell centers.

    Cell at array index [iy, ix] has center (ix + 0.5, iy + 0.5).
    """
    depth_arr = np.zeros((NY, NX))
    rough_arr = np.zeros((NY, NX))
    for _, row in grid_df.iterrows():
        ix = int(row["x"] - 0.5)
        iy = int(row["y"] - 0.5)
        depth_arr[iy, ix] = row["depth"]
        rough_arr[iy, ix] = row["roughness"]
    xs = np.arange(NX) + 0.5
    ys = np.arange(NY) + 0.5
    return depth_arr, rough_arr, xs, ys


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------
def build_prior(
    n_samples=200_000,
    alpha_mean=1.2,
    alpha_std=0.4,
    tau_mean=1.0,
    tau_std=0.4,
    iso_std=1.5,
    perp_std=1.0,
    seed=0,
):
    """
    Construct the prior over the grid by Monte Carlo sampling.

    Model:
        landing(alpha, tau) = X_E + alpha * V_PLANE + tau * (V_WIND + V_DRIFT)
                              + isotropic noise N(0, iso_std^2 I)
                              + perpendicular noise N(0, perp_std^2) along
                                the direction orthogonal to V_PLANE

    The alpha and tau means/stds encode the witness statements:
      - alpha_mean > 1 reflects Witness 2 ("farther along the trajectory")
      - moderate alpha_std reflects Witness 1 ("kept moving forward")
      - perp_std reflects Witness 2 ("slightly off to one side")

    Returns
    -------
    prior : np.ndarray of shape (NY, NX), summing to 1.
    samples : np.ndarray of shape (n_samples, 2), the raw landing samples
              (useful for diagnostics / overlaying on plots).
    """
    rng = np.random.default_rng(seed)

    alpha = rng.normal(alpha_mean, alpha_std, size=n_samples)
    tau = rng.normal(tau_mean, tau_std, size=n_samples)

    drift_total = V_WIND + V_DRIFT  # combined surface drift per unit tau
    centers = X_E + alpha[:, None] * V_PLANE + tau[:, None] * drift_total

    # Isotropic noise (catch-all for unmodeled effects)
    centers += rng.normal(0.0, iso_std, size=(n_samples, 2))

    # Perpendicular noise along the direction orthogonal to V_PLANE
    v_norm = V_PLANE / np.linalg.norm(V_PLANE)
    v_perp = np.array([-v_norm[1], v_norm[0]])  # 90-degree rotation
    perp_offset = rng.normal(0.0, perp_std, size=n_samples)
    centers += perp_offset[:, None] * v_perp

    # Bin onto the grid: cell (ix, iy) has center (ix+0.5, iy+0.5),
    # so the cell containing point (x, y) is (floor(x), floor(y)).
    ix = np.floor(centers[:, 0]).astype(int)
    iy = np.floor(centers[:, 1]).astype(int)

    # Keep only samples inside the grid
    mask = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    ix, iy = ix[mask], iy[mask]

    prior = np.zeros((NY, NX))
    np.add.at(prior, (iy, ix), 1.0)
    prior /= prior.sum()

    return prior, centers


# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------
def detection_probability(effort, depth, roughness, kappa=0.9, gamma_d=1.8, gamma_r=1.5):
    """
    Complementary log-log detection model:
        g(d, r) = kappa * exp(-gamma_d * d - gamma_r * r)
        rho     = 1 - exp(-effort * g)

    Works elementwise: depth and roughness can be scalars or arrays of any shape.

    Returns
    -------
    rho : same shape as depth/roughness, all values in [0, 1].
    """
    g = kappa * np.exp(-gamma_d * np.asarray(depth) - gamma_r * np.asarray(roughness))
    rho = 1.0 - np.exp(-effort * g)
    return rho


def calibration_table(kappa=0.9, gamma_d=1.8, gamma_r=1.5):
    """
    Show rho values for the four reported mission scenarios.
    Useful to validate parameter choices against the qualitative reports.
    """
    scenarios = [
        ("Mission 1: moderate depth, smooth, e=1", 1, 0.55, 0.20, (0.25, 0.40)),
        ("Mission 2: shallow, smooth, e=3       ", 3, 0.20, 0.05, (0.85, 0.95)),
        ("Mission 3: deep, rough, e=3           ", 3, 0.85, 0.85, (0.20, 0.35)),
        ("Mission 4: moderate, irregular, e=2   ", 2, 0.40, 0.55, (0.40, 0.55)),
    ]
    print(f"Calibration: kappa={kappa}, gamma_d={gamma_d}, gamma_r={gamma_r}")
    print("-" * 78)
    for desc, e, d, r, target in scenarios:
        rho = detection_probability(e, d, r, kappa, gamma_d, gamma_r)
        lo, hi = target
        ok = "OK " if lo <= rho <= hi else "OFF"
        print(f"  {desc} -> rho={rho:.3f}  target=[{lo:.2f},{hi:.2f}]  {ok}")


# ---------------------------------------------------------------------------
# Bayesian update
# ---------------------------------------------------------------------------
def coverage_mask(x_min, x_max, y_min, y_max):
    """
    Return a boolean array of shape (NY, NX) indicating which cells lie inside
    the half-open rectangle [x_min, x_max) x [y_min, y_max).

    Cell (ix, iy) has center (ix+0.5, iy+0.5), so it is "inside" iff
        x_min <= ix < x_max  and  y_min <= iy < y_max.
    """
    if not (0 <= x_min < x_max <= NX and 0 <= y_min < y_max <= NY):
        raise ValueError(
            f"Invalid rectangle: ({x_min},{x_max},{y_min},{y_max}) "
            f"with grid ({NX},{NY})"
        )
    mask = np.zeros((NY, NX), dtype=bool)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def mission_cost(x_min, x_max, y_min, y_max, effort):
    """Cost = number of cells * effort."""
    n_cells = (x_max - x_min) * (y_max - y_min)
    return n_cells * effort, n_cells


def posterior_update(prior, mission, depth_arr, rough_arr,
                     kappa=0.9, gamma_d=1.8, gamma_r=1.5):
    """
    Apply one Bayesian update.

    Parameters
    ----------
    prior : (NY, NX) probability array.
    mission : dict with keys 'x_min','x_max','y_min','y_max','effort','s'.
              's' is 1 if detected, 0 if not.
    depth_arr, rough_arr : (NY, NX) arrays.

    Returns
    -------
    posterior : (NY, NX) probability array (normalized).
    """
    mask = coverage_mask(mission["x_min"], mission["x_max"],
                         mission["y_min"], mission["y_max"])
    rho = detection_probability(mission["effort"], depth_arr, rough_arr,
                                kappa, gamma_d, gamma_r)
    # q_tj = c_tj * rho_tj
    q = np.where(mask, rho, 0.0)

    s = mission["s"]
    if s == 1:
        likelihood = q                # P(s=1 | Z=j) = q_tj
    else:
        likelihood = 1.0 - q          # P(s=0 | Z=j) = 1 - q_tj

    unnormalized = prior * likelihood
    Z = unnormalized.sum()
    if Z <= 0:
        raise RuntimeError(
            "Posterior is zero everywhere -- model and observation are "
            "inconsistent. Check the detection parameters or the observation."
        )
    return unnormalized / Z


# ---------------------------------------------------------------------------
# Mission selection
# ---------------------------------------------------------------------------
def expected_detection(prior, x_min, x_max, y_min, y_max, effort,
                       depth_arr, rough_arr,
                       kappa=0.9, gamma_d=1.8, gamma_r=1.5):
    """
    Expected probability of detecting the object with this mission:
        P(s=1) = sum_{j in R} pi_j * rho_tj
    """
    rho = detection_probability(effort, depth_arr[y_min:y_max, x_min:x_max],
                                rough_arr[y_min:y_max, x_min:x_max],
                                kappa, gamma_d, gamma_r)
    prior_sub = prior[y_min:y_max, x_min:x_max]
    return float(np.sum(prior_sub * rho))


def _coerce_mission_dict(d):
    """
    Cast the rectangle / effort fields of a mission dict to int.
    Needed because pandas .iloc[i].to_dict() upcasts everything to float
    if any column is float, and numpy slicing requires int indices.
    """
    out = dict(d)
    for k in ("x_min", "x_max", "y_min", "y_max", "effort", "n_cells", "cost"):
        if k in out:
            out[k] = int(out[k])
    if "s" in out:
        out["s"] = int(out["s"])
    return out


def search_best_mission(
    prior,
    depth_arr,
    rough_arr,
    budget_remaining,
    efforts=(1, 2, 3),
    min_width=3,
    max_width=20,
    min_height=3,
    max_height=20,
    min_cells=1,
    min_p_detect=0.0,
    step=2,
    kappa=0.9, gamma_d=1.8, gamma_r=1.5,
    verbose=False,
):
    """
    Brute-force search over rectangular missions for the one that maximizes
    expected detection per unit cost, subject to cost <= budget_remaining.

    Constraints
    -----------
    - `min_cells`     : reject rectangles smaller than this many cells.
    - `min_p_detect`  : reject candidates whose expected P(detect) is below
                        this threshold. Use this for *exploration* missions
                        to avoid degenerate tiny missions that maximize the
                        efficiency ratio at the cost of being uninformative.
                        Setting it to 0 (default) recovers the original
                        unconstrained behaviour.
    - `step`          : grid step for candidate corners. step=1 is exhaustive.

    Returns
    -------
    best : dict with keys x_min, x_max, y_min, y_max (all int), effort (int),
           cost (int), n_cells (int), p_detect (float), efficiency (float).
    top_candidates : DataFrame sorted by efficiency descending (after filters).
    """
    candidates = []
    for effort in efforts:
        for x_min in range(0, NX, step):
            for x_max in range(x_min + min_width,
                               min(NX, x_min + max_width) + 1, step):
                w = x_max - x_min
                for y_min in range(0, NY, step):
                    for y_max in range(y_min + min_height,
                                       min(NY, y_min + max_height) + 1, step):
                        h = y_max - y_min
                        n_cells = w * h
                        if n_cells < min_cells:
                            continue
                        cost = n_cells * effort
                        if cost > budget_remaining:
                            continue
                        p_det = expected_detection(
                            prior, x_min, x_max, y_min, y_max, effort,
                            depth_arr, rough_arr, kappa, gamma_d, gamma_r,
                        )
                        if p_det < min_p_detect:
                            continue
                        eff = p_det / cost if cost > 0 else 0.0
                        candidates.append({
                            "x_min": x_min, "x_max": x_max,
                            "y_min": y_min, "y_max": y_max,
                            "effort": effort, "n_cells": n_cells, "cost": cost,
                            "p_detect": p_det, "efficiency": eff,
                        })

    if not candidates:
        raise RuntimeError(
            f"No feasible mission within budget={budget_remaining}, "
            f"min_cells={min_cells}, min_p_detect={min_p_detect}. "
            "Try relaxing one of the constraints."
        )

    df = pd.DataFrame(candidates).sort_values("efficiency", ascending=False)
    best = _coerce_mission_dict(df.iloc[0].to_dict())
    if verbose:
        print("Top 10 candidate missions by efficiency:")
        print(df.head(10).to_string(index=False))
    return best, df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_grid_features(grid_df, ax=None):
    """Side-by-side plot of depth and roughness."""
    depth_arr, rough_arr, xs, ys = grid_to_arrays(grid_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for arr, name, ax_i in zip([depth_arr, rough_arr],
                                ["Depth", "Roughness"], axes):
        im = ax_i.imshow(arr, origin="lower", extent=[0, NX, 0, NY],
                          cmap="viridis", aspect="equal")
        ax_i.set_title(name)
        ax_i.set_xlabel("x"); ax_i.set_ylabel("y")
        plt.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, axes


def plot_distribution(dist, title="Distribution", ax=None,
                       missions=None, samples=None, show_accident=True,
                       log_scale=False, cmap="viridis"):
    """
    Plot a probability distribution as a heatmap, optionally overlaying:
      - past missions as rectangles (red dashed for s=0, green solid for s=1),
      - sampled landing points (small dots),
      - accident location and physical vectors.

    Parameters
    ----------
    dist : (NY, NX) array.
    missions : list of dicts (each must include x_min, x_max, y_min, y_max, s).
    samples : (n, 2) array of (x, y) points to overlay.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 7))
    else:
        fig = ax.figure

    norm = LogNorm(vmin=max(dist[dist > 0].min(), 1e-8), vmax=dist.max()) \
        if log_scale else None
    im = ax.imshow(dist, origin="lower", extent=[0, NX, 0, NY],
                   cmap=cmap, aspect="equal", norm=norm)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="probability")

    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.05,
                   color="white")

    if show_accident:
        ax.scatter(X_E[0], X_E[1], color="red", s=80, zorder=5,
                   label="Accident", edgecolor="black")
        ax.quiver(X_E[0], X_E[1], V_PLANE[0], V_PLANE[1],
                  angles="xy", scale_units="xy", scale=1,
                  color="blue", width=0.004, label="Plane")
        ax.quiver(X_E[0], X_E[1], V_WIND[0], V_WIND[1],
                  angles="xy", scale_units="xy", scale=1,
                  color="cyan", width=0.004, label="Wind")
        ax.quiver(X_E[0], X_E[1], V_DRIFT[0], V_DRIFT[1],
                  angles="xy", scale_units="xy", scale=1,
                  color="magenta", width=0.004, label="Drift")

    if missions:
        for i, m in enumerate(missions):
            color = "lime" if m.get("s", 0) == 1 else "red"
            ls = "-" if m.get("s", 0) == 1 else "--"
            rect = Rectangle(
                (m["x_min"], m["y_min"]),
                m["x_max"] - m["x_min"],
                m["y_max"] - m["y_min"],
                fill=False, edgecolor=color, linestyle=ls, linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(m["x_min"] + 0.3, m["y_max"] - 0.8,
                    f"M{i+1} (e={m['effort']}, s={m['s']})",
                    color=color, fontsize=9, fontweight="bold")

    ax.set_xlim(0, NX); ax.set_ylim(0, NY)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.15)
    return fig, ax


def plot_proposed_mission(prior, mission_proposal, depth_arr, rough_arr,
                           past_missions=None,
                           kappa=0.9, gamma_d=1.8, gamma_r=1.5):
    """
    Visualize a proposed mission: posterior heatmap with the proposed rectangle
    highlighted, plus a small panel showing rho inside the rectangle.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_distribution(prior, title="Current posterior + proposed mission",
                      ax=axes[0], missions=past_missions or [])
    m = mission_proposal
    rect = Rectangle(
        (m["x_min"], m["y_min"]),
        m["x_max"] - m["x_min"],
        m["y_max"] - m["y_min"],
        fill=False, edgecolor="orange", linestyle="-", linewidth=3,
    )
    axes[0].add_patch(rect)
    axes[0].text(m["x_min"] + 0.3, m["y_max"] - 0.8,
                 f"NEW (e={m['effort']})", color="orange",
                 fontsize=10, fontweight="bold")

    rho_sub = detection_probability(
        m["effort"],
        depth_arr[m["y_min"]:m["y_max"], m["x_min"]:m["x_max"]],
        rough_arr[m["y_min"]:m["y_max"], m["x_min"]:m["x_max"]],
        kappa, gamma_d, gamma_r,
    )
    im = axes[1].imshow(
        rho_sub, origin="lower",
        extent=[m["x_min"], m["x_max"], m["y_min"], m["y_max"]],
        cmap="plasma", aspect="equal", vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=axes[1], fraction=0.04, pad=0.04, label=r"$\rho_{tj}$")
    axes[1].set_title(
        f"Detection probability inside the mission "
        f"(e={m['effort']}, cost={m['cost']}, P(detect)={m['p_detect']:.3f})"
    )
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Mission logbook
# ---------------------------------------------------------------------------
def make_mission_record(team_id, mission_id, mission, budget_remaining,
                         timestamp=None):
    """
    Build a row for the missions.csv log as required by the deliverable.
    """
    if timestamp is None:
        timestamp = pd.Timestamp.utcnow().isoformat()
    return {
        "team_id": team_id,
        "mission_id": mission_id,
        "x_min": mission["x_min"],
        "x_max": mission["x_max"],
        "y_min": mission["y_min"],
        "y_max": mission["y_max"],
        "effort": mission["effort"],
        "n_cells": (mission["x_max"] - mission["x_min"]) *
                    (mission["y_max"] - mission["y_min"]),
        "cost": (mission["x_max"] - mission["x_min"]) *
                 (mission["y_max"] - mission["y_min"]) * mission["effort"],
        "s_t": mission["s"],
        "budget_remaining": budget_remaining,
        "timestamp": timestamp,
    }


def save_missions_log(records, path="missions.csv"):
    """Save the mission history."""
    pd.DataFrame(records).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Simulation (for strategy validation)
# ---------------------------------------------------------------------------
def simulate_strategy(
    prior_initial,
    depth_arr,
    rough_arr,
    n_runs=200,
    n_missions_max=8,
    budget=BUDGET_TOTAL,
    kappa=0.9, gamma_d=1.8, gamma_r=1.5,
    search_step=4,
    min_cells=40,
    min_p_detect=0.10,
    rng=None,
):
    """
    Simulate the full pipeline n_runs times to validate the strategy.

    Same `min_cells` / `min_p_detect` semantics as `search_best_mission`.
    Defaults are chosen for a meaningful per-mission information gain.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    flat_prior = prior_initial.ravel()
    n_cells_total = NY * NX

    detections = 0
    missions_to_detect = []
    budget_used_list = []

    for run in range(n_runs):
        z_flat = rng.choice(n_cells_total, p=flat_prior)
        z_iy, z_ix = divmod(z_flat, NX)

        posterior = prior_initial.copy()
        budget_remaining = budget
        detected = False
        for m_idx in range(n_missions_max):
            try:
                best, _ = search_best_mission(
                    posterior, depth_arr, rough_arr,
                    budget_remaining=budget_remaining,
                    step=search_step,
                    min_cells=min_cells, min_p_detect=min_p_detect,
                    kappa=kappa, gamma_d=gamma_d, gamma_r=gamma_r,
                )
            except RuntimeError:
                # Relax constraints if nothing feasible -- fall back to
                # the unconstrained best, which may still find the object.
                try:
                    best, _ = search_best_mission(
                        posterior, depth_arr, rough_arr,
                        budget_remaining=budget_remaining,
                        step=search_step,
                        min_cells=1, min_p_detect=0.0,
                        kappa=kappa, gamma_d=gamma_d, gamma_r=gamma_r,
                    )
                except RuntimeError:
                    break

            inside = (best["x_min"] <= z_ix < best["x_max"] and
                      best["y_min"] <= z_iy < best["y_max"])
            if inside:
                rho_z = detection_probability(
                    best["effort"], depth_arr[z_iy, z_ix],
                    rough_arr[z_iy, z_ix], kappa, gamma_d, gamma_r,
                )
                s = int(rng.random() < rho_z)
            else:
                s = 0

            mission = {**best, "s": s}
            posterior = posterior_update(
                posterior, mission, depth_arr, rough_arr,
                kappa, gamma_d, gamma_r,
            )
            budget_remaining -= best["cost"]
            if s == 1:
                detected = True
                detections += 1
                missions_to_detect.append(m_idx + 1)
                budget_used_list.append(budget - budget_remaining)
                break

        if not detected:
            budget_used_list.append(budget - budget_remaining)

    return {
        "n_runs": n_runs,
        "detection_rate": detections / n_runs,
        "mean_missions_to_detect": (np.mean(missions_to_detect)
                                    if missions_to_detect else None),
        "mean_budget_used": float(np.mean(budget_used_list)),
    }
