"""
main.py
=======
Driver for the Bayesian spatial search problem.
Structured as cell blocks (use # %% in VSCode or Jupytext to convert to .ipynb).

Workflow:
  Cell 1: Setup, load grid, inspect features.
  Cell 2: Build the prior, visualize it.
  Cell 3: Detection model: calibrate against the four reports, visualize rho.
  Cell 4: Pick mission 1 using the prior. Submit on the webpage.
  Cell 5: Record outcome, update posterior, repeat for missions 2..8.
  Cell 6: Save missions.csv.
  Cell 7 (optional): Simulate strategy to validate.
"""

# %% Cell 1 -- Setup and grid inspection ------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    NX, NY, X_E, V_PLANE, V_WIND, V_DRIFT, BUDGET_TOTAL,
    load_grid, grid_to_arrays,
    build_prior, detection_probability, calibration_table,
    coverage_mask, mission_cost, posterior_update,
    expected_detection, search_best_mission,
    plot_grid_features, plot_distribution, plot_proposed_mission,
    make_mission_record, save_missions_log,
    simulate_strategy,
)

TEAM_ID = "M4204"

grid_df = load_grid("grid_dataset.csv")
print(f"Loaded {len(grid_df)} cells. Expected: {NX*NY} = {NX*NY}")
print(grid_df.describe())

depth_arr, rough_arr, xs, ys = grid_to_arrays(grid_df)
print(f"depth_arr shape: {depth_arr.shape}, range [{depth_arr.min():.3f}, {depth_arr.max():.3f}]")
print(f"rough_arr shape: {rough_arr.shape}, range [{rough_arr.min():.3f}, {rough_arr.max():.3f}]")

plot_grid_features(grid_df)
plt.show()


# %% Cell 2 -- Build the prior ---------------------------------------------
# Witness-informed parameters:
#   alpha_mean=1.2, alpha_std=0.4  -> Witness 2: "farther along the trajectory"
#                                     Witness 1: "kept moving forward" (rules out alpha~0)
#   tau_mean=1.0, tau_std=0.4      -> drift time uncertain but order ~1 unit
#   perp_std=1.0                   -> Witness 2: "slightly off to one side"
#   iso_std=1.5                    -> generic unmodeled noise

prior, samples = build_prior(
    n_samples=200_000,
    alpha_mean=1.2, alpha_std=0.4,
    tau_mean=1.0,   tau_std=0.4,
    iso_std=1.5,    perp_std=1.0,
    seed=42,
)

print(f"Prior mass inside grid: {prior.sum():.4f} (should be 1.0)")
iy, ix = np.unravel_index(np.argmax(prior), prior.shape)
print(f"Prior peak at cell ({ix}, {iy}) with mass {prior.max():.5f}")
print(f"Expected location E[Z] = "
      f"({np.sum(prior * (np.arange(NX) + 0.5)):.2f}, "
      f"{np.sum(prior * (np.arange(NY) + 0.5)[:, None]):.2f})")

plot_distribution(prior, title="Prior over the grid", samples=samples[:5000])
plt.show()


# %% Cell 3 -- Detection model: calibration and visualization ---------------
KAPPA, GAMMA_D, GAMMA_R = 0.9, 1.8, 1.5

calibration_table(KAPPA, GAMMA_D, GAMMA_R)

# Show rho for each effort level over the whole grid (with current parameters).
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax_i, e in zip(axes, [1, 2, 3]):
    rho_grid = detection_probability(e, depth_arr, rough_arr,
                                     KAPPA, GAMMA_D, GAMMA_R)
    im = ax_i.imshow(rho_grid, origin="lower", extent=[0, NX, 0, NY],
                     cmap="plasma", aspect="equal", vmin=0, vmax=1)
    ax_i.set_title(rf"$\rho_{{tj}}$ at effort e={e}")
    ax_i.set_xlabel("x"); ax_i.set_ylabel("y")
    plt.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()


# %% Cell 4 -- Propose mission 1 -------------------------------------------
# Run the brute-force search to get the efficiency-optimal first mission.
budget_remaining = BUDGET_TOTAL
mission_history = []   # list of mission dicts (with 's' filled in)
log_records = []       # list of rows for missions.csv

best, top_df = search_best_mission(
    prior, depth_arr, rough_arr,
    budget_remaining=budget_remaining,
    efforts=(1, 2, 3),
    min_width=4, max_width=18, min_height=4, max_height=14,
    step=2, kappa=KAPPA, gamma_d=GAMMA_D, gamma_r=GAMMA_R,
    verbose=True,
)
print("\nProposed mission 1:")
print(f"  x in [{best['x_min']}, {best['x_max']}), "
      f"y in [{best['y_min']}, {best['y_max']}), e={best['effort']}")
print(f"  cells={best['n_cells']}, cost={best['cost']}, "
      f"P(detect)={best['p_detect']:.3f}, efficiency={best['efficiency']:.4f}")

plot_proposed_mission(prior, best, depth_arr, rough_arr,
                       past_missions=mission_history,
                       kappa=KAPPA, gamma_d=GAMMA_D, gamma_r=GAMMA_R)
plt.show()


# %% Cell 5 -- After submitting on the webpage, record the outcome ---------
# >>> SUBMIT THE ABOVE MISSION on https://search-missions.streamlit.app/ <<<
# Then fill in the observation:
s_observed = 0   # <-- replace with the value returned by the webpage (0 or 1)

mission_done = {**best, "s": s_observed}
mission_history.append(mission_done)
budget_remaining -= mission_done["cost"]
print(f"After mission {len(mission_history)}: budget_remaining = {budget_remaining}")

log_records.append(make_mission_record(
    TEAM_ID, mission_id=len(mission_history),
    mission=mission_done, budget_remaining=budget_remaining,
))

posterior = posterior_update(prior, mission_done, depth_arr, rough_arr,
                              KAPPA, GAMMA_D, GAMMA_R)
print(f"Posterior peak: {posterior.max():.5f} at cell "
      f"{np.unravel_index(np.argmax(posterior), posterior.shape)[::-1]}")

plot_distribution(posterior, title=f"Posterior after mission {len(mission_history)}",
                  missions=mission_history)
plt.show()


# %% Cell 6 -- Propose mission 2 (and beyond) ------------------------------
# Re-run the search using the current posterior. Repeat this block for each
# subsequent mission, updating `posterior` and `mission_history` each time.

best, top_df = search_best_mission(
    posterior, depth_arr, rough_arr,
    budget_remaining=budget_remaining,
    efforts=(1, 2, 3),
    min_width=3, max_width=15, min_height=3, max_height=12,
    step=1, kappa=KAPPA, gamma_d=GAMMA_D, gamma_r=GAMMA_R,
    verbose=True,
)
print(f"\nProposed mission {len(mission_history) + 1}:")
print(f"  x in [{best['x_min']}, {best['x_max']}), "
      f"y in [{best['y_min']}, {best['y_max']}), e={best['effort']}")
print(f"  cells={best['n_cells']}, cost={best['cost']}, "
      f"P(detect)={best['p_detect']:.3f}")

plot_proposed_mission(posterior, best, depth_arr, rough_arr,
                       past_missions=mission_history,
                       kappa=KAPPA, gamma_d=GAMMA_D, gamma_r=GAMMA_R)
plt.show()

# >>> SUBMIT MISSION, then fill in the observation: <<<
s_observed = 0   # <-- replace with the value returned by the webpage

mission_done = {**best, "s": s_observed}
mission_history.append(mission_done)
budget_remaining -= mission_done["cost"]
log_records.append(make_mission_record(
    TEAM_ID, mission_id=len(mission_history),
    mission=mission_done, budget_remaining=budget_remaining,
))
posterior = posterior_update(posterior, mission_done, depth_arr, rough_arr,
                              KAPPA, GAMMA_D, GAMMA_R)
print(f"After mission {len(mission_history)}: budget_remaining = {budget_remaining}")

plot_distribution(posterior, title=f"Posterior after mission {len(mission_history)}",
                  missions=mission_history)
plt.show()


# %% Cell 7 -- Save the mission log ----------------------------------------
save_missions_log(log_records, "missions.csv")
print(f"Saved {len(log_records)} missions to missions.csv")
print(pd.DataFrame(log_records))


# %% Cell 8 (optional) -- Validate the strategy with simulation ------------
# Run this ONCE before doing real missions, to sanity-check the strategy.
# Uses the initial prior (not a partial posterior), simulates a true cell,
# and reports detection rate and average budget consumed.

stats = simulate_strategy(
    prior_initial=prior,
    depth_arr=depth_arr, rough_arr=rough_arr,
    n_runs=100, n_missions_max=8, budget=BUDGET_TOTAL,
    kappa=KAPPA, gamma_d=GAMMA_D, gamma_r=GAMMA_R,
    search_step=4,
)
print("Simulation results:")
for k, v in stats.items():
    print(f"  {k}: {v}")
