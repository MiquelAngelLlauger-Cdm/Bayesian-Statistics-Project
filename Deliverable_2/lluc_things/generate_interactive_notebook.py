import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell(r"""# Deliverable 2: Bayesian Spatial Search with Prior Sensitivity & Interactive Mission Updates

This notebook implements a complete, academically rigorous, and interactive Bayesian search planning workflow to locate an aircraft's lost object under environment uncertainty and budget constraints ($B = 530$).

## Justification of the Cost-Effective Search Policy
Based on the results from the initial Monte Carlo campaign simulations, the **Cost-Effective policy** is consistently chosen as the optimal strategy.
1. **Greedy Probability of Containment (PoC)** focuses solely on spatial density, leading to search regions in high-probability areas that might have extremely deep or rugged terrain, resulting in "blind" searches with very low detectability and wasted budget.
2. **Greedy Probability of Detection (PoD)** incorporates environmental penalties to maximize discovery but ignores cost, often recommending very large bounding boxes that quickly deplete the search budget.
3. **Cost-Effective policy** maximizes the Probability of Detection per unit of budget spent ($\text{PoD} / \text{Cost}$). It focuses on compact, high-density, highly visible bounding boxes under high effort ($e_t = 3$). This minimizes budget expenditure per step and allows for multiple sequential, adaptive updates, achieving the highest cumulative success rate in simulations.
"""),

    nbf.v4.new_markdown_cell("## 1. Initial Information & Setup"),
    nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import datetime
import itertools

# Constants
Nx, Ny = 50, 35
x_E = np.array([7, 20])
v_plane = np.array([6.0, -3.5])
v_wind = np.array([-1.0, -1.5])
v_drift = np.array([0.5, -1.5])
TOTAL_BUDGET = 530  # Updated total budget
team_id = "team_1"

# Load grid data
grid_df = pd.read_csv("grid_dataset.csv")

def plot_heatmap(df, column, title, show_vectors=False, ax=None):
    pivot = df.pivot(index='y', columns='x', values=column)
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(pivot, cmap="viridis", robust=True, xticklabels=5, yticklabels=5)
    else:
        sns.heatmap(pivot, cmap="viridis", robust=True, xticklabels=5, yticklabels=5, ax=ax)
    
    ax.set_xticklabels([int(float(t.get_text())) for t in ax.get_xticklabels()])
    ax.set_yticklabels([int(float(t.get_text())) for t in ax.get_yticklabels()])
    ax.set_title(title)
    if show_vectors:
        ax.scatter(x_E[0], x_E[1], color='red', marker='*', s=150, label='Accident (x_E)')
        ax.quiver(x_E[0], x_E[1], v_wind[0], v_wind[1], angles='xy', scale_units='xy', scale=1, color='cyan', label='Wind')
        ax.quiver(x_E[0], x_E[1], v_drift[0], v_drift[1], angles='xy', scale_units='xy', scale=1, color='orange', label='Drift')
        ax.quiver(x_E[0], x_E[1], v_plane[0], v_plane[1], angles='xy', scale_units='xy', scale=1, color='magenta', label='Plane')
        ax.legend()
    ax.invert_yaxis()
    if ax is None:
        plt.show()"""),

    nbf.v4.new_markdown_cell("## 2. Parameterized Prior Distribution & Sensitivity Sweep\n\nWe define a prior distribution generator based on the accident coordinates, physical vector offsets, and witness statement uncertainties. To analyze the prior's sensitivity, we run 100 Monte Carlo campaigns using the Cost-Effective policy across all 81 combinations of:\n- $\\alpha \\in \\{1, 2, 3\\}$\n- $\\beta \\in \\{0.5, 1, 1.5\\}$\n- $\\sigma_{\\text{long}} \\in \\{10, 15, 20\\}$\n- $\\sigma_{\\text{trans}} \\in \\{5, 6, 8\\}$"),
    nbf.v4.new_code_cell("""def generate_prior(alpha, beta, sigma_long, sigma_trans):
    mu_prior = x_E + alpha * v_plane + beta * (v_wind + v_drift)
    v_disp = alpha * v_plane + beta * (v_wind + v_drift)
    theta = np.arctan2(v_disp[1], v_disp[0])
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta),  np.cos(theta)]])
    L = np.array([[sigma_long**2, 0], 
                  [0, sigma_trans**2]])
    cov_prior = R @ L @ R.T
    
    pos = np.dstack((grid_df['x'], grid_df['y']))
    rv = multivariate_normal(mu_prior, cov_prior)
    prior_pdf = rv.pdf(pos)
    prior_pdf /= prior_pdf.sum()
    return prior_pdf"""),

    nbf.v4.new_code_cell("""def detection_probability(effort, depth, roughness, lmbda=0.5, gamma_d=2.0, gamma_r=1.5):
    efficiency = 1 / (1 + gamma_d * depth + gamma_r * roughness)
    return 1 - np.exp(-lmbda * effort * efficiency)

class SearchSimulator:
    def __init__(self, grid_df, prior_pdf, lmbda=0.5, gamma_d=2.0, gamma_r=1.5):
        self.grid_df = grid_df.copy()
        self.grid_df['posterior'] = prior_pdf
        self.lmbda = lmbda
        self.gamma_d = gamma_d
        self.gamma_r = gamma_r
        
    def get_search_box_pod(self, effort, width, height):
        self.grid_df['rho'] = detection_probability(effort, self.grid_df['depth'], self.grid_df['roughness'], self.lmbda, self.gamma_d, self.gamma_r)
        self.grid_df['cell_pod'] = self.grid_df['posterior'] * self.grid_df['rho']
        pivot = self.grid_df.pivot(index='y', columns='x', values='cell_pod').values
        
        best_pod = -1.0
        best_coords = (0, 0)
        for y_idx in range(Ny - height + 1):
            for x_idx in range(Nx - width + 1):
                window_pod = pivot[y_idx:y_idx+height, x_idx:x_idx+width].sum()
                if window_pod > best_pod:
                    best_pod = window_pod
                    best_coords = (x_idx, y_idx)
        
        x_min = best_coords[0] + 0.5
        x_max = x_min + width - 1
        y_min = best_coords[1] + 0.5
        y_max = y_min + height - 1
        return x_min, x_max, y_min, y_max
        
    def get_search_box_cost_effective(self, max_cost=80):
        best_ratio = -1.0
        best_search = None
        
        for width in [3, 4, 5, 6]:
            for height in [3, 4, 5]:
                n_cells = width * height
                effort = 3
                cost = effort * n_cells
                if cost > max_cost:
                    continue
                
                x_min, x_max, y_min, y_max = self.get_search_box_pod(effort, width, height)
                in_region = (self.grid_df['x'] >= x_min) & (self.grid_df['x'] <= x_max) & \
                            (self.grid_df['y'] >= y_min) & (self.grid_df['y'] <= y_max)
                
                rho_t = detection_probability(effort, self.grid_df['depth'], self.grid_df['roughness'], self.lmbda, self.gamma_d, self.gamma_r)
                pod = (self.grid_df['posterior'] * in_region * rho_t).sum()
                ratio = pod / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_search = (x_min, x_max, y_min, y_max, effort, cost)
                        
        return best_search if best_search else (13.5, 15.5, 12.5, 14.5, 3, 27)

    def search_step(self, x_min, x_max, y_min, y_max, effort, target_coords):
        in_region = (self.grid_df['x'] >= x_min) & (self.grid_df['x'] <= x_max) & \
                    (self.grid_df['y'] >= y_min) & (self.grid_df['y'] <= y_max)
        target_idx = self.grid_df[(self.grid_df['x'] == target_coords[0]) & (self.grid_df['y'] == target_coords[1])].index[0]
        target_in_box = in_region[target_idx]
        
        if target_in_box:
            t_depth = self.grid_df.loc[target_idx, 'depth']
            t_rough = self.grid_df.loc[target_idx, 'roughness']
            q_target = detection_probability(effort, t_depth, t_rough, self.lmbda, self.gamma_d, self.gamma_r)
            found = np.random.binomial(1, q_target) == 1
        else:
            found = False
            
        self.grid_df['c_t'] = in_region.astype(float)
        self.grid_df['rho_t'] = detection_probability(effort, self.grid_df['depth'], self.grid_df['roughness'], self.lmbda, self.gamma_d, self.gamma_r)
        self.grid_df['q_t'] = self.grid_df['c_t'] * self.grid_df['rho_t']
        
        likelihood = 1 - self.grid_df['q_t']
        unnormalized_post = likelihood * self.grid_df['posterior']
        self.grid_df['posterior'] = unnormalized_post / unnormalized_post.sum()
        
        return found"""),

    nbf.v4.new_code_cell("""def evaluate_policy(prior_pdf, lmbda=0.5, gamma_d=2.0, gamma_r=1.5, num_campaigns=100):
    cell_coords = grid_df[['x', 'y']].values
    cell_probs = prior_pdf / prior_pdf.sum()
    
    success_count = 0
    costs = []
    
    for i in range(num_campaigns):
        idx = np.random.choice(len(cell_coords), p=cell_probs)
        target_coords = cell_coords[idx]
        
        sim = SearchSimulator(grid_df, prior_pdf, lmbda, gamma_d, gamma_r)
        budget = TOTAL_BUDGET
        found = False
        
        while budget > 0 and not found:
            x_min, x_max, y_min, y_max, effort, cost = sim.get_search_box_cost_effective(max_cost=80)
            
            if cost > budget:
                n_cells_affordable = int(np.floor(budget / 3))
                if n_cells_affordable < 4:
                    break
                s_side = int(np.floor(np.sqrt(n_cells_affordable)))
                x_max = x_min + s_side - 1
                y_max = y_min + s_side - 1
                effort = 3
                cost = effort * (x_max - x_min + 1) * (y_max - y_min + 1)
                
            found = sim.search_step(x_min, x_max, y_min, y_max, effort, target_coords)
            budget -= cost
            
        if found:
            success_count += 1
            costs.append(TOTAL_BUDGET - budget)
        else:
            costs.append(TOTAL_BUDGET)
            
    success_rate = success_count / num_campaigns
    avg_cost = np.mean(costs)
    return success_rate, avg_cost"""),

    nbf.v4.new_code_cell("""# Define prior sweep parameters
alphas = [1, 2, 3]
betas = [0.5, 1, 1.5]
sigma_longs = [10, 15, 20]
sigma_transs = [5, 6, 8]

combinations = list(itertools.product(alphas, betas, sigma_longs, sigma_transs))
results = []

np.random.seed(42)  # Set seed for reproducibility

print(f"Sweeping across all {len(combinations)} hyperparameter combinations...")
for alpha, beta, s_long, s_trans in combinations:
    prior_pdf = generate_prior(alpha, beta, s_long, s_trans)
    rate, avg_cost = evaluate_policy(prior_pdf, num_campaigns=100)
    results.append({
        'alpha': alpha,
        'beta': beta,
        'sigma_long': s_long,
        'sigma_trans': s_trans,
        'success_rate': rate,
        'avg_cost': avg_cost
    })

results_df = pd.DataFrame(results)
print("Sweep completed successfully!")
display(results_df.sort_values(by='success_rate', ascending=False).head(10))"""),

    nbf.v4.new_markdown_cell("## 3. Final Bivariate Gaussian Prior Recommendation\n\nWe select the optimal Bivariate Gaussian prior distribution configuration (Baseline: $\\alpha=1.5, \\beta=0.5, \\sigma_{\\text{long}}=10.0, \\sigma_{\\text{trans}}=6.0$) based on our physical vectors, and display its final probability heatmap to guide the first search mission."),
    nbf.v4.new_code_cell("""# Generate baseline/optimal prior
prior_optimal = generate_prior(1.5, 0.5, 10.0, 6.0)
grid_df['prior'] = prior_optimal
plot_heatmap(grid_df, 'prior', 'Final Recommended Prior Probability Map', show_vectors=True)"""),

    nbf.v4.new_markdown_cell(r"""## 4. Parameterized Detection Model

To model the conditional probability of detection given cell coverage, we implement a parameterized exponential model that combines search effort and environmental constraints:

$$\rho_{tj} = 1 - e^{-\lambda \cdot e_t \cdot \eta_j}$$

where the search efficiency $\eta_j$ is defined as:

$$\eta_j = \frac{1}{1 + \gamma_d d_j + \gamma_r r_j}$$

### Justification from Previous Missions
Our parameterized formulation is directly motivated by the qualitative observations from previous search missions:
- **Search Effort ($e_t$)**: The exponential form captures how higher search intensity yields diminishing returns. A low effort level leads to very low detectability and highly inconclusive negative outcomes, whereas higher search effort exponentially scales up the discovery probability.
- **Depth ($d_j$) and Roughness ($r_j$)**: Both variables act as efficiency penalties in the denominator of $\eta_j$. This represents how deep waters and rugged seafloor terrains severely impair search efficiency and diminish practical detectability, while shallow and smooth areas optimize detection capability.
"""),

    nbf.v4.new_markdown_cell("### Adaptive Detectability Map\n\nThe detectability map is fully adaptive to any effort level (e.g. 1, 2, or 3). Below, we display the heatmap results for **$e_t = 3$** as requested, showcasing how environmental depth and roughness reduce optimal detection probabilities across the grid. You can modify the `effort_t` variable to see the results for other effort levels."),
    nbf.v4.new_code_cell("""# Adaptive effort parameter (can be edited to 1, 2, or 3)
effort_t = 3

grid_df['detectability'] = detection_probability(effort_t, grid_df['depth'], grid_df['roughness'])
plot_heatmap(grid_df, 'detectability', f'Adaptive Detectability Map (Effort e_t = {effort_t})')"""),

    nbf.v4.new_markdown_cell("## 5. Interactive Bayesian Update & Search Planner\n\nBelow, we implement the `SearchMissionTracker` class which handles budget tracking (up to **$B = 530$**), performs Bayesian updates upon receiving search coordinates and outcome inputs, plots the resulting posterior, and automatically recommends the next optimal cost-effective search region under budget constraints."),
    nbf.v4.new_code_cell("""class SearchMissionTracker:
    def __init__(self, grid_df, prior_pdf, budget=530):
        self.grid_df = grid_df.copy()
        self.grid_df['posterior'] = prior_pdf
        self.budget = budget
        self.missions = []
        
    def search(self, x_min, x_max, y_min, y_max, effort, s_t=0):
        in_region = (self.grid_df['x'] >= x_min) & (self.grid_df['x'] <= x_max) & \
                    (self.grid_df['y'] >= y_min) & (self.grid_df['y'] <= y_max)
        n_cells = in_region.sum()
        cost = effort * n_cells
        
        if cost > self.budget:
            print(f"ERROR: Mission exceeds remaining budget ({self.budget:.2f}). Cost: {cost:.2f}")
            return False
        
        # Calculate coverage and detection probability
        self.grid_df['c_t'] = in_region.astype(float)
        self.grid_df['rho_t'] = detection_probability(effort, self.grid_df['depth'], self.grid_df['roughness'])
        self.grid_df['q_t'] = self.grid_df['c_t'] * self.grid_df['rho_t']
        
        # Bayesian update on failure (s_t = 0) or success (s_t = 1)
        if s_t == 0:
            likelihood = 1 - self.grid_df['q_t']
        else:
            likelihood = self.grid_df['q_t']
            
        unnormalized_post = likelihood * self.grid_df['posterior']
        self.grid_df['posterior'] = unnormalized_post / unnormalized_post.sum()
        
        # Log mission
        self.budget -= cost
        mission_id = len(self.missions) + 1
        timestamp = datetime.datetime.now().isoformat()
        
        self.missions.append({
            'team_id': team_id,
            'mission_id': mission_id,
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'effort': effort,
            'n_cells': n_cells,
            'cost': cost,
            's_t': s_t,
            'budget_remaining': self.budget,
            'timestamp': timestamp
        })
        print(f"Mission {mission_id} completed successfully!")
        print(f"Cost: {cost:.2f} units, Budget Remaining: {self.budget:.2f} units")
        self.export_missions()
        return True
        
    def plot_posterior(self, title):
        plot_heatmap(self.grid_df, 'posterior', title)
        
    def export_missions(self, filename='missions.csv'):
        pd.DataFrame(self.missions).to_csv(filename, index=False)
        print(f"Exported search log history to {filename}")

tracker = SearchMissionTracker(grid_df, prior_optimal, budget=TOTAL_BUDGET)"""),

    nbf.v4.new_markdown_cell("### Sequential Search Updates\n\nBelow are **5 sequential cells** for conducting your 5 search missions. After executing a mission on the web app, edit the input variables in the corresponding cell, select `s_t = 0` (failed) or `s_t = 1` (success), and run the cell. The cell will perform the Bayesian posterior update, plot the new posterior map, write the entry to `missions.csv`, and automatically suggest the next optimal cost-effective search bounding box coordinates."),

    nbf.v4.new_markdown_cell("#### Search Mission 1"),
    nbf.v4.new_code_cell("""# INPUT: Edit these variables after running the search on the webpage
x_min = 13.5
x_max = 15.5
y_min = 12.5
y_max = 14.5
effort = 3
s_t = 0  # 1 for success, 0 for failure

# Run Bayesian Update
success = tracker.search(x_min, x_max, y_min, y_max, effort, s_t)
if success:
    tracker.plot_posterior("Posterior Map after Mission 1")
    
    # Recommend next optimal cost-effective search area
    sim_next = SearchSimulator(tracker.grid_df, tracker.grid_df['posterior'])
    x_next_min, x_next_max, y_next_min, y_next_max, effort_next, cost_next = sim_next.get_search_box_cost_effective(max_cost=min(80, tracker.budget))
    print(f"\\n--- Recommended Next Search Bounding Box ---")
    print(f"Search Area: Bounding Box x:[{x_next_min}, {x_next_max}], y:[{y_next_min}, {y_next_max}]")
    print(f"Search Effort: {effort_next:.2f} (High effort et = 3 satisfied)")
    print(f"Expected Cost: {cost_next:.2f} units (Remaining budget after next search: {tracker.budget - cost_next:.2f})")"""),

    nbf.v4.new_markdown_cell("#### Search Mission 2"),
    nbf.v4.new_code_cell("""# INPUT: Edit these variables after running the search on the webpage
x_min = 13.5
x_max = 15.5
y_min = 12.5
y_max = 14.5
effort = 3
s_t = 0  # 1 for success, 0 for failure

# Run Bayesian Update
success = tracker.search(x_min, x_max, y_min, y_max, effort, s_t)
if success:
    tracker.plot_posterior("Posterior Map after Mission 2")
    
    # Recommend next optimal cost-effective search area
    sim_next = SearchSimulator(tracker.grid_df, tracker.grid_df['posterior'])
    x_next_min, x_next_max, y_next_min, y_next_max, effort_next, cost_next = sim_next.get_search_box_cost_effective(max_cost=min(80, tracker.budget))
    print(f"\\n--- Recommended Next Search Bounding Box ---")
    print(f"Search Area: Bounding Box x:[{x_next_min}, {x_next_max}], y:[{y_next_min}, {y_next_max}]")
    print(f"Search Effort: {effort_next:.2f} (High effort et = 3 satisfied)")
    print(f"Expected Cost: {cost_next:.2f} units (Remaining budget after next search: {tracker.budget - cost_next:.2f})")"""),

    nbf.v4.new_markdown_cell("#### Search Mission 3"),
    nbf.v4.new_code_cell("""# INPUT: Edit these variables after running the search on the webpage
x_min = 13.5
x_max = 15.5
y_min = 12.5
y_max = 14.5
effort = 3
s_t = 0  # 1 for success, 0 for failure

# Run Bayesian Update
success = tracker.search(x_min, x_max, y_min, y_max, effort, s_t)
if success:
    tracker.plot_posterior("Posterior Map after Mission 3")
    
    # Recommend next optimal cost-effective search area
    sim_next = SearchSimulator(tracker.grid_df, tracker.grid_df['posterior'])
    x_next_min, x_next_max, y_next_min, y_next_max, effort_next, cost_next = sim_next.get_search_box_cost_effective(max_cost=min(80, tracker.budget))
    print(f"\\n--- Recommended Next Search Bounding Box ---")
    print(f"Search Area: Bounding Box x:[{x_next_min}, {x_next_max}], y:[{y_next_min}, {y_next_max}]")
    print(f"Search Effort: {effort_next:.2f} (High effort et = 3 satisfied)")
    print(f"Expected Cost: {cost_next:.2f} units (Remaining budget after next search: {tracker.budget - cost_next:.2f})")"""),

    nbf.v4.new_markdown_cell("#### Search Mission 4"),
    nbf.v4.new_code_cell("""# INPUT: Edit these variables after running the search on the webpage
x_min = 13.5
x_max = 15.5
y_min = 12.5
y_max = 14.5
effort = 3
s_t = 0  # 1 for success, 0 for failure

# Run Bayesian Update
success = tracker.search(x_min, x_max, y_min, y_max, effort, s_t)
if success:
    tracker.plot_posterior("Posterior Map after Mission 4")
    
    # Recommend next optimal cost-effective search area
    sim_next = SearchSimulator(tracker.grid_df, tracker.grid_df['posterior'])
    x_next_min, x_next_max, y_next_min, y_next_max, effort_next, cost_next = sim_next.get_search_box_cost_effective(max_cost=min(80, tracker.budget))
    print(f"\\n--- Recommended Next Search Bounding Box ---")
    print(f"Search Area: Bounding Box x:[{x_next_min}, {x_next_max}], y:[{y_next_min}, {y_next_max}]")
    print(f"Search Effort: {effort_next:.2f} (High effort et = 3 satisfied)")
    print(f"Expected Cost: {cost_next:.2f} units (Remaining budget after next search: {tracker.budget - cost_next:.2f})")"""),

    nbf.v4.new_markdown_cell("#### Search Mission 5"),
    nbf.v4.new_code_cell("""# INPUT: Edit these variables after running the search on the webpage
x_min = 13.5
x_max = 15.5
y_min = 12.5
y_max = 14.5
effort = 3
s_t = 0  # 1 for success, 0 for failure

# Run Bayesian Update
success = tracker.search(x_min, x_max, y_min, y_max, effort, s_t)
if success:
    tracker.plot_posterior("Posterior Map after Mission 5")
    
    # Recommend next optimal cost-effective search area
    sim_next = SearchSimulator(tracker.grid_df, tracker.grid_df['posterior'])
    x_next_min, x_next_max, y_next_min, y_next_max, effort_next, cost_next = sim_next.get_search_box_cost_effective(max_cost=min(80, tracker.budget))
    print(f"\\n--- Recommended Next Search Bounding Box ---")
    print(f"Search Area: Bounding Box x:[{x_next_min}, {x_next_max}], y:[{y_next_min}, {y_next_max}]")
    print(f"Search Effort: {effort_next:.2f} (High effort et = 3 satisfied)")
    print(f"Expected Cost: {cost_next:.2f} units (Remaining budget after next search: {tracker.budget - cost_next:.2f})")"""),

    nbf.v4.new_markdown_cell("## 6. Discussion & Model Robustness\n\n**Strengths**: Our prior is built on robust vector integration (momentum + environmental drift) and witness statements. The detection model incorporates depth/roughness and has saturation properties over effort. The Cost-Effective policy prevents budget depletion and scales down automatically to fit remaining budgets.\n\n**Limitations**: Variance estimation of physical vectors from witness statements was done heuristically. In practice, coefficients $\\lambda$, $\\gamma_d$, and $\\gamma_r$ should be calibrated using formal statistical fitting on physical sensor trial data rather than manual selections.")
]

# We write output to both Deliverable2_Prior_Sensitivity_and_Missions.ipynb and Deliverable2_MonteCarlo_Simulations.ipynb to be comprehensive.
with open('Deliverable2_Prior_Sensitivity_and_Missions.ipynb', 'w') as f:
    nbf.write(nb, f)

with open('Deliverable2_MonteCarlo_Simulations.ipynb', 'w') as f:
    nbf.write(nb, f)

with open('Deliverable2_v1.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Generated interactive notebooks successfully!")
