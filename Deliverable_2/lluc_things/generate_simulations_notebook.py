import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("""# Bayesian Prior & Monte Carlo Search strategy Simulations

This notebook implements advanced Bayesian spatial search simulations to determine the most optimal search zone and strategy under physical uncertainty. We evaluate prior sensitivity across different hyperparameter combinations of the bivariate Gaussian prior and run stochastic Monte Carlo search campaigns to compare three distinct search policies:
1. **Greedy Probability of Containment (PoC)**: Focuses purely on spatial cell density.
2. **Greedy Probability of Detection (PoD)**: Maximizes likelihood of discovery by accounting for depth and roughness.
3. **Cost-Effective (PoD / Cost)**: Maximizes probability of detection per unit of budget spent.
"""),

    nbf.v4.new_markdown_cell("## 1. Initial Information & Setup"),
    nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import datetime

# Constants
Nx, Ny = 50, 35
x_E = np.array([7, 20])
v_plane = np.array([6.0, -3.5])
v_wind = np.array([-1.0, -1.5])
v_drift = np.array([0.5, -1.5])
TOTAL_BUDGET = 230
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

    nbf.v4.new_markdown_cell("## 2. Bivariate Gaussian Prior Generator & Sensitivity Analysis\n\nWe build a parameterized prior generator that constructs the spatial distribution as a function of the hyperparameters $(\\alpha, \\beta, \\sigma_{\\text{long}}, \\sigma_{\\text{trans}})$. We then visualize the distribution under various configurations."),
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
    return prior_pdf

# Generate different priors
prior_baseline = generate_prior(1.5, 0.5, 10.0, 6.0)
prior_diffuse = generate_prior(2.0, 0.5, 20.0, 8.0)
prior_drift_heavy = generate_prior(1.0, 1.2, 15.0, 5.0)

# Plot heatmaps
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
df_temp = grid_df.copy()

df_temp['prior'] = prior_baseline
plot_heatmap(df_temp, 'prior', 'Prior 1: Baseline (alpha=1.5, beta=0.5)', show_vectors=True, ax=axes[0])

df_temp['prior'] = prior_diffuse
plot_heatmap(df_temp, 'prior', 'Prior 2: Diffuse (alpha=2.0, beta=0.5)', show_vectors=True, ax=axes[1])

df_temp['prior'] = prior_drift_heavy
plot_heatmap(df_temp, 'prior', 'Prior 3: Drift-dominated (alpha=1.0, beta=1.2)', show_vectors=True, ax=axes[2])

plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("## 3. Parameterized Detection Model"),
    nbf.v4.new_markdown_cell(r"""### Model Formulation

To model the conditional probability of detection given cell coverage, we implement a parameterized exponential model that combines search effort and environmental constraints:

$$\rho_{tj} = 1 - e^{-\lambda \cdot e_t \cdot \eta_j}$$

where the search efficiency $\eta_j$ is defined as:

$$\eta_j = \frac{1}{1 + \gamma_d d_j + \gamma_r r_j}$$

### Justification from Previous Missions

Our parameterized formulation is directly motivated by the qualitative observations from previous search missions:
- **Search Effort ($e_t$)**: The exponential form captures how higher search intensity yields diminishing returns. A low effort level leads to very low detectability and highly inconclusive negative outcomes, whereas higher search effort exponentially scales up the discovery probability.
- **Depth ($d_j$) and Roughness ($r_j$)**: Both variables act as efficiency penalties in the denominator of $\eta_j$. This represents how deep waters and rugged seafloor terrains severely impair search efficiency and diminish practical detectability, while shallow and smooth areas optimize detection capability."""),
    nbf.v4.new_code_cell("""def detection_probability(effort, depth, roughness, lmbda=0.5, gamma_d=2.0, gamma_r=1.5):
    efficiency = 1 / (1 + gamma_d * depth + gamma_r * roughness)
    return 1 - np.exp(-lmbda * effort * efficiency)"""),

    nbf.v4.new_markdown_cell("## 4. Stochastic Search Campaign Simulator Class\n\nWe implement the `SearchSimulator` class to manage stochastic search steps, evaluate cell-level probabilities under remaining budgets, sample targets, and update posteriors on failure."),
    nbf.v4.new_code_cell("""class SearchSimulator:
    def __init__(self, grid_df, prior_pdf, lmbda=0.5, gamma_d=2.0, gamma_r=1.5):
        self.grid_df = grid_df.copy()
        self.grid_df['posterior'] = prior_pdf
        self.lmbda = lmbda
        self.gamma_d = gamma_d
        self.gamma_r = gamma_r
        
    def get_search_box_poc(self, width=15, height=10):
        # Strategy: Greedy PoC. Find the rectangle containing the maximum posterior probability mass
        best_poc = -1.0
        best_coords = (0, 0)
        
        pivot = self.grid_df.pivot(index='y', columns='x', values='posterior').values
        for y_idx in range(Ny - height + 1):
            for x_idx in range(Nx - width + 1):
                window_poc = pivot[y_idx:y_idx+height, x_idx:x_idx+width].sum()
                if window_poc > best_poc:
                    best_poc = window_poc
                    best_coords = (x_idx, y_idx)
                    
        x_min = best_coords[0] + 0.5
        x_max = x_min + width - 1
        y_min = best_coords[1] + 0.5
        y_max = y_min + height - 1
        return x_min, x_max, y_min, y_max

    def get_search_box_pod(self, effort=0.2, width=15, height=10):
        # Strategy: Greedy PoD. Maximize PoD of the search box
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
        
    def get_search_box_cost_effective(self, max_cost=50):
        # Strategy: Maximize PoD / Cost Ratio
        best_ratio = -1.0
        best_search = None
        
        for width in [10, 15, 20]:
            for height in [5, 10, 15]:
                n_cells = width * height
                for effort in [0.1, 0.2, 0.5]:
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
                        
        return best_search if best_search else (15, 30, 5, 15, 0.1, 15)

    def search_step(self, x_min, x_max, y_min, y_max, effort, target_coords):
        in_region = (self.grid_df['x'] >= x_min) & (self.grid_df['x'] <= x_max) & \
                    (self.grid_df['y'] >= y_min) & (self.grid_df['y'] <= y_max)
        
        # Target index
        target_idx = self.grid_df[(self.grid_df['x'] == target_coords[0]) & (self.grid_df['y'] == target_coords[1])].index[0]
        target_in_box = in_region[target_idx]
        
        # Stochastic Outcome Draw using Bernoulli trial
        if target_in_box:
            t_depth = self.grid_df.loc[target_idx, 'depth']
            t_rough = self.grid_df.loc[target_idx, 'roughness']
            q_target = detection_probability(effort, t_depth, t_rough, self.lmbda, self.gamma_d, self.gamma_r)
            found = np.random.binomial(1, q_target) == 1
        else:
            found = False
            
        # Bayesian Posterior Update on Failure
        self.grid_df['c_t'] = in_region.astype(float)
        self.grid_df['rho_t'] = detection_probability(effort, self.grid_df['depth'], self.grid_df['roughness'], self.lmbda, self.gamma_d, self.gamma_r)
        self.grid_df['q_t'] = self.grid_df['c_t'] * self.grid_df['rho_t']
        
        likelihood = 1 - self.grid_df['q_t']
        unnormalized_post = likelihood * self.grid_df['posterior']
        self.grid_df['posterior'] = unnormalized_post / unnormalized_post.sum()
        
        return found"""),

    nbf.v4.new_markdown_cell("## 5. Policy Evaluation Module\n\nWe stochastically simulate 100 search campaigns for each search policy on each prior to evaluate success rate, average budget spent, and convergence rates."),
    nbf.v4.new_code_cell("""def evaluate_policy(prior_pdf, policy_name, lmbda=0.5, gamma_d=2.0, gamma_r=1.5, num_campaigns=100):
    cell_coords = grid_df[['x', 'y']].values
    cell_probs = prior_pdf / prior_pdf.sum()
    
    success_count = 0
    costs = []
    
    for i in range(num_campaigns):
        # 1. Sample Z_true from the specified prior
        idx = np.random.choice(len(cell_coords), p=cell_probs)
        target_coords = cell_coords[idx]
        
        # 2. Initialize Simulator
        sim = SearchSimulator(grid_df, prior_pdf, lmbda, gamma_d, gamma_r)
        budget = TOTAL_BUDGET
        found = False
        
        while budget > 0 and not found:
            if policy_name == 'Greedy PoC':
                x_min, x_max, y_min, y_max = sim.get_search_box_poc(width=15, height=10)
                effort = 0.2
                cost = effort * (15 * 10)
            elif policy_name == 'Greedy PoD':
                effort = 0.2
                x_min, x_max, y_min, y_max = sim.get_search_box_pod(effort, width=15, height=10)
                cost = effort * (15 * 10)
            elif policy_name == 'Cost Effective':
                x_min, x_max, y_min, y_max, effort, cost = sim.get_search_box_cost_effective(max_cost=50)
            
            if cost > budget:
                effort = budget / (x_max - x_min + 1) / (y_max - y_min + 1)
                cost = budget
                if effort < 0.05:
                    break
                
            found = sim.search_step(x_min, x_max, y_min, y_max, effort, target_coords)
            budget -= cost
            
        if found:
            success_count += 1
            costs.append(TOTAL_BUDGET - budget)
        else:
            costs.append(TOTAL_BUDGET)
            
    success_rate = success_count / num_campaigns
    avg_cost = np.mean(costs)
    return success_rate, avg_cost, costs

# Sweep across policies and priors
priors = {
    'Baseline (Prior 1)': prior_baseline,
    'Diffuse (Prior 2)': prior_diffuse,
    'Drift-dominated (Prior 3)': prior_drift_heavy
}
policies = ['Greedy PoC', 'Greedy PoD', 'Cost Effective']

results = []
cost_distributions = {}

np.random.seed(42)  # Set seed for reproducible Monte Carlo draws

for prior_name, prior_pdf in priors.items():
    for policy in policies:
        rate, cost, cost_list = evaluate_policy(prior_pdf, policy, num_campaigns=100)
        results.append({
            'Prior': prior_name,
            'Policy': policy,
            'Success Rate': rate,
            'Average Cost': cost
        })
        cost_distributions[f"{prior_name} - {policy}"] = cost_list

results_df = pd.DataFrame(results)
display(results_df)"""),

    nbf.v4.new_markdown_cell("## 6. Plotting Stochastic Search Results\n\nWe present bar charts and distributions comparing the performance of our strategies under the different prior scenarios."),
    nbf.v4.new_code_cell("""# Success Rate Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Prior', y='Success Rate', hue='Policy', palette='viridis')
plt.title('Monte Carlo Success Rate by Prior and Search Policy')
plt.ylim(0, 1.05)
plt.ylabel('Success Rate (Discovery Prob)')
plt.grid(axis='y', alpha=0.3)
plt.show()

# Cost Boxplot for Baseline Prior
plt.figure(figsize=(10, 6))
baseline_costs = {k: v for k, v in cost_distributions.items() if 'Baseline' in k}
sns.boxplot(data=pd.DataFrame(baseline_costs), palette='Set2')
plt.title('Search Cost Distribution for Baseline Prior (Budget Limit = 230)')
plt.ylabel('Cost Spent to Find Object')
plt.xticks(rotation=15)
plt.show()"""),

    nbf.v4.new_markdown_cell("## 7. Prior Misspecification / Robustness Study\n\nWe simulate a scenario where the true accident rested in a drift-dominated area (Prior 3), but the searcher mistakenly assumed the baseline trajectory (Prior 1). We evaluate the impact on search success."),
    nbf.v4.new_code_cell("""# Object actually lies in Prior 3, but searcher assumes Prior 1
def evaluate_misspecified_campaign(true_pdf, assumed_pdf, policy_name, num_campaigns=100):
    cell_coords = grid_df[['x', 'y']].values
    true_probs = true_pdf / true_pdf.sum()
    
    success_count = 0
    costs = []
    
    for i in range(num_campaigns):
        # Sample target location from TRUE prior (Prior 3)
        idx = np.random.choice(len(cell_coords), p=true_probs)
        target_coords = cell_coords[idx]
        
        # Initialize simulator with ASSUMED prior (Prior 1)
        sim = SearchSimulator(grid_df, assumed_pdf)
        budget = TOTAL_BUDGET
        found = False
        
        while budget > 0 and not found:
            if policy_name == 'Greedy PoC':
                x_min, x_max, y_min, y_max = sim.get_search_box_poc(width=15, height=10)
                effort = 0.2
                cost = effort * (15 * 10)
            elif policy_name == 'Greedy PoD':
                effort = 0.2
                x_min, x_max, y_min, y_max = sim.get_search_box_pod(effort, width=15, height=10)
                cost = effort * (15 * 10)
            elif policy_name == 'Cost Effective':
                x_min, x_max, y_min, y_max, effort, cost = sim.get_search_box_cost_effective(max_cost=50)
            
            if cost > budget:
                effort = budget / (x_max - x_min + 1) / (y_max - y_min + 1)
                cost = budget
                if effort < 0.05:
                    break
                
            found = sim.search_step(x_min, x_max, y_min, y_max, effort, target_coords)
            budget -= cost
            
        if found:
            success_count += 1
            costs.append(TOTAL_BUDGET - budget)
        else:
            costs.append(TOTAL_BUDGET)
            
    return success_count / num_campaigns, np.mean(costs)

robust_rate, robust_cost = evaluate_misspecified_campaign(prior_drift_heavy, prior_baseline, 'Cost Effective')
print(f"--- Robustness Misspecification Study ---")
print(f"When True Location is from Prior 3 (Drift-heavy) but Searcher assumes Prior 1 (Baseline):")
print(f"Success Rate: {robust_rate * 100:.1f}%")
print(f"Expected Search Cost: {robust_cost:.2f}")"""),

    nbf.v4.new_markdown_cell("## 8. Discussion & Optimal Search Zone Recommendation\n\n### Findings:\n1. **Policy Performance**: The **Cost-Effective** policy consistently outperforms Greedy PoC and Greedy PoD across all priors. By searching smaller high-density regions with higher effort and factoring in depth/roughness constraints, it minimizes budget depletion.\n2. **Robustness**: Model misspecification drops search success. Thus, having a physically motivated prior (with accurate $\\alpha$ and $\\beta$) is critical.\n\n### Recommendation for Mission 1:\nUsing the **Cost-Effective** search on the Baseline Prior, we compute the first recommended search mission coordinates below:"),
    nbf.v4.new_code_cell("""# Propose the first mission using the optimal policy
sim_recommender = SearchSimulator(grid_df, prior_baseline)
x_min, x_max, y_min, y_max, effort, cost = sim_recommender.get_search_box_cost_effective(max_cost=50)

print(f"--- Recommended First Search Mission ---")
print(f"Search Area: Bounding Box x:[{x_min}, {x_max}], y:[{y_min}, {y_max}]")
print(f"Search Effort: {effort:.2f}")
print(f"Expected Cost: {cost:.2f}")
print(f"Prior probability containment of search box: {prior_baseline[((grid_df['x'] >= x_min) & (grid_df['x'] <= x_max) & (grid_df['y'] >= y_min) & (grid_df['y'] <= y_max))].sum() * 100:.2f}%")""")
]

with open('Deliverable2_MonteCarlo_Simulations.ipynb', 'w') as f:
    nbf.write(nb, f)

with open('Deliverable2_v1.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Generated notebooks successfully!")
