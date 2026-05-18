import nbformat

nb_path = "Deliverable2_Solution.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

old_code = """def plot_heatmap(df, column, title):
    pivot = df.pivot(index='y', columns='x', values=column)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="viridis", robust=True)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

plot_heatmap(grid_df, 'prior', 'Prior Distribution over the Grid')"""

new_code = """def plot_heatmap(df, column, title, show_vectors=False):
    pivot = df.pivot(index='y', columns='x', values=column)
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(pivot, cmap="viridis", robust=True)
    plt.title(title)
    if show_vectors:
        ax.scatter(x_E[0], x_E[1], color='red', marker='*', s=150, label='Accident (x_E)')
        ax.quiver(x_E[0], x_E[1], v_wind[0], v_wind[1], angles='xy', scale_units='xy', scale=1, color='cyan', label='Wind')
        ax.quiver(x_E[0], x_E[1], v_drift[0], v_drift[1], angles='xy', scale_units='xy', scale=1, color='orange', label='Drift')
        ax.quiver(x_E[0], x_E[1], v_plane[0], v_plane[1], angles='xy', scale_units='xy', scale=1, color='magenta', label='Plane')
        ax.legend()
    plt.gca().invert_yaxis()
    plt.show()

plot_heatmap(grid_df, 'prior', 'Prior Distribution over the Grid', show_vectors=True)"""

for cell in nb.cells:
    if cell.cell_type == "code" and old_code in cell.source:
        cell.source = cell.source.replace(old_code, new_code)

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
