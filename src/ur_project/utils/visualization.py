import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

from ur_project.data_processing.arc_types import ARCGrid

# Define a default color map for ARC pixels (0-9)
# Colors are chosen to be somewhat distinct and visually clear.
# Black (0), Blue (1), Red (2), Green (3), Yellow (4),
# Grey (5), Magenta (6), Orange (7), Cyan (8), Brown (9)
# User can customize this map if needed.
DEFAULT_ARC_COLOR_MAP = {
    0: (0, 0, 0),        # Black
    1: (0, 0, 1),        # Blue
    2: (1, 0, 0),        # Red
    3: (0, 1, 0),        # Green
    4: (1, 1, 0),        # Yellow
    5: (0.5, 0.5, 0.5),  # Grey
    6: (1, 0, 1),        # Magenta
    7: (1, 0.647, 0),    # Orange (e.g., darkorange)
    8: (0, 1, 1),        # Cyan
    9: (0.647, 0.165, 0.165) # Brown
    # Add more if ARC tasks use numbers > 9, though standard is 0-9
}

def visualize_arc_grid(
    grid: ARCGrid,
    color_map: Optional[dict] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_grid_lines: bool = True,
    show_ticks: bool = False
):
    """
    Visualizes a single ARC grid using matplotlib.

    Args:
        grid (ARCGrid): The ARC grid to visualize (list of lists of ARCPixel).
        color_map (Optional[dict]): A dictionary mapping pixel values (int) to RGB tuples (0-1 range).
                                     If None, uses DEFAULT_ARC_COLOR_MAP.
        title (Optional[str]): Title for the plot.
        ax (Optional[plt.Axes]): A matplotlib Axes object to plot on. If None, a new figure and axes are created.
        show_grid_lines (bool): Whether to display grid lines between cells.
        show_ticks (bool): Whether to display axis ticks and labels.
    """
    if not grid: # Handle empty grid
        if ax is None:
            fig, ax = plt.subplots(figsize=(2,2))
            created_fig = True
        else:
            created_fig = False
        ax.text(0.5, 0.5, "Empty Grid", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)
        if created_fig:
            plt.show()
        return

    cmap_to_use = color_map if color_map is not None else DEFAULT_ARC_COLOR_MAP
    
    # Convert ARCGrid to a NumPy array of RGB values for imshow
    # Determine grid dimensions
    rows = len(grid)
    cols = len(grid[0])
    
    # Create an empty RGB image array
    image_rgb = np.zeros((rows, cols, 3), dtype=float) # Use float for RGB values in [0,1]
    
    for r in range(rows):
        for c in range(cols):
            pixel_value = grid[r][c]
            image_rgb[r, c, :] = cmap_to_use.get(pixel_value, (1, 1, 1)) # Default to white if color not in map

    if ax is None:
        fig, ax = plt.subplots(figsize=(cols/2 if cols > 0 else 1, rows/2 if rows > 0 else 1))
        created_fig = True
    else:
        created_fig = False

    ax.imshow(image_rgb)

    if title:
        ax.set_title(title)

    # Configure grid lines and ticks
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    
    if show_grid_lines:
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    else:
        ax.grid(False)

    if show_ticks:
        ax.set_xticks(np.arange(0, cols, 1))
        ax.set_yticks(np.arange(0, rows, 1))
        ax.tick_params(which="minor", bottom=False, left=False)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis='both', which='both', length=0)

    if created_fig:
        plt.show()


def visualize_arc_task_pairs(
    arc_task, # ARCTask object from arc_loader
    max_pairs_to_show: int = 3,
    color_map: Optional[dict] = None,
    main_title: Optional[str] = None
):
    """
    Visualizes the training and test pairs of an ARC task.

    Args:
        arc_task (ARCTask): The ARCTask object.
        max_pairs_to_show (int): Maximum number of training/test pairs to display per category.
        color_map (Optional[dict]): Custom color map for grids.
        main_title (Optional[str]): Overall title for the visualization.
    """
    
    num_train = len(arc_task.training_pairs)
    num_test = len(arc_task.test_pairs)

    if main_title is None:
        main_title = f"ARC Task: {arc_task.task_id}"
    
    # Determine number of rows needed for plotting
    # Each pair (input/output) takes one row in the subplot grid
    rows_for_train = min(num_train, max_pairs_to_show)
    rows_for_test = min(num_test, max_pairs_to_show)
    total_plot_rows = rows_for_train + rows_for_test
    
    if total_plot_rows == 0:
        print(f"Task {arc_task.task_id} has no pairs to visualize.")
        return

    fig, axes = plt.subplots(total_plot_rows, 2, figsize=(8, total_plot_rows * 3))
    fig.suptitle(main_title, fontsize=16)
    
    current_plot_row = 0
    
    # Training Pairs
    if num_train > 0:
        ax_flat = axes.flatten() if total_plot_rows > 1 else axes # Handle single row case
        for i in range(rows_for_train):
            pair = arc_task.training_pairs[i]
            input_ax_idx = current_plot_row * 2
            output_ax_idx = current_plot_row * 2 + 1

            visualize_arc_grid(pair.input_grid, color_map=color_map, title=f"Train {i}: Input", ax=ax_flat[input_ax_idx])
            visualize_arc_grid(pair.output_grid, color_map=color_map, title=f"Train {i}: Output", ax=ax_flat[output_ax_idx])
            current_plot_row += 1
            
    # Test Pairs
    if num_test > 0:
        ax_flat = axes.flatten() if total_plot_rows > 1 else axes # Handle single row case
        for i in range(rows_for_test):
            pair = arc_task.test_pairs[i]
            input_ax_idx = current_plot_row * 2
            output_ax_idx = current_plot_row * 2 + 1
            
            title_prefix = f"Test {i}"
            # For test pairs, output might be ground truth or a prediction if we extend this later
            visualize_arc_grid(pair.input_grid, color_map=color_map, title=f"{title_prefix}: Input", ax=ax_flat[input_ax_idx])
            visualize_arc_grid(pair.output_grid, color_map=color_map, title=f"{title_prefix}: Output (GT)", ax=ax_flat[output_ax_idx])
            current_plot_row += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

# Example Usage (for illustration):
# if __name__ == '__main__':
#     from ur_project.data_processing.arc_types import ARCPair, ARCTask # For dummy data
#     # Create a dummy ARC task for visualization testing
#     grid1_in = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
#     grid1_out = [[2, 0, 2], [0, 2, 0], [2, 0, 2]]
#     grid2_in = [[3, 3], [3, 3]]
#     grid2_out = [[4, 4], [4, 4]]
#     grid3_in = [[5]]
#     grid3_out = [[0]]

#     dummy_task = ARCTask(
#         task_id="dummy_viz_task",
#         training_pairs=[
#             ARCPair(input_grid=grid1_in, output_grid=grid1_out, pair_id="train_0"),
#             ARCPair(input_grid=grid2_in, output_grid=grid2_out, pair_id="train_1")
#         ],
#         test_pairs=[
#             ARCPair(input_grid=grid3_in, output_grid=grid3_out, pair_id="test_0")
#         ]
#     )

#     print("--- Visualizing single grid ---")
#     visualize_arc_grid(grid1_in, title="Sample Input Grid")
#     visualize_arc_grid([], title="Empty Grid Test") # Test empty grid

#     print("\n--- Visualizing task pairs ---")
#     visualize_arc_task_pairs(dummy_task)

#     # Example with a task having more pairs than max_pairs_to_show
#     grid_many_in = [[1,2],[3,4]]
#     grid_many_out = [[0,0],[0,0]]
#     many_pairs_task = ARCTask(
#         task_id="many_pairs_task",
#         training_pairs=[
#             ARCPair(grid_many_in, grid_many_out, "train_0"),
#             ARCPair(grid_many_in, grid_many_out, "train_1"),
#             ARCPair(grid_many_in, grid_many_out, "train_2"),
#             ARCPair(grid_many_in, grid_many_out, "train_3"),
#         ],
#         test_pairs=[
#             ARCPair(grid_many_in, grid_many_out, "test_0"),
#             ARCPair(grid_many_in, grid_many_out, "test_1"),
#         ]
#     )
#     visualize_arc_task_pairs(many_pairs_task, max_pairs_to_show=2, main_title="Limited Pairs Demo") 