# Notebook 1: ARC Data Exploration
# This script will be used to load, inspect, and visualize ARC tasks.

import os
from ur_project.data_processing.arc_loader import load_arc_task_from_file, load_arc_tasks_from_directory
from ur_project.utils.visualization import visualize_arc_grid, visualize_arc_task_pairs

def main():
    print("ARC Data Exploration Script")

    # Determine base path relative to this script's location
    # Assumes this notebook is in AlphaZero_Cosmos/notebooks/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # AlphaZero_Cosmos
    
    # Define paths to ARC data
    # Using a known small task for initial demonstration to avoid loading everything.
    # The ARC dataset contains tasks like '007bbfb7.json'
    # Make sure the download_arc.py script has been run and data is in data/arc/
    arc_training_dir = os.path.join(project_root, "data", "arc", "training")
    # Pick a specific task file for detailed view, if it exists
    # (You should replace this with an actual task ID from your downloaded dataset)
    example_task_id = "007bbfb7.json" # A commonly known small task
    example_task_file = os.path.join(arc_training_dir, example_task_id)

    print(f"\n--- Attempting to load a single ARC task: {example_task_file} ---")
    if os.path.exists(example_task_file):
        arc_task = load_arc_task_from_file(example_task_file)
        if arc_task:
            print(f"Successfully loaded Task ID: {arc_task.task_id}")
            print(f"  Source File: {arc_task.source_file}")
            print(f"  Number of training pairs: {len(arc_task.training_pairs)}")
            print(f"  Number of test pairs: {len(arc_task.test_pairs)}")
            
            # Visualize the first training pair's input and output grids directly
            if arc_task.training_pairs:
                print("\nVisualizing first training pair grids directly:")
                first_train_pair = arc_task.training_pairs[0]
                visualize_arc_grid(first_train_pair.input_grid, title=f"{arc_task.task_id} - Train 0 Input")
                visualize_arc_grid(first_train_pair.output_grid, title=f"{arc_task.task_id} - Train 0 Output")
            
            # Visualize all pairs of this task using the dedicated function
            print("\nVisualizing all pairs for the loaded task:")
            visualize_arc_task_pairs(arc_task, max_pairs_to_show=5) # Show up to 5 of each
        else:
            print(f"Failed to load task from {example_task_file}")
    else:
        print(f"Example task file not found: {example_task_file}")
        print("Please ensure the ARC dataset is downloaded and extracted to data/arc/")
        print("You can run `python scripts/download_arc.py` from the project root.")

    print(f"\n--- Attempting to load all tasks from directory: {arc_training_dir} (summary) ---")
    if os.path.isdir(arc_training_dir):
        all_training_tasks = load_arc_tasks_from_directory(arc_training_dir)
        if all_training_tasks:
            print(f"Successfully loaded {len(all_training_tasks)} tasks from {arc_training_dir}.")
            # Print summary of a few tasks
            for i, task in enumerate(all_training_tasks[:3]): # Print details for the first 3 tasks
                print(f"  Task {i+1}: ID {task.task_id}, Train Pairs: {len(task.training_pairs)}, Test Pairs: {len(task.test_pairs)}")
            if len(all_training_tasks) > 3:
                print(f"  ... and {len(all_training_tasks) - 3} more tasks.")
            
            # Example: Visualize the first task from the full list if not already shown
            if all_training_tasks[0].task_id != os.path.splitext(example_task_id)[0]:
                 print("\nVisualizing first task from the full list:")
                 visualize_arc_task_pairs(all_training_tasks[0], max_pairs_to_show=3)
        else:
            print(f"No tasks found or loaded from {arc_training_dir}. Ensure .json files are present.")
    else:
        print(f"Training directory not found: {arc_training_dir}")
        print("Please ensure the ARC dataset is downloaded and extracted to data/arc/training/")

if __name__ == "__main__":
    main() 