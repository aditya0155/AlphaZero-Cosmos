import json
from typing import Optional, Dict, Any, List
import os

from ur_project.data_processing.arc_types import ARCTask, ARCPair, ARCGrid, ARCPixel

def _parse_grid(raw_grid: List[List[int]]) -> ARCGrid:
    """Converts a raw grid (list of lists of ints) to ARCGrid type."""
    return [[ARCPixel(cell) for cell in row] for row in raw_grid]

def load_arc_task_from_file(file_path: str) -> Optional[ARCTask]:
    """
    Loads a single ARC task from a JSON file.

    Args:
        file_path (str): The path to the ARC task JSON file.

    Returns:
        Optional[ARCTask]: An ARCTask object if loading is successful, otherwise None.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file - {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
        return None

    try:
        task_id = os.path.splitext(os.path.basename(file_path))[0]
        
        training_pairs = []
        for i, pair_data in enumerate(data.get("train", [])):
            input_grid = _parse_grid(pair_data["input"])
            output_grid = _parse_grid(pair_data["output"])
            training_pairs.append(ARCPair(input_grid=input_grid, output_grid=output_grid, pair_id=f"train_{i}"))

        test_pairs = []
        for i, pair_data in enumerate(data.get("test", [])):
            input_grid = _parse_grid(pair_data["input"])
            output_grid_raw = pair_data.get("output")
            if output_grid_raw is not None:
                output_grid = _parse_grid(output_grid_raw)
            else:
                raise ValueError(f"Test pair {i} in task {task_id} is missing an output grid in the JSON.")
            test_pairs.append(ARCPair(input_grid=input_grid, output_grid=output_grid, pair_id=f"test_{i}"))
            
        return ARCTask(
            task_id=task_id,
            training_pairs=training_pairs,
            test_pairs=test_pairs,
            source_file=file_path
        )

    except KeyError as e:
        print(f"Error: Missing expected key {e} in JSON data for task {file_path}")
        return None
    except ValueError as e: # Catching our own raised ValueError for missing test output
        print(f"Error processing task {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while parsing task data from {file_path}: {e}")
        return None

def load_arc_tasks_from_directory(directory_path: str) -> List[ARCTask]:
    """
    Loads all ARC tasks from JSON files in a specified directory.

    Args:
        directory_path (str): The path to the directory containing ARC task JSON files.

    Returns:
        List[ARCTask]: A list of ARCTask objects.
    """
    all_tasks = []
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found - {directory_path}")
        return []
        
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            task = load_arc_task_from_file(file_path)
            if task:
                all_tasks.append(task)
    return all_tasks

# Example Usage (for illustration):
# if __name__ == '__main__':
#     # Create dummy JSON files for testing
#     dummy_task_data_valid = {
#         "train": [
#             {"input": [[1]], "output": [[2]]}
#         ],
#         "test": [
#             {"input": [[3]], "output": [[4]]}
#         ]
#     }
#     dummy_task_data_missing_key = {
#         "train": [
#             {"input": [[1]]} # Missing "output"
#         ],
#         "test": [
#             {"input": [[3]], "output": [[4]]}
#         ]
#     }
#     dummy_task_data_missing_test_output = {
#         "train": [
#             {"input": [[1]], "output": [[2]]}
#         ],
#         "test": [
#             {"input": [[3]]} # Missing "output" for test
#         ]
#     }

#     os.makedirs("temp_arc_loader_test/data/arc/training", exist_ok=True)
#     with open("temp_arc_loader_test/data/arc/training/task1.json", "w") as f:
#         json.dump(dummy_task_data_valid, f)
#     with open("temp_arc_loader_test/data/arc/training/task2_invalid.json", "w") as f:
#         json.dump(dummy_task_data_missing_key, f)
#     with open("temp_arc_loader_test/data/arc/training/task3_missing_test.json", "w") as f:
#         json.dump(dummy_task_data_missing_test_output, f)
    
#     print("--- Testing load_arc_task_from_file ---")
#     task1 = load_arc_task_from_file("temp_arc_loader_test/data/arc/training/task1.json")
#     if task1:
#         print(f"Successfully loaded task: {task1.task_id}")
#         print(f"  Training pairs: {len(task1.training_pairs)}")
#         print(f"  Test pairs: {len(task1.test_pairs)}")
#         if task1.training_pairs:
#             print(f"    Train Pair 0 Input: {task1.training_pairs[0].input_grid}")
#             print(f"    Train Pair 0 Output: {task1.training_pairs[0].output_grid}")
#     else:
#         print("Failed to load task1.json")

#     print("\n--- Testing load_arc_task_from_file (invalid key) ---")
#     task2 = load_arc_task_from_file("temp_arc_loader_test/data/arc/training/task2_invalid.json")
#     if task2:
#         print(f"Loaded task: {task2.task_id}") # Should not happen
#     else:
#         print("Correctly failed to load task2_invalid.json")

#     print("\n--- Testing load_arc_task_from_file (missing test output) ---")
#     task3 = load_arc_task_from_file("temp_arc_loader_test/data/arc/training/task3_missing_test.json")
#     if task3:
#         print(f"Loaded task: {task3.task_id}") # Should not happen
#     else:
#         print("Correctly failed to load task3_missing_test.json due to missing test output")

#     print("\n--- Testing load_arc_tasks_from_directory ---")
#     # Assuming you have the ARC dataset downloaded and extracted in `../data/arc/training` relative to where this runs
#     # For a self-contained test, let's use our temp directory
#     loaded_tasks = load_arc_tasks_from_directory("temp_arc_loader_test/data/arc/training")
#     print(f"Loaded {len(loaded_tasks)} tasks from directory.")
#     for task in loaded_tasks:
#         print(f"  Task ID: {task.task_id}, Training Pairs: {len(task.training_pairs)}, Test Pairs: {len(task.test_pairs)}")

#     # Clean up dummy files
#     import shutil
#     shutil.rmtree("temp_arc_loader_test", ignore_errors=True) 