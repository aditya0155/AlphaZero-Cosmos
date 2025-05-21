# src/ur_project/core/arc_proposer.py
import os
import random
from typing import List, Optional, Iterator

from ur_project.core.proposer import BaseProposer
from ur_project.data_processing.arc_types import ARCTask, ARCPuzzle, ARCGrid
from ur_project.data_processing.arc_loader import load_arc_tasks_from_directory, load_arc_task_from_file

class ARCProposer(BaseProposer):
    """
    Proposes ARC puzzles from a given set of ARC tasks.
    It can iterate through all training pairs of all tasks, or sample randomly.
    """
    def __init__(
        self,
        arc_tasks_directory: str,
        task_ids: Optional[List[str]] = None, # Specific task IDs to load, e.g., ["007bbfb7", "1cf80156"]
        propose_mode: str = "iterate_train", # "iterate_train", "random_train", "iterate_test", "random_test"
        shuffle_tasks: bool = True,
        shuffle_pairs: bool = True,
        proposer_id: str = "ARCProposer_v1"
    ):
        self.arc_tasks_directory = arc_tasks_directory
        self.propose_mode = propose_mode
        self.shuffle_tasks = shuffle_tasks
        self.shuffle_pairs = shuffle_pairs
        self.proposer_id = proposer_id
        
        self.tasks: List[ARCTask] = []
        self._puzzle_iterator: Optional[Iterator[ARCPuzzle]] = None

        self._load_tasks(task_ids)
        self._initialize_iterator()

    def _load_tasks(self, task_ids: Optional[List[str]] = None):
        """Loads ARC tasks from the specified directory or specific IDs."""
        if task_ids:
            for task_id in task_ids:
                # Assume task_id might or might not include .json extension
                task_file_name = task_id if task_id.endswith(".json") else f"{task_id}.json"
                # We need to know if we are looking in 'training' or 'evaluation'
                # For simplicity, let's assume task_ids are from training if not specified.
                # A more robust solution might require explicit paths or searching both.
                # For now, we'll assume task_ids are found in the base arc_tasks_directory
                # which could be data/arc/training or data/arc/evaluation
                task_file_path = os.path.join(self.arc_tasks_directory, task_file_name)
                task = load_arc_task_from_file(task_file_path)
                if task:
                    self.tasks.append(task)
                else:
                    print(f"Warning: Could not load specified task: {task_file_path}")
        else:
            self.tasks = load_arc_tasks_from_directory(self.arc_tasks_directory)

        if not self.tasks:
            raise ValueError(f"No ARC tasks found or loaded from {self.arc_tasks_directory} with specified IDs.")

        if self.shuffle_tasks:
            random.shuffle(self.tasks)
        print(f"ARCProposer: Loaded {len(self.tasks)} tasks.")

    def _generate_all_puzzles(self) -> Iterator[ARCPuzzle]:
        """Generates all possible ARCPuzzles based on the propose_mode."""
        tasks_to_iterate = list(self.tasks) # Copy for potential shuffling
        if self.shuffle_tasks:
            random.shuffle(tasks_to_iterate)

        for task in tasks_to_iterate:
            pairs_to_use = []
            pair_type_prefix = ""

            if "train" in self.propose_mode:
                pairs_to_use = list(task.training_pairs) # Copy
                pair_type_prefix = "train"
            elif "test" in self.propose_mode: # For now, we assume test pairs have outputs for this proposer
                pairs_to_use = list(task.test_pairs) # Copy
                pair_type_prefix = "test"
            
            if self.shuffle_pairs:
                random.shuffle(pairs_to_use)

            for i, pair in enumerate(pairs_to_use):
                puzzle_id = f"{task.task_id}_{pair.pair_id if pair.pair_id else f'{pair_type_prefix}_{i}'}"
                description = (
                    f"Solve the ARC puzzle. Task ID: {task.task_id}, Pair: {pair.pair_id or f'{pair_type_prefix}_{i}'}. "
                    f"Input grid has dimensions {len(pair.input_grid)}x{len(pair.input_grid[0]) if pair.input_grid else 0}. "
                    f"Transform the input grid to match the expected output grid."
                )
                # Ensure output_grid exists for the puzzle creation.
                # The arc_loader should guarantee this for training, and for test if they are complete.
                if pair.output_grid is None:
                    print(f"Warning: Task {task.task_id}, Pair {pair.pair_id or f'{pair_type_prefix}_{i}'} has no output grid. Skipping.")
                    continue

                yield ARCPuzzle(
                    id=puzzle_id,
                    description=description,
                    data=pair.input_grid, # data is the input_grid
                    expected_output_grid=pair.output_grid,
                    text_description=f"TASK_TYPE: ARC - TASK_ID: {task.task_id}", # Placeholder text_description
                    source_task_id=task.task_id,
                    source_pair_id=pair.pair_id if pair.pair_id else f'{pair_type_prefix}_{i}'
                )
    
    def _initialize_iterator(self):
        """Initializes or re-initializes the puzzle iterator."""
        self._puzzle_iterator = self._generate_all_puzzles()

    def propose_task(self) -> ARCPuzzle:
        """
        Proposes the next ARC puzzle.
        If iterate mode is used, it will cycle through tasks and their pairs.
        If random mode is used, it selects a task and a pair randomly (less efficient for full coverage).
        Currently, only iterate mode is fully implemented by the iterator.
        Random mode will effectively behave like iterate with shuffling.
        """
        if self._puzzle_iterator is None:
            self._initialize_iterator()
        
        try:
            return next(self._puzzle_iterator)
        except StopIteration:
            # Reached the end of all puzzles, re-initialize to loop
            print("ARCProposer: Reached end of puzzles, re-shuffling and starting over.")
            self._initialize_iterator()
            try:
                return next(self._puzzle_iterator)
            except StopIteration:
                # Should not happen if tasks exist, but as a safeguard
                raise RuntimeError("ARCProposer: No puzzles available even after reset. Check task loading.")

# Example Usage (for illustration, requires ARC data to be present)
# if __name__ == '__main__':
#     # Adjust this path to your actual ARC training data directory
#     # e.g., ../../data/arc/training (if running from src/ur_project/core)
#     dummy_arc_training_path = "path_to_your_arc_data/training" 
#     if not os.path.exists(dummy_arc_training_path) or not os.listdir(dummy_arc_training_path):
#         print(f"Warning: ARC training data not found at {dummy_arc_training_path}. Example will not run fully.")
#     else:
#         try:
#             proposer = ARCProposer(arc_tasks_directory=dummy_arc_training_path)
#             print("\n--- Proposing tasks (iterate_train mode) ---")
#             for i in range(5): # Propose a few tasks
#                 try:
#                     puzzle = proposer.propose_task()
#                     print(f"Proposed Puzzle {i+1}: ID={puzzle.id}, Task={puzzle.source_task_id}, Pair={puzzle.source_pair_id}")
#                     # print(f"  Input grid rows: {len(puzzle.data)}")
#                     # print(f"  Expected output grid rows: {len(puzzle.expected_output_grid)}")
#                 except RuntimeError as e:
#                     print(f"Error proposing task: {e}")
#                     break
#                 except Exception as e:
#                     print(f"An unexpected error during proposing: {e}")
#                     break
            
#             # Example with specific task IDs
#             # Replace with actual task IDs from your dataset (without .json)
#             # specific_ids = ["007bbfb7", "1cf80156"] 
#             # print(f"\n--- Proposing tasks from specific IDs: {specific_ids} ---")
#             # try:
#             #     proposer_specific = ARCProposer(arc_tasks_directory=dummy_arc_training_path, task_ids=specific_ids)
#             #     puzzle_specific = proposer_specific.propose_task()
#             #     print(f"Proposed Specific Puzzle: ID={puzzle_specific.id}, Task={puzzle_specific.source_task_id}")
#             # except ValueError as e:
#             #     print(f"Error initializing specific proposer (likely tasks not found): {e}")
#             # except RuntimeError as e:
#             #     print(f"Error proposing task from specific set: {e}")


#         except ValueError as e:
#             print(f"Error initializing ARCProposer: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred in example: {e}") 