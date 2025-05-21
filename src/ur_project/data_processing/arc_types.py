from typing import List, NewType, Tuple, Dict, Optional, Any
from dataclasses import dataclass

# Represents the value of a single cell in an ARC grid (typically 0-9)
ARCPixel = NewType('ARCPixel', int)

# Represents an ARC grid as a list of lists of ARCPixels
ARCGrid = List[List[ARCPixel]]

@dataclass
class ARCPair:
    """Represents a single input-output pair in an ARC task."""
    input_grid: ARCGrid
    output_grid: ARCGrid
    pair_id: Optional[str] = None # e.g., "train_0", "test_1"

@dataclass
class ARCTask:
    """Represents a full ARC task, including its ID, training pairs, and test pairs."""
    task_id: str
    training_pairs: List[ARCPair]
    test_pairs: List[ARCPair]
    # Optionally, store the original file path or other metadata
    source_file: Optional[str] = None

@dataclass
class ARCPuzzle:
    """
    Represents a single ARC puzzle instance derived from an ARCTask and ARCPair,
    formatted to be compatible with the Task protocol for the AZLoop.
    """
    id: str  # Unique ID for this specific puzzle instance (e.g., "task_id_train_0")
    description: str # Textual description of the puzzle
    data: ARCGrid  # The input grid for the solver
    expected_output_grid: ARCGrid # The target output grid for the verifier
    expected_solution_type: str = "arc_grid" # Type hint for the expected solution
    text_description: Optional[str] = None # Optional textual description of the task from the source
    source_task_id: str # Original ARCTask ID
    source_pair_id: str # Original ARCPair ID
    metadata: Optional[Dict[str, Any]] = None # For any other relevant info

# Example usage (for illustration, not part of the file content normally):
# if __name__ == '__main__':
#     example_grid: ARCGrid = [[1, 2], [3, 4]]
#     example_pair = ARCPair(input_grid=example_grid, output_grid=example_grid)
#     example_task = ARCTask(task_id="test_task", training_pairs=[example_pair], test_pairs=[example_pair])
#     print(example_task) 