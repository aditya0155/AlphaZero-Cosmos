from typing import List, Tuple, Any, Dict
from copy import deepcopy
from src.ur_project.data_processing.arc_types import ARCGrid # Assuming ARCGrid is List[List[int]] or similar
from .arc_dsl import (
    DSLProgram, DSLOperation, ChangeColorOp, MoveOp, CopyObjectOp,
    CreateObjectOp, FillRectangleOp, DeleteObjectOp,
    DSLObjectSelector, DSLPosition, DSLColor
)

class ARCDSLInterpreter:
    def __init__(self):
        pass

    def execute_program(self, program: DSLProgram, initial_grid: ARCGrid) -> ARCGrid:
        current_grid = deepcopy(initial_grid)
        if not program or not program.operations:
            return current_grid

        for op in program.operations:
            current_grid = self._execute_operation(op, current_grid)
        return current_grid

    def _execute_operation(self, operation: DSLOperation, grid: ARCGrid) -> ARCGrid:
        if isinstance(operation, FillRectangleOp):
            return self._execute_fill_rectangle(operation, grid)
        elif isinstance(operation, ChangeColorOp):
            return self._execute_change_color(operation, grid)
        elif isinstance(operation, MoveOp):
            return self._execute_move_op(operation, grid)
        elif isinstance(operation, CopyObjectOp):
            return self._execute_copy_object_op(operation, grid)
        elif isinstance(operation, CreateObjectOp):
            return self._execute_create_object_op(operation, grid)
        elif isinstance(operation, DeleteObjectOp):
            return self._execute_delete_object_op(operation, grid)
        else:
            print(f"Warning: Operation type {type(operation)} not recognized or not yet implemented.")
            return grid

    def _get_grid_dimensions(self, grid: ARCGrid) -> Tuple[int, int]:
        if not grid or not isinstance(grid, list) or not grid[0] or not isinstance(grid[0], list):
            # Return 0,0 or raise an error if the grid is malformed
            print("Warning: Grid is empty or malformed.")
            return 0, 0
        rows = len(grid)
        cols = len(grid[0])
        return rows, cols

    def _execute_fill_rectangle(self, op: FillRectangleOp, grid: ARCGrid) -> ARCGrid:
        rows, cols = self._get_grid_dimensions(grid)
        if rows == 0 or cols == 0:
            return grid

        r1, c1 = op.top_left.coordinates
        r2, c2 = op.bottom_right.coordinates
        color_to_fill = op.color.value

        start_row = max(0, min(r1, r2))
        end_row = min(rows - 1, max(r1, r2))
        start_col = max(0, min(c1, c2))
        end_col = min(cols - 1, max(c1, c2))

        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                grid[r][c] = color_to_fill
        return grid

    def _execute_change_color(self, op: ChangeColorOp, grid: ARCGrid) -> ARCGrid:
        rows, cols = self._get_grid_dimensions(grid)
        if rows == 0 or cols == 0:
            return grid

        new_color_val = op.new_color.value
        criteria = op.selector.criteria

        if 'position' in criteria and isinstance(criteria['position'], tuple) and len(criteria['position']) == 2:
            r, c = criteria['position']
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = new_color_val
        elif 'old_color' in criteria:
            old_color_val = criteria['old_color']
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == old_color_val:
                        grid[r][c] = new_color_val
        else:
            print(f"Warning: ChangeColorOp selector criteria not understood: {criteria}")
            
        return grid

    def _execute_move_op(self, op: MoveOp, grid: ARCGrid) -> ARCGrid:
        print(f"Warning: MoveOp with selector '{op.selector.criteria}' to position '{op.new_position.coordinates}' not yet implemented.")
        return grid

    def _execute_copy_object_op(self, op: CopyObjectOp, grid: ARCGrid) -> ARCGrid:
        print(f"Warning: CopyObjectOp with selector '{op.selector.criteria}' to position '{op.target_position.coordinates}' not yet implemented.")
        return grid

    def _execute_create_object_op(self, op: CreateObjectOp, grid: ARCGrid) -> ARCGrid:
        print(f"Warning: CreateObjectOp with description '{op.object_description}' at position '{op.position.coordinates}' not yet implemented.")
        return grid

    def _execute_delete_object_op(self, op: DeleteObjectOp, grid: ARCGrid) -> ARCGrid:
        print(f"Warning: DeleteObjectOp with selector '{op.selector.criteria}' not yet implemented.")
        return grid

if __name__ == '__main__':
    # Example Usage (Basic)
    print("ARCDSLInterpreter Example Usage")

    # 1. Define a sample program
    program = DSLProgram(operations=[
        FillRectangleOp(
            top_left=DSLPosition(coordinates=(1, 1)),
            bottom_right=DSLPosition(coordinates=(2, 2)),
            color=DSLColor(value=5)
        ),
        ChangeColorOp(
            selector=DSLObjectSelector(criteria={'position': (0,0)}),
            new_color=DSLColor(value=3)
        ),
        ChangeColorOp(
            selector=DSLObjectSelector(criteria={'old_color': 0}), # Change all 0s to 7
            new_color=DSLColor(value=7)
        ),
        MoveOp( # Unimplemented
            selector=DSLObjectSelector(criteria={'comment': 'object1'}),
            new_position=DSLPosition(coordinates=(5,5))
        )
    ])

    # 2. Define an initial grid
    initial_grid_data: ARCGrid = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    print("\nInitial Grid:")
    for row in initial_grid_data:
        print(row)

    # 3. Create interpreter and execute
    interpreter = ARCDSLInterpreter()
    final_grid = interpreter.execute_program(program, initial_grid_data)

    print("\nFinal Grid:")
    for row in final_grid:
        print(row)

    print("\n--- Test Empty Program ---")
    empty_program = DSLProgram(operations=[])
    grid_after_empty_program = interpreter.execute_program(empty_program, initial_grid_data)
    print("Grid after empty program (should be same as initial copy):")
    for row in grid_after_empty_program:
        print(row)
    
    print("\n--- Test Program on Empty Grid ---")
    empty_grid: ARCGrid = []
    final_grid_from_empty = interpreter.execute_program(program, empty_grid)
    print("Final grid from empty initial grid:")
    for row in final_grid_from_empty:
        print(row)

    print("\n--- Test FillRectangle out of bounds ---")
    program_fill_oob = DSLProgram(operations=[
        FillRectangleOp(
            top_left=DSLPosition(coordinates=(5, 5)), # out of 4x4 bounds
            bottom_right=DSLPosition(coordinates=(6, 6)),
            color=DSLColor(value=8)
        )
    ])
    initial_grid_for_oob: ARCGrid = [[1,1],[1,1]]
    print("Initial for OOB fill:")
    for r in initial_grid_for_oob: print(r)
    final_grid_oob_fill = interpreter.execute_program(program_fill_oob, initial_grid_for_oob)
    print("Final grid after OOB fill (should be unchanged or gracefully handled):")
    for r in final_grid_oob_fill: print(r)
    # Expected: The fill operation should be clipped to the grid boundaries or do nothing if entirely outside.
    # Current _execute_fill_rectangle handles this by clipping.

    print("\n--- Test ChangeColorOp with specific non-existent position ---")
    program_change_pos_oob = DSLProgram(operations=[
        ChangeColorOp(
            selector=DSLObjectSelector(criteria={'position': (10,10)}), # out of bounds
            new_color=DSLColor(value=9)
        )
    ])
    initial_grid_for_oob_change: ARCGrid = [[2,2],[2,2]]
    print("Initial for OOB change:")
    for r in initial_grid_for_oob_change: print(r)
    final_grid_oob_change = interpreter.execute_program(program_change_pos_oob, initial_grid_for_oob_change)
    print("Final grid after OOB position change (should be unchanged):")
    for r in final_grid_oob_change: print(r)
    # Expected: No change if position is out of bounds.
    # Current _execute_change_color handles this.

    print("\n--- Test ChangeColorOp with non-matching old_color ---")
    program_change_old_color_nomatch = DSLProgram(operations=[
        ChangeColorOp(
            selector=DSLObjectSelector(criteria={'old_color': 5}), # '5' does not exist
            new_color=DSLColor(value=9)
        )
    ])
    initial_grid_old_color_nomatch: ARCGrid = [[2,2],[2,2]]
    print("Initial for old_color no match change:")
    for r in initial_grid_old_color_nomatch: print(r)
    final_grid_old_color_nomatch = interpreter.execute_program(program_change_old_color_nomatch, initial_grid_old_color_nomatch)
    print("Final grid after old_color no match change (should be unchanged):")
    for r in final_grid_old_color_nomatch: print(r)
    # Expected: No change if old_color is not found.
    # Current _execute_change_color handles this.

    print("\n--- Test ChangeColorOp with unknown criteria ---")
    program_change_unknown_criteria = DSLProgram(operations=[
        ChangeColorOp(
            selector=DSLObjectSelector(criteria={'unknown_selector': 'value'}),
            new_color=DSLColor(value=9)
        )
    ])
    initial_grid_unknown_criteria: ARCGrid = [[2,2],[2,2]]
    print("Initial for unknown criteria change:")
    for r in initial_grid_unknown_criteria: print(r)
    final_grid_unknown_criteria = interpreter.execute_program(program_change_unknown_criteria, initial_grid_unknown_criteria)
    print("Final grid after unknown criteria change (should be unchanged, warning printed):")
    for r in final_grid_unknown_criteria: print(r)
    # Expected: No change, warning printed.
    # Current _execute_change_color handles this. 