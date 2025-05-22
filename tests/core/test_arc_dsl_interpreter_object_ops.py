import unittest
from copy import deepcopy

from src.ur_project.core.arc_dsl_interpreter import ARCDSLInterpreter, DSLExecutionError
from src.ur_project.core.arc_dsl import (
    DSLProgram,
    MoveOp, CopyObjectOp, DeleteObjectOp, CreateObjectOp, # Operation classes
    DSLObjectSelector, DSLPosition, DSLColor # Supporting classes
)
from src.ur_project.data_processing.arc_types import ARCGrid

class TestARCDSLInterpreterObjectOps(unittest.TestCase):

    def setUp(self):
        self.interpreter = ARCDSLInterpreter()

    def _create_grid(self, rows: int, cols: int, default_value: int = 0) -> ARCGrid:
        return [[default_value for _ in range(cols)] for _ in range(rows)]

    def assertGridsEqual(self, grid1: ARCGrid, grid2: ARCGrid, msg: str = None):
        self.assertEqual(len(grid1), len(grid2), f"Grid row counts differ. {msg or ''}")
        for i, (row1, row2) in enumerate(zip(grid1, grid2)):
            self.assertEqual(len(row1), len(row2), f"Grid col counts differ at row {i}. {msg or ''}")
            self.assertEqual(row1, row2, f"Grid content differs at row {i}. {msg or ''}")

    # --- Test Scenarios for _execute_move_op ---

    def test_move_single_object_color_selection(self):
        initial_grid = self._create_grid(5, 5)
        initial_grid[1][1] = 1
        initial_grid[1][2] = 1
        initial_grid[2][1] = 1

        expected_grid = self._create_grid(5, 5)
        expected_grid[2][2] = 1 # Moved by (1,1)
        expected_grid[2][3] = 1
        expected_grid[3][2] = 1
        
        op = MoveOp(
            selector=DSLObjectSelector(criteria={'color': 1}),
            destination=DSLPosition(row=1, col=1) # Relative offset
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Move single object failed")

    def test_move_multiple_matching_objects(self):
        initial_grid = self._create_grid(5, 5)
        initial_grid[0][0] = 2 # Obj 1
        initial_grid[0][1] = 2
        initial_grid[3][3] = 2 # Obj 2
        initial_grid[3][4] = 2

        expected_grid = self._create_grid(5, 5)
        expected_grid[1][1] = 2 # Obj 1 moved by (1,1)
        expected_grid[1][2] = 2
        expected_grid[4][4] = 2 # Obj 2 moved by (1,1)
        expected_grid[4][5-1] = 2 # (3,4) -> (4,5) -> clipped to (4,4) if grid is 5x5. Oh, destination is offset.
        # Obj2 at (3,3), (3,4). Top-left (3,3). Moved by (1,1) -> new top-left (4,4)
        # (3,3) -> (4,4)
        # (3,4) -> (4,5) which is (4,4) if cols=5
        # The code calculates new_obj_top_left_r, new_obj_top_left_c then adds r_offset, c_offset
        # For Obj2: min_r=3, min_c=3. offset_r=1, offset_c=1. new_obj_top_left=(4,4)
        # Pixel (3,3,'color'): r_offset=0,c_offset=0. dest=(4,4)
        # Pixel (3,4,'color'): r_offset=0,c_offset=1. dest=(4,5) -> clipped to (4,4) if cols=5
        # This should be (4,4) and (4,4) if cols=5. Let's make grid 6x6 for clarity.
        
        initial_grid = self._create_grid(6, 6)
        initial_grid[0][0] = 2; initial_grid[0][1] = 2
        initial_grid[3][3] = 2; initial_grid[3][4] = 2

        expected_grid = self._create_grid(6, 6)
        expected_grid[1][1] = 2; expected_grid[1][2] = 2 # Obj1 moved
        expected_grid[4][4] = 2; expected_grid[4][5] = 2 # Obj2 moved
        
        op = MoveOp(
            selector=DSLObjectSelector(criteria={'color': 2}),
            destination=DSLPosition(row=1, col=1)
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Move multiple objects failed")

    def test_move_object_off_grid_clipping(self):
        initial_grid = self._create_grid(3, 3)
        initial_grid[0][0] = 3
        initial_grid[0][1] = 3

        expected_grid = self._create_grid(3, 3)
        expected_grid[0][2] = 3 # Moved by (0,2). (0,0)->(0,2), (0,1)->(0,3) clipped
        
        op = MoveOp(
            selector=DSLObjectSelector(criteria={'color': 3}),
            destination=DSLPosition(row=0, col=2)
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Move object off-grid (clipping) failed")

    def test_move_object_onto_another_overwrite(self):
        initial_grid = self._create_grid(5, 5)
        initial_grid[1][1] = 4 # Obj A
        initial_grid[3][1] = 5 # Obj B

        expected_grid = self._create_grid(5, 5)
        expected_grid[3][1] = 4 # Obj A moved by (2,0) onto Obj B
        
        op = MoveOp(
            selector=DSLObjectSelector(criteria={'color': 4}),
            destination=DSLPosition(row=2, col=0)
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Move object onto another (overwrite) failed")

    def test_move_non_existent_object(self):
        initial_grid = self._create_grid(3, 3)
        initial_grid[0][0] = 1
        
        expected_grid = deepcopy(initial_grid)
        op = MoveOp(
            selector=DSLObjectSelector(criteria={'color': 99}), # Color 99 does not exist
            destination=DSLPosition(row=1, col=1)
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Move non-existent object failed")

    # --- Test Scenarios for _execute_copy_object_op ---

    def test_copy_single_object(self):
        initial_grid = self._create_grid(5, 5)
        initial_grid[1][1] = 1
        initial_grid[1][2] = 1

        expected_grid = deepcopy(initial_grid)
        expected_grid[2][2] = 1 # Copied by (1,1)
        expected_grid[2][3] = 1
        
        op = CopyObjectOp(
            selector=DSLObjectSelector(criteria={'color': 1}),
            destination=DSLPosition(row=1, col=1)
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Copy single object failed")

    def test_copy_multiple_matching_objects(self):
        initial_grid = self._create_grid(6, 6)
        initial_grid[0][0] = 2; initial_grid[0][1] = 2 # Obj 1
        initial_grid[3][3] = 2; initial_grid[3][4] = 2 # Obj 2

        expected_grid = deepcopy(initial_grid)
        expected_grid[1][1] = 2; expected_grid[1][2] = 2 # Obj1 copied by (1,1)
        expected_grid[4][4] = 2; expected_grid[4][5] = 2 # Obj2 copied by (1,1)
        
        op = CopyObjectOp(
            selector=DSLObjectSelector(criteria={'color': 2}),
            destination=DSLPosition(row=1, col=1)
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Copy multiple objects failed")

    def test_copy_object_off_grid_clipping(self):
        initial_grid = self._create_grid(3, 3)
        initial_grid[0][0] = 3
        initial_grid[0][1] = 3

        expected_grid = deepcopy(initial_grid)
        expected_grid[0][2] = 3 # Copied by (0,2). (0,1) -> (0,3) is clipped
        
        op = CopyObjectOp(
            selector=DSLObjectSelector(criteria={'color': 3}),
            destination=DSLPosition(row=0, col=2)
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Copy object off-grid (clipping) failed")

    def test_copy_object_onto_another_overwrite(self):
        initial_grid = self._create_grid(5, 5)
        initial_grid[1][1] = 4 # Obj A
        initial_grid[3][1] = 5 # Obj B

        expected_grid = deepcopy(initial_grid)
        expected_grid[3][1] = 4 # Obj A copied by (2,0) onto Obj B
        
        op = CopyObjectOp(
            selector=DSLObjectSelector(criteria={'color': 4}),
            destination=DSLPosition(row=2, col=0)
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Copy object onto another (overwrite) failed")

    def test_copy_non_existent_object(self):
        initial_grid = self._create_grid(3, 3)
        initial_grid[0][0] = 1
        
        expected_grid = deepcopy(initial_grid)
        op = CopyObjectOp(
            selector=DSLObjectSelector(criteria={'color': 99}),
            destination=DSLPosition(row=1, col=1)
        )
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Copy non-existent object failed")

    # --- Test Scenarios for _execute_delete_object_op ---

    def test_delete_single_object(self):
        initial_grid = self._create_grid(3, 3)
        initial_grid[1][1] = 1
        initial_grid[1][2] = 1

        expected_grid = self._create_grid(3, 3) # Default 0 is background
        
        op = DeleteObjectOp(selector=DSLObjectSelector(criteria={'color': 1}))
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Delete single object failed")

    def test_delete_multiple_matching_objects(self):
        initial_grid = self._create_grid(5, 5)
        initial_grid[0][0] = 2; initial_grid[0][1] = 2
        initial_grid[3][3] = 2; initial_grid[3][4] = 2
        initial_grid[1][1] = 1 # Unrelated object

        expected_grid = self._create_grid(5, 5)
        expected_grid[1][1] = 1
        
        op = DeleteObjectOp(selector=DSLObjectSelector(criteria={'color': 2}))
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Delete multiple objects failed")

    def test_delete_non_existent_object(self):
        initial_grid = self._create_grid(3, 3)
        initial_grid[0][0] = 1
        
        expected_grid = deepcopy(initial_grid)
        op = DeleteObjectOp(selector=DSLObjectSelector(criteria={'color': 99}))
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Delete non-existent object failed")

    # --- Test Scenarios for _execute_create_object_op ---
    def test_create_simple_object(self):
        initial_grid = self._create_grid(5, 5)
        # Pixels are (row_offset, col_offset, color_value) relative to destination
        pixels_to_create = [
            (0, 0, 7), (0, 1, 7), (1, 0, 7) 
        ]
        destination = DSLPosition(row=1, col=1)

        expected_grid = self._create_grid(5, 5)
        expected_grid[1][1] = 7
        expected_grid[1][2] = 7
        expected_grid[2][1] = 7
        
        op = CreateObjectOp(pixels=pixels_to_create, destination=destination)
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Create simple object failed")

    def test_create_object_partially_off_grid(self):
        initial_grid = self._create_grid(3, 3)
        pixels_to_create = [(0,0,8),(0,1,8),(0,2,8)] # 1x3 line
        destination = DSLPosition(row=1, col=1)

        expected_grid = self._create_grid(3,3)
        expected_grid[1][1] = 8
        expected_grid[1][2] = 8
        # (1,3) is off-grid for col
        
        op = CreateObjectOp(pixels=pixels_to_create, destination=destination)
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Create object partially off-grid failed")

    def test_create_object_onto_existing_overwrite(self):
        initial_grid = self._create_grid(3, 3)
        initial_grid[1][1] = 9 # Existing pixel
        
        pixels_to_create = [(0,0,8)] # Single pixel
        destination = DSLPosition(row=1, col=1) # Overlap with existing

        expected_grid = self._create_grid(3,3)
        expected_grid[1][1] = 8 # Overwritten
        
        op = CreateObjectOp(pixels=pixels_to_create, destination=destination)
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Create object onto existing (overwrite) failed")

    def test_create_object_empty_pixels_list(self):
        initial_grid = self._create_grid(3, 3)
        initial_grid[0][0] = 1
        
        expected_grid = deepcopy(initial_grid)
        op = CreateObjectOp(pixels=[], destination=DSLPosition(row=1,col=1))
        program = DSLProgram(operations=[op])
        final_grid = self.interpreter.execute_program(program, initial_grid)
        self.assertGridsEqual(final_grid, expected_grid, "Create object with empty pixels list failed")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
