from typing import List, Tuple, Any, Dict
from copy import deepcopy
from src.ur_project.data_processing.arc_types import ARCGrid # Assuming ARCGrid is List[List[int]] or similar
from .arc_dsl import (
    DSLProgram, DSLOperation, ChangeColorOp, MoveOp, CopyObjectOp,
    CreateObjectOp, FillRectangleOp, DeleteObjectOp,
    DSLObjectSelector, DSLPosition, DSLColor,
    COMMAND_MAP  # Import COMMAND_MAP
)

# Custom Exceptions
class DSLParsingError(ValueError):
    """Custom exception for errors during DSL parsing."""
    pass

class DSLExecutionError(RuntimeError):
    """Custom exception for errors during DSL execution."""
    pass


class ARCDSLInterpreter:
    def __init__(self):
        pass

    def _parse_dsl_arguments(self, arg_string: str, command_name: str) -> Dict[str, Any]:
        """
        Parses the argument string for a DSL command.
        Example: "top_left=DSLPosition(1,1), bottom_right=DSLPosition(2,2), color=DSLColor(5)"
        This is a simplified parser and needs to be made more robust.
        """
        args = {}
        if not arg_string.strip():
            return args

        # Simple split by comma for now, but this won't handle nested structures well.
        # A more robust parser (e.g., using regex or a parsing library) would be needed for complex args.
        # For now, let's assume simple key=value pairs and specific parsing for known types.
        
        # Example parsing for FillRectangleOp: "top_left=DSLPosition(1,1), bottom_right=DSLPosition(2,2), color=DSLColor(5)"
        # Example parsing for ChangeColorOp: "selector=DSLObjectSelector(criteria={'old_color':0}), new_color=DSLColor(7)"

        current_arg_name = None
        current_arg_value_str = ""
        parenthesis_level = 0

        for char in arg_string + ",": # Add trailing comma to process last argument
            if char == '=' and parenthesis_level == 0 and current_arg_name is None:
                current_arg_name = current_arg_value_str.strip()
                current_arg_value_str = ""
            elif char == ',' and parenthesis_level == 0:
                if current_arg_name is None: # Handle cases like "DSLPosition(1,1)" directly if not key-value
                    # This part needs more thought if we have non-key-value args.
                    # For now, assuming all args are key=value.
                    raise DSLParsingError(f"Argument parsing error near '{current_arg_value_str}' for command {command_name}. Expected key=value.")

                arg_value_str = current_arg_value_str.strip()
                
                # Basic type parsing (needs to be command-specific and more robust)
                if arg_value_str.startswith("DSLPosition("):
                    try:
                        coords_str = arg_value_str[len("DSLPosition("):-1]
                        r_str, c_str = coords_str.split(',')
                        args[current_arg_name] = DSLPosition(row=int(r_str.strip()), col=int(c_str.strip()))
                    except Exception as e:
                        raise DSLParsingError(f"Invalid DSLPosition format for '{current_arg_name}': {arg_value_str}. Error: {e}")
                elif arg_value_str.startswith("DSLColor("):
                    try:
                        color_val_str = arg_value_str[len("DSLColor("):-1]
                        # Handle potential ARCPixel wrapping if needed, for now assume direct int
                        args[current_arg_name] = DSLColor(value=int(color_val_str.strip()))
                    except Exception as e:
                        raise DSLParsingError(f"Invalid DSLColor format for '{current_arg_name}': {arg_value_str}. Error: {e}")
                elif arg_value_str.startswith("DSLObjectSelector("):
                    try:
                        criteria_str = arg_value_str[len("DSLObjectSelector(criteria="):-1]
                        if criteria_str.startswith("{") and criteria_str.endswith("}"):
                            # Very basic dict parsing, not safe for general use
                            # Example: "{'old_color':0}"
                            inner_crit_str = criteria_str[1:-1].strip()
                            if not inner_crit_str: # Empty criteria dict
                                crit_dict = {}
                            else:
                                # Assuming simple structure like "'key':value"
                                key_val_pair = inner_crit_str.split(':')
                                dict_key = key_val_pair[0].strip().strip("'\"")
                                # This part is very fragile, assumes int value for color
                                dict_val = int(key_val_pair[1].strip())
                                crit_dict = {dict_key: dict_val}
                            args[current_arg_name] = DSLObjectSelector(criteria=crit_dict)
                        else:
                            raise DSLParsingError(f"Invalid DSLObjectSelector criteria format for '{current_arg_name}': {criteria_str}")
                    except Exception as e:
                        raise DSLParsingError(f"Invalid DSLObjectSelector format for '{current_arg_name}': {arg_value_str}. Error: {e}")
                else:
                    # Fallback for simple values (int, string - needs context)
                    # This is where knowing the expected type for the command's arg is crucial.
                    # For now, we'll assume if it's not a known complex type, it might be an error
                    # or needs command-specific handling.
                    raise DSLParsingError(f"Unsupported argument type or format for '{current_arg_name}' in command '{command_name}': {arg_value_str}")

                current_arg_name = None
                current_arg_value_str = ""
            else:
                current_arg_value_str += char
                if char == '(':
                    parenthesis_level += 1
                elif char == ')':
                    parenthesis_level -= 1
        
        return args


    def parse_dsl(self, dsl_string: str) -> DSLProgram:
        if dsl_string is None:
            return DSLProgram(operations=[]) # Or raise error, based on desired strictness
        
        dsl_string = dsl_string.strip()
        if not dsl_string:
            return DSLProgram(operations=[])

        operations: List[DSLOperation] = []
        lines = dsl_string.splitlines()

        for line_number, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                # Basic parsing: COMMAND_NAME(arg1=val1, arg2=val2)
                if '(' not in line or not line.endswith(')'):
                    raise DSLParsingError(f"Invalid command format on line {line_number}: '{line}'. Expected 'COMMAND_NAME(arguments)'.")

                command_name = line[:line.find('(')].strip()
                arg_string = line[line.find('(') + 1 : line.rfind(')')] # Content between outer parentheses

                if command_name not in COMMAND_MAP:
                    raise DSLParsingError(f"Unknown command '{command_name}' on line {line_number}.")
                
                op_class = COMMAND_MAP[command_name]
                
                # This argument parsing is highly simplified and needs to be robust.
                # For a real system, a proper argument parser (regex, dedicated library) is needed.
                parsed_args = self._parse_dsl_arguments(arg_string, command_name)
                
                # Validate required arguments based on op_class definition (not done here yet)
                # Example: FillRectangleOp requires 'top_left', 'bottom_right', 'color'
                # This would ideally use introspection or predefined schemas for each command.
                try:
                    operation_instance = op_class(**parsed_args)
                except TypeError as e: # Catches missing/unexpected arguments for op_class constructor
                    raise DSLParsingError(f"Argument error for command '{command_name}' on line {line_number}: {e}. Provided args: {parsed_args}")
                
                operations.append(operation_instance)

            except DSLParsingError as e: # Re-raise with line info if not already there
                if f"line {line_number}" not in str(e):
                    raise DSLParsingError(f"Error on line {line_number}: {e}")
                else:
                    raise e # Raise the original error that already includes line info
            except Exception as e: # Catch other unexpected errors during parsing of a line
                 raise DSLParsingError(f"Unexpected parsing error on line {line_number} ('{line}'): {e}")


        return DSLProgram(operations=operations)

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
        if not grid or not isinstance(grid, list):
             raise DSLExecutionError("Grid is malformed or not a list.")
        if not grid: # Empty grid (e.g. [])
            return 0,0 
        if not grid[0] or not isinstance(grid[0], list):
            # Grid like [[]] or [[1,2], []] - this logic might need refinement based on strictness
            raise DSLExecutionError("Grid contains empty rows or is malformed.")
        
        rows = len(grid)
        cols = len(grid[0])
        # Optional: Check for consistent row lengths
        # for r_idx, row_item in enumerate(grid):
        #     if len(row_item) != cols:
        #         raise DSLExecutionError(f"Grid has inconsistent row lengths at row {r_idx}.")
        return rows, cols

    def _execute_fill_rectangle(self, op: FillRectangleOp, grid: ARCGrid) -> ARCGrid:
        try:
            rows, cols = self._get_grid_dimensions(grid)
        except DSLExecutionError: # If grid is empty/malformed, no-op for fill
            return grid 
        if rows == 0 or cols == 0:
            return grid

        r1, c1 = op.top_left.row, op.top_left.col # Assuming DSLPosition has row, col directly
        r2, c2 = op.bottom_right.row, op.bottom_right.col
        color_to_fill = op.color.value

        start_row = max(0, min(r1, r2))
        end_row = min(rows - 1, max(r1, r2))
        start_col = max(0, min(c1, c2))
        end_col = min(cols - 1, max(c1, c2))

        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                if not (0 <= r < rows and 0 <= c < cols): # Should be redundant due to clipping, but defensive
                    continue
                grid[r][c] = color_to_fill
        return grid

    def _execute_change_color(self, op: ChangeColorOp, grid: ARCGrid) -> ARCGrid:
        try:
            rows, cols = self._get_grid_dimensions(grid)
        except DSLExecutionError:
            return grid
        if rows == 0 or cols == 0:
            return grid

        new_color_val = op.new_color.value
        criteria = op.selector.criteria # This is a Dict[str, Any]

        # Ensure criteria is a dictionary
        if not isinstance(criteria, dict):
            raise DSLExecutionError(f"ChangeColorOp selector criteria is not a dictionary: {criteria}")

        if 'position' in criteria:
            pos_val = criteria['position']
            if not isinstance(pos_val, DSLPosition): # Check if it's already a DSLPosition object
                 raise DSLExecutionError(f"ChangeColorOp 'position' criteria is not a DSLPosition object: {pos_val}")
            r, c = pos_val.row, pos_val.col
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = new_color_val
            # else: silently ignore out-of-bounds position changes, or raise error?
            # For now, ignore.
        elif 'old_color' in criteria:
            old_color_val = criteria['old_color']
            # If old_color_val was parsed into DSLColor, use .value. Otherwise, it's direct.
            # Based on current _parse_dsl_arguments, it's likely direct int for 'old_color':0 in criteria.
            # This part needs to align with how _parse_dsl_arguments actually stores it.
            # Assuming it's a raw value for now.
            if isinstance(old_color_val, DSLColor): old_color_val = old_color_val.value

            for r_idx in range(rows):
                for c_idx in range(cols):
                    if grid[r_idx][c_idx] == old_color_val:
                        grid[r_idx][c_idx] = new_color_val
        else:
            raise DSLExecutionError(f"ChangeColorOp selector criteria not understood or unsupported: {criteria}")
            
        return grid

    def _execute_move_op(self, op: MoveOp, grid: ARCGrid) -> ARCGrid:
        # Placeholder - actual implementation needed
        raise DSLExecutionError(f"MoveOp with selector '{op.selector.criteria}' to position '({op.destination.row},{op.destination.col})' not yet implemented.")
        # return grid

    def _execute_copy_object_op(self, op: CopyObjectOp, grid: ARCGrid) -> ARCGrid:
        raise DSLExecutionError(f"CopyObjectOp with selector '{op.selector.criteria}' to position '({op.destination.row},{op.destination.col})' not yet implemented.")
        # return grid

    def _execute_create_object_op(self, op: CreateObjectOp, grid: ARCGrid) -> ARCGrid:
        raise DSLExecutionError(f"CreateObjectOp at position '({op.destination.row},{op.destination.col})' not yet implemented.")
        # return grid

    def _execute_delete_object_op(self, op: DeleteObjectOp, grid: ARCGrid) -> ARCGrid:
        raise DSLExecutionError(f"DeleteObjectOp with selector '{op.selector.criteria}' not yet implemented.")
        # return grid

if __name__ == '__main__':
    # Example Usage
    print("ARCDSLInterpreter Example Usage")
    interpreter = ARCDSLInterpreter()

    print("\n--- Test DSL Parsing ---")
    dsl_string_valid = """
    # This is a comment
    FILL_RECTANGLE(top_left=DSLPosition(1,1), bottom_right=DSLPosition(2,2), color=DSLColor(5))
    CHANGE_COLOR(selector=DSLObjectSelector(criteria={'old_color':0}), new_color=DSLColor(7))
    """
    dsl_string_invalid_command = "UNKNOWN_COMMAND(args)"
    dsl_string_invalid_syntax = "FILL_RECTANGLE(top_left=DSLPosition(1,1)" # Missing parenthesis and args

    try:
        print(f"Parsing valid DSL:\n{dsl_string_valid}")
        program_valid = interpreter.parse_dsl(dsl_string_valid)
        print(f"Parsed program: {len(program_valid.operations)} operations")
        for op in program_valid.operations:
            print(f"  - {op}")
    except DSLParsingError as e:
        print(f"Error parsing valid DSL: {e}") # Should not happen

    try:
        print(f"\nParsing invalid command DSL: {dsl_string_invalid_command}")
        interpreter.parse_dsl(dsl_string_invalid_command)
    except DSLParsingError as e:
        print(f"Successfully caught error: {e}")

    try:
        print(f"\nParsing invalid syntax DSL: {dsl_string_invalid_syntax}")
        interpreter.parse_dsl(dsl_string_invalid_syntax)
    except DSLParsingError as e:
        print(f"Successfully caught error: {e}")
    
    print("\n--- Test Program Execution ---")
    # Use the successfully parsed program_valid if available, or define one
    if 'program_valid' not in locals():
         program_valid = DSLProgram(operations=[
            FillRectangleOp(
                top_left=DSLPosition(row=1, col=1), # Corrected to row, col
                bottom_right=DSLPosition(row=2, col=2),
                color=DSLColor(value=5)
            ),
            ChangeColorOp(
                selector=DSLObjectSelector(criteria={'old_color': 0}), 
                new_color=DSLColor(value=7)
            )
        ])


    initial_grid_exec: ARCGrid = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    print("\nInitial Grid for Execution:")
    for row in initial_grid_exec: print(row)

    try:
        final_grid = interpreter.execute_program(program_valid, initial_grid_exec)
        print("\nFinal Grid after valid program execution:")
        for row in final_grid: print(row)
    except DSLExecutionError as e:
        print(f"Error executing program: {e}")

    print("\n--- Test Execution Error (e.g., unimplemented op) ---")
    program_with_unimplemented = DSLProgram(operations=[
        MoveOp( 
            selector=DSLObjectSelector(criteria={'comment': 'object1'}),
            destination=DSLPosition(row=5,col=5) # Corrected to row,col
        )
    ])
    try:
        interpreter.execute_program(program_with_unimplemented, initial_grid_exec)
    except DSLExecutionError as e:
        print(f"Successfully caught execution error: {e}")
    
    print("\n--- Test Execution on Empty Grid ---")
    empty_grid_exec: ARCGrid = []
    try:
        # Note: _get_grid_dimensions will raise DSLExecutionError for empty grid if strict,
        # or operations should handle it.
        # Current FillRectangle/ChangeColor return grid if rows/cols is 0.
        # Let's test with a program that would normally modify.
        final_grid_from_empty_exec = interpreter.execute_program(program_valid, empty_grid_exec)
        print("Final grid from empty initial grid (depends on op handling):")
        for row in final_grid_from_empty_exec: print(row)
    except DSLExecutionError as e:
         print(f"Execution error on empty grid: {e}")