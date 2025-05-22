# src/ur_project/core/arc_dsl.py
from typing import List, Tuple, Union, Optional, Any, Dict
from dataclasses import dataclass, field, asdict, is_dataclass
from abc import ABC, abstractmethod

from ur_project.data_processing.arc_types import ARCPixel, ARCGrid
# Potential future import: from .knowledge_base import SymbolicEntity # If DSL operates on symbolic entities

# --- Helper function for to_dict ---

def _convert_value(value: Any) -> Any:
    """Recursively converts values for to_dict methods."""
    if isinstance(value, ARCPixel):
        return value.value
    elif isinstance(value, list):
        return [_convert_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    elif hasattr(value, 'to_dict') and callable(value.to_dict):
        return value.to_dict()
    # Check for dataclasses that are not our specific DSL objects but might contain them
    # This handles cases like ARCGrid (List[List[ARCPixel]]) correctly
    # if it wasn't handled by the isinstance(value, list) above.
    # However, ARCGrid is List[List[ARCPixel]] so the list comprehension handles it.
    # Generic dataclasses not part of DSL but containing DSL parts would need explicit handling
    # if not covered by the above. For now, this seems sufficient.
    return value

# --- DSL Core Concepts ---

@dataclass
class DSLObjectSelector:
    """Specifies how to select an object or objects for an operation."""
    # Selection criteria (can be extended significantly)
    # Examples:
    #   - by color: { "color": ARCPixel(1) }
    #   - by id (if objects are uniquely identified by a perception system): { "id": "object_1" }
    #   - by position (e.g., a specific pixel, or objects within a bounding box)
    #   - by properties (e.g., all objects of size > 5)
    criteria: Dict[str, Any]
    select_all_matching: bool = False # If true, operation applies to all matching, else typically the first or a specific one.

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Specifically handle the 'criteria' dict as its values might be ARCPixel or other DSL objects
        if 'criteria' in data and isinstance(data['criteria'], dict):
            data['criteria'] = {k: _convert_value(v) for k, v in data['criteria'].items()}
        # Apply conversion to other fields as a general step, though criteria is the main complex one here
        for key, value in data.items():
            if key != 'criteria': # Already handled
                 data[key] = _convert_value(value)
        return data

@dataclass
class DSLPosition:
    """Represents a position, absolute or relative."""
    row: int
    col: int
    is_relative: bool = False # If true, (row, col) are deltas from a reference point

    def to_dict(self) -> Dict[str, Any]:
        # Simple fields, asdict is enough, _convert_value handles potential ARCPixel if they were part of this
        return asdict(self) # No complex types like ARCPixel or nested DSL objects expected here directly
                            # but if they were, _convert_value would be needed in a loop like DSLObjectSelector
                            # For now, assuming row/col/is_relative are basic types.
                            # If DSLPosition could hold an ARCPixel for e.g. a coordinate, then conversion would be needed.
                            # Let's stick to the provided example and keep it simple.
                            # data = asdict(self)
                            # for key, value in data.items(): data[key] = _convert_value(value)
                            # return data
                            # Given current structure, direct asdict is fine.
        return asdict(self)

@dataclass
class DSLColor:
    """Represents a color, which could be an ARCPixel or a special keyword (e.g., 'background_color')."""
    value: Union[ARCPixel, str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['value'] = _convert_value(self.value)
        return data

# --- DSL Operations (Version 1 - Basic) ---

@dataclass
class BaseOperation(ABC): # Python 3.9+ for ABC from dataclasses
    """Abstract base class for all DSL operations."""
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            data[key] = _convert_value(value)
        return data

@dataclass
class ChangeColorOp(BaseOperation):
    """Changes the color of selected object(s) or area."""
    selector: DSLObjectSelector # Could also be a region selector in future
    new_color: DSLColor
    operation_name: str = field(default="ChangeColor", init=False)

@dataclass
class MoveOp(BaseOperation):
    """Moves selected object(s)."""
    selector: DSLObjectSelector
    destination: DSLPosition # Could be absolute or relative to object's current position
    # Future: Could specify how to handle overlaps, boundaries etc.
    operation_name: str = field(default="Move", init=False)

@dataclass
class CopyObjectOp(BaseOperation):
    """Copies selected object(s) to a new position."""
    selector: DSLObjectSelector
    destination: DSLPosition # Top-left of where the copy should be placed
    # Future: Could include transformations during copy (e.g., recolor, resize)
    operation_name: str = field(default="CopyObject", init=False)

@dataclass
class CreateObjectOp(BaseOperation):
    """Creates a new object (e.g., draws a shape)."""
    shape_data: ARCGrid # The grid representing the shape to draw
    destination: DSLPosition # Top-left position to draw the shape
    color: Optional[DSLColor] = None # If shape_data is binary, this color is used. If shape_data has colors, this might override.
    operation_name: str = field(default="CreateObject", init=False)

@dataclass
class FillRectangleOp(BaseOperation):
    """Fills a rectangular area with a color."""
    top_left: DSLPosition
    bottom_right: DSLPosition
    color: DSLColor
    operation_name: str = field(default="FillRectangle", init=False)

@dataclass
class DeleteObjectOp(BaseOperation):
    """Deletes selected object(s) (e.g., changes their pixels to background color)."""
    selector: DSLObjectSelector
    operation_name: str = field(default="DeleteObject", init=False)

# --- Conditional Logic DSL Components ---

@dataclass
class Condition(ABC):
    """Abstract base class for different types of conditions."""
    condition_type: str = field(init=False)

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Converts the condition to a dictionary."""
        pass

@dataclass
class PixelMatchesCondition(Condition):
    """Condition that checks if a pixel at a position matches a specific color."""
    position: DSLPosition
    expected_color: DSLColor
    condition_type: str = field(default="PixelMatchesCondition", init=False)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['position'] = _convert_value(self.position) # Uses helper for consistency
        data['expected_color'] = _convert_value(self.expected_color) # Uses helper for consistency
        return data

@dataclass
class ObjectExistsCondition(Condition):
    """Condition that checks if an object matching certain criteria exists."""
    selector: DSLObjectSelector
    condition_type: str = field(default="ObjectExistsCondition", init=False)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['selector'] = _convert_value(self.selector) # Uses helper for consistency
        return data

@dataclass
class ConditionalBranch:
    """Represents a branch of execution: a condition and a program to run if true."""
    condition: Condition # Union of specific condition types can be enforced by type checkers if needed
    program: 'DSLProgram' # Forward reference for DSLProgram

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": _convert_value(self.condition),
            "program": _convert_value(self.program)
        }

@dataclass
class IfElseOp(BaseOperation):
    """Operation for conditional logic (if-else)."""
    main_branch: ConditionalBranch
    else_branch: Optional['DSLProgram'] = None # Optional 'else' program
    operation_name: str = field(default="IfElse", init=False)

    # to_dict is inherited from BaseOperation and should work correctly
    # as long as ConditionalBranch and DSLProgram have proper to_dict methods
    # and _convert_value handles them, which it does.

# --- DSL Program ---

@dataclass
class DSLProgram:
    """Represents a sequence of DSL operations to solve an ARC task."""
    operations: List[BaseOperation]  # IfElseOp is a BaseOperation, so this is fine.
    # Future: Could include variables, control flow (loops, conditionals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operations": [_convert_value(op) for op in self.operations]
            # If DSLProgram had other fields needing conversion, handle them here
        }

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # (Previous examples from DSLProgram) ...
    # Example: Select all red objects
    selector_red_objects = DSLObjectSelector(criteria={"color": ARCPixel(2)}, select_all_matching=True)
    
    # Example: Define a new color (blue)
    new_blue_color = DSLColor(value=ARCPixel(1))
    
    # Operation 1: Change all red objects to blue
    op1_change_to_blue = ChangeColorOp(selector=selector_red_objects, new_color=new_blue_color)
    print(f"Op1: {op1_change_to_blue.operation_name}, Selector: {op1_change_to_blue.selector.criteria}, NewColor: {op1_change_to_blue.new_color.value}")

    # Example: Define a specific object (e.g., by a conceptual ID if perception provides it)
    selector_specific_obj = DSLObjectSelector(criteria={"id": "perceived_object_0"})
    
    # Example: Define a relative move (move 2 down, 1 right)
    relative_move_pos = DSLPosition(row=2, col=1, is_relative=True)
    
    # Operation 2: Move the specific object
    op2_move_specific = MoveOp(selector=selector_specific_obj, destination=relative_move_pos)
    print(f"Op2: {op2_move_specific.operation_name}, Selector: {op2_move_specific.selector.criteria}, Dest: ({op2_move_specific.destination.row},{op2_move_specific.destination.col}) rel={op2_move_specific.destination.is_relative}")

    # Example: Create a small L-shape object
    l_shape: ARCGrid = [[ARCPixel(5), ARCPixel(0)], [ARCPixel(5), ARCPixel(0)], [ARCPixel(5), ARCPixel(5)]]
    draw_pos = DSLPosition(row=0, col=0)
    op3_create_l = CreateObjectOp(shape_data=l_shape, destination=draw_pos, color=DSLColor(ARCPixel(7))) # Draw in color 7
    print(f"Op3: {op3_create_l.operation_name}, Dest: ({op3_create_l.destination.row},{op3_create_l.destination.col}), Shape rows: {len(op3_create_l.shape_data)}")

    # DSL Program
    program = DSLProgram(operations=[op1_change_to_blue, op2_move_specific, op3_create_l])
    print(f"\nProgram has {len(program.operations)} operations.")
    for i, op in enumerate(program.operations):
        print(f"  Step {i}: {op.operation_name}") 

    # Example for Conditional Logic
    print("\n--- Conditional Logic Example ---")

    # 1. Define a condition: Check if pixel (0,0) is black (ARCPixel(0))
    cond_pixel_is_black = PixelMatchesCondition(
        position=DSLPosition(row=0, col=0),
        expected_color=DSLColor(value=ARCPixel(0))
    )
    print(f"Condition defined: {cond_pixel_is_black.to_dict()}")

    # 2. Define a sub-program to run if condition is true
    #    (e.g., fill rectangle (1,1)-(2,2) with blue (ARCPixel(1)))
    program_if_true = DSLProgram(operations=[
        FillRectangleOp(
            top_left=DSLPosition(1,1),
            bottom_right=DSLPosition(2,2),
            color=DSLColor(ARCPixel(1))
        )
    ])
    print(f"Sub-program if true: {program_if_true.to_dict()}")

    # 3. Create a conditional branch
    main_branch = ConditionalBranch(
        condition=cond_pixel_is_black,
        program=program_if_true
    )
    print(f"Main branch defined: {main_branch.to_dict()}")

    # 4. (Optional) Define an 'else' program
    #    (e.g., delete object with color red (ARCPixel(2)))
    program_if_false = DSLProgram(operations=[
        DeleteObjectOp(selector=DSLObjectSelector(criteria={"color": ARCPixel(2)}))
    ])
    print(f"Sub-program if false (else branch): {program_if_false.to_dict()}")

    # 5. Create the IfElse Operation
    if_else_op = IfElseOp(
        main_branch=main_branch,
        else_branch=program_if_false
    )
    print(f"IfElseOp defined: {if_else_op.to_dict()}")
    
    # Add it to the main program
    program.operations.append(if_else_op)

    print(f"\nUpdated program has {len(program.operations)} operations.")
    for i, op in enumerate(program.operations):
        print(f"  Step {i}: {op.operation_name}") 
        if op.operation_name == "IfElse":
            print(f"    If condition: {op.main_branch.condition.condition_type}") # type: ignore
            print(f"    Else branch exists: {op.else_branch is not None}") # type: ignore

    # Example for ObjectExistsCondition
    cond_object_exists = ObjectExistsCondition(
        selector=DSLObjectSelector(criteria={"color": ARCPixel(3)}, select_all_matching=False)
    )
    print(f"ObjectExistsCondition defined: {cond_object_exists.to_dict()}")

    if_object_exists_op = IfElseOp(
        main_branch=ConditionalBranch(
            condition=cond_object_exists,
            program=DSLProgram(operations=[CreateObjectOp(ARCGrid([[ARCPixel(8)]]), DSLPosition(5,5))])
        )
    )
    print(f"IfElseOp with ObjectExistsCondition: {if_object_exists_op.to_dict()}")