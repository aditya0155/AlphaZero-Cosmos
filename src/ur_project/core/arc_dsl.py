# src/ur_project/core/arc_dsl.py
from typing import List, Tuple, Union, Optional, Any, Dict
from dataclasses import dataclass, field

from ur_project.data_processing.arc_types import ARCPixel, ARCGrid
# Potential future import: from .knowledge_base import SymbolicEntity # If DSL operates on symbolic entities

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

@dataclass
class DSLPosition:
    """Represents a position, absolute or relative."""
    row: int
    col: int
    is_relative: bool = False # If true, (row, col) are deltas from a reference point

@dataclass
class DSLColor:
    """Represents a color, which could be an ARCPixel or a special keyword (e.g., 'background_color')."""
    value: Union[ARCPixel, str]

# --- DSL Operations (Version 1 - Basic) ---

@dataclass
class DSLOperation(ABC): # Python 3.9+ for ABC from dataclasses
    """Abstract base class for all DSL operations."""
    pass

@dataclass
class ChangeColorOp(DSLOperation):
    """Changes the color of selected object(s) or area."""
    selector: DSLObjectSelector # Could also be a region selector in future
    new_color: DSLColor
    operation_name: str = field(default="ChangeColor", init=False)

@dataclass
class MoveOp(DSLOperation):
    """Moves selected object(s)."""
    selector: DSLObjectSelector
    destination: DSLPosition # Could be absolute or relative to object's current position
    # Future: Could specify how to handle overlaps, boundaries etc.
    operation_name: str = field(default="Move", init=False)

@dataclass
class CopyObjectOp(DSLOperation):
    """Copies selected object(s) to a new position."""
    selector: DSLObjectSelector
    destination: DSLPosition # Top-left of where the copy should be placed
    # Future: Could include transformations during copy (e.g., recolor, resize)
    operation_name: str = field(default="CopyObject", init=False)

@dataclass
class CreateObjectOp(DSLOperation):
    """Creates a new object (e.g., draws a shape)."""
    shape_data: ARCGrid # The grid representing the shape to draw
    destination: DSLPosition # Top-left position to draw the shape
    color: Optional[DSLColor] = None # If shape_data is binary, this color is used. If shape_data has colors, this might override.
    operation_name: str = field(default="CreateObject", init=False)

@dataclass
class FillRectangleOp(DSLOperation):
    """Fills a rectangular area with a color."""
    top_left: DSLPosition
    bottom_right: DSLPosition
    color: DSLColor
    operation_name: str = field(default="FillRectangle", init=False)

@dataclass
class DeleteObjectOp(DSLOperation):
    """Deletes selected object(s) (e.g., changes their pixels to background color)."""
    selector: DSLObjectSelector
    operation_name: str = field(default="DeleteObject", init=False)

# --- DSL Program ---

@dataclass
class DSLProgram:
    """Represents a sequence of DSL operations to solve an ARC task."""
    operations: List[DSLOperation]
    # Future: Could include variables, control flow (loops, conditionals)

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    from abc import ABC # Add this import for the example to run standalone

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