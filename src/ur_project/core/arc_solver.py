# src/ur_project/core/arc_solver.py

from typing import Optional, List, Tuple
import json # For trying to parse LLM output if it's a stringified grid

from ur_project.core.solver import BaseSolver, Solution
from ur_project.core.foundational_llm import BaseLLM, LLMResponse
from ur_project.data_processing.arc_types import ARCPuzzle, ARCGrid, ARCPixel # ARCPuzzle is the Task
from ur_project.utils.visualization import visualize_arc_grid # For potential debug
from .perception import BaseFeatureExtractor, GridFeatures # Import perception components
from .knowledge_base import ARCKnowledgeBase # Import Knowledge Base
from .arc_dsl import (
    DSLProgram, DSLOperation, ChangeColorOp, MoveOp, CopyObjectOp,
    CreateObjectOp, FillRectangleOp, DeleteObjectOp,
    DSLObjectSelector, DSLPosition, DSLColor
) # Added DSL imports
from .arc_dsl_interpreter import ARCDSLInterpreter # Added DSL Interpreter import

class ARCSolver(BaseSolver):
    """
    A placeholder solver for ARC tasks using a foundational LLM.
    It attempts to generate the output grid, possibly as a string representation.
    It now uses a feature extractor to provide more context to the LLM.
    It also initializes and can use an ARCKnowledgeBase.
    """
    def __init__(self, llm: BaseLLM, feature_extractor: BaseFeatureExtractor, solver_id: str = "ARCSolver_LLM_v0.3"):
        self.llm = llm
        self.feature_extractor = feature_extractor
        self.kb = ARCKnowledgeBase() # Initialize a knowledge base instance
        self.solver_id = solver_id

    def _grid_to_string_representation(self, grid: ARCGrid) -> str:
        """Converts an ARCGrid to a simple string representation (list of lists)."""
        return json.dumps(grid)

    def _string_to_grid(self, grid_str: str) -> Optional[ARCGrid]:
        """Attempts to parse a string (e.g., LLM output from OUTPUT_GRID section) into an ARCGrid.
        """
        if not grid_str: # If the string for grid parsing is empty, return None early
            return None

        try:
            # Handle potential markdown code block ```python ... ``` or just ``` ... ```
            if grid_str.startswith("```python"):
                grid_str = grid_str.split("```python",1)[1].rsplit("```",1)[0].strip()
            elif grid_str.startswith("```"):
                 grid_str = grid_str.split("```",1)[1].rsplit("```",1)[0].strip()

            parsed_list = json.loads(grid_str)
            if not isinstance(parsed_list, list):
                return None
            # Basic validation: list of lists of ints
            grid: ARCGrid = []
            # Handle cases where parsed_list might be empty or contain non-list rows early
            if not parsed_list:
                return [] # Valid empty grid
            
            for row_idx, raw_row in enumerate(parsed_list):
                if not isinstance(raw_row, list):
                    print(f"ARCSolver parsing error: Row {row_idx} is not a list: {raw_row}")
                    return None
                # Handle empty rows within the grid if they are valid for ARC (usually not)
                # For now, let's assume rows must contain ARCPixels
                # if not raw_row and len(parsed_list) > 1: # allow [[],[]] only if that means empty grid
                #    print(f"ARCSolver parsing error: Row {row_idx} is empty.")
                #    return None

                current_row: List[ARCPixel] = []
                for col_idx, cell in enumerate(raw_row):
                    if not isinstance(cell, int):
                        # Try to coerce if it's a string representation of an int
                        if isinstance(cell, str) and cell.isdigit():
                            cell = int(cell)
                        else:
                            print(f"ARCSolver parsing error: Cell ({row_idx},{col_idx}) is not an int: {cell} (type: {type(cell)})")
                            return None
                    current_row.append(ARCPixel(cell))
                grid.append(current_row)
            
            # Further validation: check if all rows have the same length
            if grid:
                if not grid[0] and len(grid) > 1: # e.g. [[], [1]] is invalid, but [[]] might be ok if it implies 0x0 or 1x0
                    # This case implies a grid like [[]], which means 1 row, 0 columns.
                    # Or if grid was [[],[]] -> 2 rows, 0 columns
                    # This is valid if all rows are empty
                    if not all(not r for r in grid):
                        print(f"ARCSolver parsing error: Inconsistent empty/non-empty rows.")
                        return None
                elif grid[0]: # If first row is not empty, all must match its length
                    first_row_len = len(grid[0])
                    if not all(len(r) == first_row_len for r in grid):
                        print(f"ARCSolver parsing error: Not all rows have the same length.")
                        return None
            return grid
        except json.JSONDecodeError:
            print(f"ARCSolver parsing error: LLM output '{grid_str[:100]}...' is not valid JSON.")
            return None
        except Exception as e:
            print(f"ARCSolver parsing error: Unexpected error parsing grid string: {e}")
            return None

    def _parse_dsl_string_to_program(self, dsl_json_str: str) -> Optional[DSLProgram]:
        if not dsl_json_str or dsl_json_str.lower() == "[none]":
            return None

        try:
            # Handle potential markdown code block ```json ... ``` or just ``` ... ```
            if dsl_json_str.startswith("```json"):
                dsl_json_str = dsl_json_str.split("```json",1)[1].rsplit("```",1)[0].strip()
            elif dsl_json_str.startswith("```"):
                dsl_json_str = dsl_json_str.split("```",1)[1].rsplit("```",1)[0].strip()
            
            op_list_data = json.loads(dsl_json_str)
            if not isinstance(op_list_data, list):
                print(f"ARCSolver DSL parsing error: Expected a list of operations, got {type(op_list_data)}")
                return None

            parsed_ops: List[DSLOperation] = []
            for i, op_data in enumerate(op_list_data):
                if not isinstance(op_data, dict):
                    print(f"ARCSolver DSL parsing error: Operation {i} is not a dictionary: {op_data}")
                    continue

                op_name = op_data.get("operation_name")
                op_instance: Optional[DSLOperation] = None

                try:
                    if op_name == "ChangeColor":
                        selector_data = op_data.get("selector")
                        new_color_data = op_data.get("new_color")
                        if selector_data and new_color_data:
                            selector = DSLObjectSelector(criteria=selector_data.get("criteria", {})) # Add select_all_matching later if needed
                            new_color = DSLColor(value=new_color_data.get("value"))
                            op_instance = ChangeColorOp(selector=selector, new_color=new_color)
                        else:
                            print(f"ARCSolver DSL parsing error: Missing 'selector' or 'new_color' for ChangeColorOp: {op_data}")
                    
                    elif op_name == "FillRectangle":
                        tl_data = op_data.get("top_left")
                        br_data = op_data.get("bottom_right")
                        color_data = op_data.get("color")
                        if tl_data and br_data and color_data:
                            top_left = DSLPosition(coordinates=(tl_data.get("row"), tl_data.get("col")))
                            bottom_right = DSLPosition(coordinates=(br_data.get("row"), br_data.get("col")))
                            color = DSLColor(value=color_data.get("value"))
                            op_instance = FillRectangleOp(top_left=top_left, bottom_right=bottom_right, color=color)
                        else:
                             print(f"ARCSolver DSL parsing error: Missing data for FillRectangleOp: {op_data}")

                    elif op_name == "Move":
                        selector_data = op_data.get("selector")
                        dest_data = op_data.get("destination") # Renamed in prompt from new_position for consistency
                        if selector_data and dest_data:
                            selector = DSLObjectSelector(criteria=selector_data.get("criteria", {}))
                            # Assuming destination contains row, col, and is_relative
                            new_pos = DSLPosition(coordinates=(dest_data.get("row"), dest_data.get("col")))
                            is_relative = dest_data.get("is_relative", False) # Default to absolute
                            # Note: MoveOp in arc_dsl.py currently takes new_position, not destination & is_relative directly
                            # This will need alignment or more complex parsing here. For now, simplified:
                            op_instance = MoveOp(selector=selector, new_position=new_pos) # Simplified for now
                        else:
                            print(f"ARCSolver DSL parsing error: Missing 'selector' or 'destination' for MoveOp: {op_data}")
                    
                    elif op_name == "CopyObject":
                        selector_data = op_data.get("selector")
                        dest_data = op_data.get("destination") # Renamed in prompt from target_position
                        if selector_data and dest_data:
                            selector = DSLObjectSelector(criteria=selector_data.get("criteria", {}))
                            target_pos = DSLPosition(coordinates=(dest_data.get("row"), dest_data.get("col")))
                            op_instance = CopyObjectOp(selector=selector, target_position=target_pos)
                        else:
                            print(f"ARCSolver DSL parsing error: Missing 'selector' or 'destination' for CopyObjectOp: {op_data}")

                    elif op_name == "CreateObject":
                        # { "operation_name": "CreateObject", "shape_data": [[<pixel_int>, ...], ...], "destination": {"row": <y_int>, "col": <x_int>}, "color": {"value": <optional_fill_color_int>} }
                        shape_data = op_data.get("shape_data")
                        dest_data = op_data.get("destination")
                        # color_data for CreateObjectOp is more complex in current DSL (object_description, etc.)
                        # For now, let's parse position and make a placeholder description.
                        if shape_data and dest_data:
                            position = DSLPosition(coordinates=(dest_data.get("row"), dest_data.get("col")))
                            # Simplification: using shape_data as a string for object_description
                            object_description_str = f"Shape: {json.dumps(shape_data)}"
                            if op_data.get("color") and op_data["color"].get("value") is not None:
                                object_description_str += f", Color: {op_data['color']['value']}"
                            
                            op_instance = CreateObjectOp(
                                object_description=object_description_str, # Placeholder
                                position=position,
                                # color field in CreateObjectOp is not a DSLColor, it's part of description.
                                # This needs careful alignment with DSL definition if we want to pass full pixel data.
                                # For now, the prompt's CreateObjectOp is simpler.
                            )
                        else:
                            print(f"ARCSolver DSL parsing error: Missing 'shape_data' or 'destination' for CreateObjectOp: {op_data}")
                    
                    elif op_name == "DeleteObject":
                        selector_data = op_data.get("selector")
                        if selector_data:
                            selector = DSLObjectSelector(criteria=selector_data.get("criteria", {}))
                            op_instance = DeleteObjectOp(selector=selector)
                        else:
                            print(f"ARCSolver DSL parsing error: Missing 'selector' for DeleteObjectOp: {op_data}")
                    
                    else:
                        if op_name:
                            print(f"ARCSolver DSL parsing warning: Unknown operation_name '{op_name}' in op: {op_data}")
                        else:
                            print(f"ARCSolver DSL parsing error: Missing 'operation_name' in op: {op_data}")
                    
                    if op_instance:
                        # Further validation can be added here for types of coordinates, color values etc.
                        # e.g. for FillRectangleOp, check if coordinates are tuples of two integers
                        if isinstance(op_instance, FillRectangleOp):
                            if not (isinstance(op_instance.top_left.coordinates, tuple) and len(op_instance.top_left.coordinates) == 2 and \
                                    isinstance(op_instance.top_left.coordinates[0], int) and isinstance(op_instance.top_left.coordinates[1], int) and \
                                    isinstance(op_instance.bottom_right.coordinates, tuple) and len(op_instance.bottom_right.coordinates) == 2 and \
                                    isinstance(op_instance.bottom_right.coordinates[0], int) and isinstance(op_instance.bottom_right.coordinates[1], int) and \
                                    isinstance(op_instance.color.value, int)):
                                print(f"ARCSolver DSL parsing error: Invalid data types for FillRectangleOp fields: {op_data}")
                                op_instance = None # Discard malformed op
                        # Similar detailed validation for other ops if needed
                            
                        if op_instance:
                             parsed_ops.append(op_instance)

                except Exception as e:
                    print(f"ARCSolver DSL parsing error: Failed to instantiate DSL operation for '{op_name}' from {op_data}. Error: {e}")
                    # Continue to next operation
            
            if not parsed_ops and op_list_data: # If we had data but couldn't parse any ops
                 print(f"ARCSolver DSL parsing warning: No operations were successfully parsed from non-empty DSL string: {dsl_json_str[:200]}...")
                 return None
            elif not parsed_ops: # Completely empty or invalid initial JSON
                return None


            return DSLProgram(operations=parsed_ops)

        except json.JSONDecodeError:
            print(f"ARCSolver DSL parsing error: LLM output for DSL program '{dsl_json_str[:100]}...' is not valid JSON.")
            return None
        except Exception as e:
            print(f"ARCSolver DSL parsing error: Unexpected error parsing DSL string: {e}")
            return None

    def solve_task(self, task: ARCPuzzle) -> Solution:
        if not isinstance(task, ARCPuzzle):
            return Solution(
                task_id=task.id,
                solved_by=self.solver_id,
                raw_answer="Task type not supported by ARCSolver.",
                parsed_answer=None,
                metadata={"error": "Unsupported task type"}
            )

        input_grid_str = self._grid_to_string_representation(task.data) # task.data is the input_grid
        
        # Clear and populate KB for the current task context
        self.kb.clear_task_context()
        grid_features: Optional[GridFeatures] = self.feature_extractor.extract_features(task.data)
        
        symbolic_info_segment = "No symbolic information extracted or available."
        if grid_features:
            # Populate KB from grid_features
            grid_entity_id = "input_grid_entity"
            self.kb.add_grid_features_as_entity(grid_features, grid_id=grid_entity_id)
            for arc_obj in grid_features.objects:
                self.kb.add_arc_object_as_entity(arc_obj, grid_context="input")
            
            # TODO: Add relationship extraction and inference here
            # self.kb.infer_spatial_relationships() # Example

            # TODO: Query the KB and format symbolic information for the prompt
            # This is a placeholder for what could be extracted from the KB
            kb_derived_object_props = []
            for obj_entity_id, sym_entity in self.kb.entities.items():
                if sym_entity.entity_type == "arc_object":
                    props_str_list = []
                    for prop in sym_entity.properties:
                        props_str_list.append(f"{prop.name}: {prop.value.value}")
                    kb_derived_object_props.append(f"  Object {sym_entity.id}: { ' | '.join(props_str_list) }")
            if kb_derived_object_props:
                symbolic_info_segment = "Symbolic Object Properties (from KB):\n" + "\n".join(kb_derived_object_props)
            else:
                symbolic_info_segment = "Symbolic information (KB): No objects found or properties extracted into KB."

        features_str_parts = ["Input Grid Features (from Perception Module):"]
        if grid_features:
            features_str_parts.append(f"  - Dimensions: {grid_features.grid_height}x{grid_features.grid_width}")
            features_str_parts.append(f"  - Background Color: {grid_features.background_color}")
            features_str_parts.append(f"  - Unique Colors: {sorted(list(grid_features.unique_colors))}")
            features_str_parts.append(f"  - Number of Objects: {len(grid_features.objects)}")
            for i, obj in enumerate(grid_features.objects[:3]): # Show details for first 3 objects
                features_str_parts.append(f"    Object {i+1}: Color={obj.color}, Size={obj.pixel_count}, Centroid=({obj.centroid[0]:.1f},{obj.centroid[1]:.1f}), BBox={obj.bounding_box}")
            if len(grid_features.objects) > 3:
                features_str_parts.append(f"    ... and {len(grid_features.objects) - 3} more objects.")
        else:
            features_str_parts.append("  - Could not extract features.")
        
        features_prompt_segment = "\n".join(features_str_parts)

        task_text_description_segment = ""
        if task.text_description:
            task_text_description_segment = f"\nTask Textual Description (from source):\n{task.text_description}\n"
        
        # DSL Explanation for the prompt
        dsl_explanation = ( 
            "You can use a simple DSL to describe transformations. Represent the DSL program as a JSON list of operation objects.\n"
            "Available operations:\n"
            "- ChangeColorOp: { \"operation_name\": \"ChangeColor\", \"selector\": {\"criteria\": {\"color\": <old_color_int>}, \"select_all_matching\": true}, \"new_color\": {\"value\": <new_color_int>} }\n"
            "- MoveOp: { \"operation_name\": \"Move\", \"selector\": {\"criteria\": {\"id\": \"<object_id_from_perception>\"}}, \"destination\": {\"row\": <y_int>, \"col\": <x_int>, \"is_relative\": <bool>} }\n"
            "- CopyObjectOp: { \"operation_name\": \"CopyObject\", \"selector\": {\"criteria\": {\"id\": \"<object_id_from_perception>\"}}, \"destination\": {\"row\": <y_int>, \"col\": <x_int>} }\n"
            "- CreateObjectOp: { \"operation_name\": \"CreateObject\", \"shape_data\": [[<pixel_int>, ...], ...], \"destination\": {\"row\": <y_int>, \"col\": <x_int>}, \"color\": {\"value\": <optional_fill_color_int>} }\n"
            "- FillRectangleOp: { \"operation_name\": \"FillRectangle\", \"top_left\": {\"row\": <y1_int>, \"col\": <x1_int>}, \"bottom_right\": {\"row\": <y2_int>, \"col\": <x2_int>}, \"color\": {\"value\": <color_int>} }\n"
            "- DeleteObjectOp: { \"operation_name\": \"DeleteObject\", \"selector\": {\"criteria\": {\"color\": <color_int>}} }\n"
            "Selectors can use criteria like {\"color\": ...}, {\"id\": \"input_obj_N\" (use IDs from Symbolic Object Properties if available)}, {\"size\": ...}, etc.\n"
            "Object IDs for selectors like 'input_obj_1', 'input_obj_2' correspond to the order objects are listed in Symbolic Object Properties if provided, or conceptual objects if not."
        )

        prompt = (
            f"You are an expert ARC puzzle solver. Analyze the input grid and its features to determine the transformation needed.\n"
            f"{task_text_description_segment}\n"
            f"Input Grid (raw, list of lists of integers):\n{input_grid_str}\n\n"
            f"{features_prompt_segment}\n\n"
            f"Symbolic Information (from Knowledge Base):\n{symbolic_info_segment}\n\n"
            f"Think step-by-step to solve this puzzle. Explain your reasoning before providing the final grid. "
            f"Consider the properties of objects, their relationships, and any transformations that might map the input to a plausible output.\n"
            f"If symbolic information about objects and relationships is provided, use it to guide your reasoning.\n\n"
            f"Available DSL for transformations:\n{dsl_explanation}\n\n"
            f"Your thought process should be clearly articulated. "
            f"Then, provide a DSL program if you can formulate one. "
            f"Finally, provide your answer *only* as a Python-style list of lists of integers "
            f"representing the output grid. For example: [[1, 1], [1, 1]]\n"
            f"Format your response like this:\n"
            f"THOUGHTS:\n<your detailed step-by-step reasoning here>\n"
            f"HYPOTHESIZED_TRANSFORMATIONS_OR_RULES:\n<describe potential transformations or rules, e.g., \"all blue objects become red\", \"reflect input grid horizontally\">\n"
            f"DSL_PROGRAM:\n<your DSL program as a JSON list of operations, or \"[NONE]\" if not applicable>\n"
            f"OUTPUT_GRID:\n<your python list of lists for the grid here>"
        )

        llm_response: LLMResponse = self.llm.generate(
            prompt,
            max_new_tokens=1536, # Increased further for DSL program
            temperature=0.2 
        )

        raw_answer_text = llm_response.text.strip()
        
        # Extract different parts of the response
        thoughts_text = ""
        hypotheses_text = ""
        dsl_program_str = ""
        grid_str_for_parsing = raw_answer_text # Fallback

        thoughts_marker = "THOUGHTS:"
        hypotheses_marker = "HYPOTHESIZED_TRANSFORMATIONS_OR_RULES:"
        dsl_marker = "DSL_PROGRAM:"
        output_grid_marker = "OUTPUT_GRID:"

        # More robust parsing order
        current_text = raw_answer_text

        if thoughts_marker in current_text:
            _, content_after_thoughts = current_text.split(thoughts_marker, 1)
            thoughts_text = content_after_thoughts.split(hypotheses_marker, 1)[0].strip() if hypotheses_marker in content_after_thoughts else \
                            content_after_thoughts.split(dsl_marker, 1)[0].strip() if dsl_marker in content_after_thoughts else \
                            content_after_thoughts.split(output_grid_marker, 1)[0].strip() if output_grid_marker in content_after_thoughts else \
                            content_after_thoughts.strip()
            current_text = content_after_thoughts.split(thoughts_text,1)[-1].strip() if thoughts_text in content_after_thoughts else current_text
        
        if hypotheses_marker in current_text:
            _, content_after_hypotheses = current_text.split(hypotheses_marker, 1)
            hypotheses_text = content_after_hypotheses.split(dsl_marker, 1)[0].strip() if dsl_marker in content_after_hypotheses else \
                              content_after_hypotheses.split(output_grid_marker, 1)[0].strip() if output_grid_marker in content_after_hypotheses else \
                              content_after_hypotheses.strip()
            current_text = content_after_hypotheses.split(hypotheses_text,1)[-1].strip() if hypotheses_text in content_after_hypotheses else current_text

        if dsl_marker in current_text:
            _, content_after_dsl = current_text.split(dsl_marker, 1)
            dsl_program_str = content_after_dsl.split(output_grid_marker, 1)[0].strip() if output_grid_marker in content_after_dsl else \
                              content_after_dsl.strip()
            grid_str_for_parsing = content_after_dsl.split(output_grid_marker, 1)[-1].strip() if output_grid_marker in content_after_dsl else ""
        elif output_grid_marker in current_text: # No DSL marker, but grid marker exists
            grid_str_for_parsing = current_text.split(output_grid_marker, 1)[-1].strip()
        else: # No DSL and no Grid marker after thoughts/hypotheses
            grid_str_for_parsing = ""

        parsed_grid: Optional[ARCGrid] = self._string_to_grid(grid_str_for_parsing)

        if parsed_grid is None and grid_str_for_parsing and grid_str_for_parsing.lower() != "[none]":
            print(f"ARCSolver: Failed to parse LLM output for task {task.id} into a valid grid. Raw output for grid part: '{grid_str_for_parsing[:200]}...'")
        
        solution_obj = Solution(
            task_id=task.id,
            solved_by=self.solver_id,
            raw_answer=raw_answer_text,
            parsed_answer=parsed_grid, 
            metadata=llm_response.metadata
        )
        solution_obj.hypothesized_transformations = hypotheses_text if hypotheses_text else None
        solution_obj.raw_dsl_program = dsl_program_str if dsl_program_str and dsl_program_str.lower() != "[none]" else None
        solution_obj.parsed_dsl_program = None # Initialize
        
        executed_grid_from_dsl: Optional[ARCGrid] = None

        if solution_obj.raw_dsl_program:
            parsed_program = self._parse_dsl_string_to_program(solution_obj.raw_dsl_program)
            solution_obj.parsed_dsl_program = parsed_program
            
            if parsed_program and parsed_program.operations: # Ensure there are operations to execute
                try:
                    interpreter = ARCDSLInterpreter()
                    # Ensure task.data is in the correct ARCGrid format if it isn't already
                    # The interpreter expects List[List[int]] based on its current implementation.
                    # ARCPuzzle.data should already be ARCGrid, which is List[List[ARCPixel]].
                    # We need to convert List[List[ARCPixel]] to List[List[int]] for the interpreter.
                    grid_for_interpreter: List[List[int]] = [[pixel.value for pixel in row] for row in task.data]
                    
                    print(f"ARCSolver: Attempting to execute DSL program with {len(parsed_program.operations)} operations.")
                    executed_grid_raw = interpreter.execute_program(parsed_program, grid_for_interpreter)
                    
                    # Convert executed_grid_raw (List[List[int]]) back to ARCGrid (List[List[ARCPixel]])
                    if executed_grid_raw is not None:
                        executed_grid_from_dsl = [[ARCPixel(value) for value in row] for row in executed_grid_raw]
                        print(f"ARCSolver: DSL program executed successfully.")
                        # If DSL execution produced a grid, prioritize it as the parsed_answer
                        solution_obj.parsed_answer = executed_grid_from_dsl 
                        # Store some metadata about DSL execution success
                        solution_obj.metadata["dsl_execution_status"] = "success"
                        solution_obj.metadata["dsl_operations_executed"] = len(parsed_program.operations)
                    else:
                        print(f"ARCSolver: DSL program execution resulted in None grid.")
                        solution_obj.metadata["dsl_execution_status"] = "execution_returned_none"

                except Exception as e:
                    print(f"ARCSolver: Error during DSL program execution: {e}")
                    solution_obj.metadata["dsl_execution_status"] = "execution_error"
                    solution_obj.metadata["dsl_execution_error_message"] = str(e)
            elif parsed_program: # Parsed but no operations
                print(f"ARCSolver: DSL program parsed but contained no operations.")
                solution_obj.metadata["dsl_execution_status"] = "no_operations_to_execute"
            else: # Parsing failed
                solution_obj.metadata["dsl_execution_status"] = "parsing_failed"

        # If DSL execution did not produce a grid, the original parsed_grid (from LLM's OUTPUT_GRID) remains.
        # If DSL produced a grid, solution_obj.parsed_answer is now updated.

        return solution_obj

# Example Usage (requires a BaseLLM and ARCPuzzle instance):
# if __name__ == '__main__':
#     from ur_project.core.foundational_llm import HuggingFaceLLM # Or a MockLLM
#     from ur_project.core.arc_proposer import ARCProposer
#     # This is a mock LLM for testing purposes
#     class MockLLMForARCSolver(BaseLLM):
#         def _load_model(self):
#             print("MockLLMForARCSolver loaded.")
#         def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, **kwargs) -> LLMResponse:
#             print(f"MockLLMForARCSolver received prompt (first 100 chars): {prompt[:100]}...")
#             # Example: If input is [[1]], pretend LLM outputs [[2]]
#             if "[[1]]" in prompt: 
#                 return LLMResponse("[[2]]")
#             # Example: If input is [[1,0],[0,1]], pretend LLM outputs [[0,1],[1,0]]
#             elif "[[1, 0], [0, 1]]" in prompt:
#                 return LLMResponse("[[0, 1], [1, 0]]")
#             elif "[[[1, 2], [3, 4]]]" in prompt: # Test nested if grid_to_string produces that (it shouldn't for ARCPixel)
#                 return LLMResponse("[[[5,6],[7,8]]]") # An invalid grid format for ARCPixel based grid
#             else:
#                 return LLMResponse("This is not a valid grid string.")
#         def batch_generate(self, prompts: list[str], **kwargs) -> list[LLMResponse]:
#             return [self.generate(p) for p in prompts]

#     # Setup - this requires ARC data to be available for the proposer
#     # Ensure ARCProposer can find some tasks.
#     # Create dummy task files if necessary or point to a real ARC data subset.
#     dummy_arc_dir = "./temp_arc_data/training"
#     os.makedirs(dummy_arc_dir, exist_ok=True)
#     dummy_task_content = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[1]], "output": [[2]]}]}
#     with open(os.path.join(dummy_arc_dir, "dummy01.json"), "w") as f:
#         json.dump(dummy_task_content, f)
#     dummy_task_content_2 = {"train": [{"input": [[1,0],[0,1]], "output": [[0,1],[1,0]]}], "test": []}
#     with open(os.path.join(dummy_arc_dir, "dummy02.json"), "w") as f:
#         json.dump(dummy_task_content_2, f)

#     try:
#         arc_proposer = ARCProposer(arc_tasks_directory=dummy_arc_dir)
#         mock_llm = MockLLMForARCSolver(model_path_or_name="mock_arc_solver_llm")
#         feature_extractor_instance = BasicARCFeatureExtractor()
#         arc_solver = ARCSolver(llm=mock_llm, feature_extractor=feature_extractor_instance)

#         for _ in range(2):
#             puzzle_to_solve: ARCPuzzle = arc_proposer.propose_task()
#             print(f"\nAttempting to solve ARC Puzzle ID: {puzzle_to_solve.id}")
#             print(f"Input Grid (data): {puzzle_to_solve.data}")
            
#             solution = arc_solver.solve_task(puzzle_to_solve)
#             print(f"  Solver ID: {solution.solved_by}")
#             print(f"  Raw Answer: '{solution.raw_answer}'")
#             print(f"  Parsed Answer (Grid): {solution.parsed_answer}")
#             if solution.parsed_answer:
#                 visualize_arc_grid(solution.parsed_answer, title=f"Solver Output for {puzzle_to_solve.id}")
#             print(f"  Expected Output Grid: {puzzle_to_solve.expected_output_grid}")
#             visualize_arc_grid(puzzle_to_solve.expected_output_grid, title=f"Expected Output for {puzzle_to_solve.id}")
#             print("---")

#     except ValueError as e:
#         print(f"Error setting up ARCProposer (ensure data exists): {e}")
#     except RuntimeError as e:
#         print(f"Runtime error during example: {e}")
#     except ImportError:
#         print("ImportError: matplotlib likely not installed, skipping visualization in example.")
#     finally:
#         # Clean up dummy files
#         import shutil
#         if os.path.exists("./temp_arc_data"):
#             shutil.rmtree("./temp_arc_data") 