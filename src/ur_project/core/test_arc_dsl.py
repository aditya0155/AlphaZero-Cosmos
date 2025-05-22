# src/ur_project/core/test_arc_dsl.py
import unittest
# Import the module itself to check its attributes for renaming test
import ur_project.core.arc_dsl as arc_dsl_module
from ur_project.core.arc_dsl import (
    BaseOperation, DSLObjectSelector, DSLPosition, DSLColor,
    ChangeColorOp, MoveOp, CopyObjectOp, CreateObjectOp, FillRectangleOp, DeleteObjectOp,
    DSLProgram,
    Condition, PixelMatchesCondition, ObjectExistsCondition, ConditionalBranch, IfElseOp
)
# Make sure to import ARCPixel if you need to create instances for testing
from ur_project.data_processing.arc_types import ARCPixel, ARCGrid 

# If ur_project.data_processing.arc_types is not directly importable,
# adjust the import path as needed. Assume it's available for now.

class TestArcDslRenaming(unittest.TestCase):
    def test_base_operation_exists(self):
        self.assertTrue(hasattr(arc_dsl_module, 'BaseOperation'))
        self.assertFalse(hasattr(arc_dsl_module, 'DSLOperation'))

    def test_subclass_inheritance(self):
        self.assertTrue(issubclass(ChangeColorOp, BaseOperation))
        self.assertTrue(issubclass(MoveOp, BaseOperation))
        self.assertTrue(issubclass(CopyObjectOp, BaseOperation))
        self.assertTrue(issubclass(CreateObjectOp, BaseOperation))
        self.assertTrue(issubclass(FillRectangleOp, BaseOperation))
        self.assertTrue(issubclass(DeleteObjectOp, BaseOperation))
        self.assertTrue(issubclass(IfElseOp, BaseOperation)) # Also an operation

class TestArcDslToDict(unittest.TestCase):
    def test_dsl_color_to_dict(self):
        color = DSLColor(value=ARCPixel(5))
        expected_dict = {"value": 5}
        self.assertEqual(color.to_dict(), expected_dict)

        color_str = DSLColor(value="background")
        expected_dict_str = {"value": "background"}
        self.assertEqual(color_str.to_dict(), expected_dict_str)

    def test_dsl_position_to_dict(self):
        pos = DSLPosition(row=1, col=2, is_relative=True)
        expected_dict = {"row": 1, "col": 2, "is_relative": True}
        self.assertEqual(pos.to_dict(), expected_dict)

    def test_dsl_object_selector_to_dict(self):
        selector = DSLObjectSelector(criteria={"color": ARCPixel(3), "size": 5}, select_all_matching=True)
        expected_dict = {"criteria": {"color": 3, "size": 5}, "select_all_matching": True}
        self.assertEqual(selector.to_dict(), expected_dict)
        
        selector_nested = DSLObjectSelector(criteria={"anchor_pixel": DSLPosition(0,0)})
        expected_dict_nested = {"criteria": {"anchor_pixel": {"row":0, "col":0, "is_relative": False}}, "select_all_matching": False}
        self.assertEqual(selector_nested.to_dict(), expected_dict_nested)

    def test_change_color_op_to_dict(self):
        selector = DSLObjectSelector(criteria={"color": ARCPixel(2)})
        new_color = DSLColor(value=ARCPixel(1))
        op = ChangeColorOp(selector=selector, new_color=new_color)
        expected_dict = {
            "operation_name": "ChangeColor",
            "selector": {"criteria": {"color": 2}, "select_all_matching": False},
            "new_color": {"value": 1}
        }
        self.assertEqual(op.to_dict(), expected_dict)

    def test_move_op_to_dict(self):
        selector = DSLObjectSelector(criteria={"id": "obj1"})
        destination = DSLPosition(row=5, col=5, is_relative=False)
        op = MoveOp(selector=selector, destination=destination)
        expected_dict = {
            "operation_name": "Move",
            "selector": {"criteria": {"id": "obj1"}, "select_all_matching": False},
            "destination": {"row": 5, "col": 5, "is_relative": False}
        }
        self.assertEqual(op.to_dict(), expected_dict)

    def test_copy_object_op_to_dict(self):
        selector = DSLObjectSelector(criteria={"color": ARCPixel(7)})
        destination = DSLPosition(row=2, col=3, is_relative=True)
        op = CopyObjectOp(selector=selector, destination=destination)
        expected_dict = {
            "operation_name": "CopyObject",
            "selector": {"criteria": {"color": 7}, "select_all_matching": False},
            "destination": {"row": 2, "col": 3, "is_relative": True}
        }
        self.assertEqual(op.to_dict(), expected_dict)

    def test_create_object_op_to_dict(self):
        shape: ARCGrid = [[ARCPixel(1), ARCPixel(0)], [ARCPixel(1), ARCPixel(1)]]
        dest = DSLPosition(1,1)
        color = DSLColor(ARCPixel(5))
        op = CreateObjectOp(shape_data=shape, destination=dest, color=color)
        expected_dict = {
            "operation_name": "CreateObject",
            "shape_data": [[1,0], [1,1]],
            "destination": {"row":1, "col":1, "is_relative":False},
            "color": {"value": 5}
        }
        self.assertEqual(op.to_dict(), expected_dict)
        
        # Test with no explicit color (should be None)
        op_no_color = CreateObjectOp(shape_data=shape, destination=dest)
        expected_dict_no_color = {
            "operation_name": "CreateObject",
            "shape_data": [[1,0], [1,1]],
            "destination": {"row":1, "col":1, "is_relative":False},
            "color": None
        }
        self.assertEqual(op_no_color.to_dict(), expected_dict_no_color)


    def test_fill_rectangle_op_to_dict(self):
        top_left = DSLPosition(row=0, col=0)
        bottom_right = DSLPosition(row=4, col=4)
        color = DSLColor(value=ARCPixel(8))
        op = FillRectangleOp(top_left=top_left, bottom_right=bottom_right, color=color)
        expected_dict = {
            "operation_name": "FillRectangle",
            "top_left": {"row": 0, "col": 0, "is_relative": False},
            "bottom_right": {"row": 4, "col": 4, "is_relative": False},
            "color": {"value": 8}
        }
        self.assertEqual(op.to_dict(), expected_dict)

    def test_delete_object_op_to_dict(self):
        selector = DSLObjectSelector(criteria={"size": 10, "shape": "square"})
        op = DeleteObjectOp(selector=selector)
        expected_dict = {
            "operation_name": "DeleteObject",
            "selector": {"criteria": {"size": 10, "shape": "square"}, "select_all_matching": False}
        }
        self.assertEqual(op.to_dict(), expected_dict)

    def test_dsl_program_to_dict(self):
        op1 = ChangeColorOp(DSLObjectSelector(criteria={"id":"obj1"}), DSLColor(ARCPixel(2)))
        op2 = MoveOp(DSLObjectSelector(criteria={"id":"obj2"}), DSLPosition(1,1,True))
        program = DSLProgram(operations=[op1, op2])
        expected_dict = {
            "operations": [
                {
                    "operation_name": "ChangeColor",
                    "selector": {"criteria": {"id":"obj1"}, "select_all_matching": False},
                    "new_color": {"value": 2}
                },
                {
                    "operation_name": "Move",
                    "selector": {"criteria": {"id":"obj2"}, "select_all_matching": False},
                    "destination": {"row":1, "col":1, "is_relative":True}
                }
            ]
        }
        self.assertEqual(program.to_dict(), expected_dict)

class TestConditionalDsl(unittest.TestCase):
    def test_conditional_class_structure(self):
        self.assertTrue(issubclass(IfElseOp, BaseOperation))
        self.assertTrue(issubclass(PixelMatchesCondition, Condition))
        self.assertTrue(issubclass(ObjectExistsCondition, Condition))

    def test_pixel_matches_condition_to_dict(self):
        cond = PixelMatchesCondition(
            position=DSLPosition(1, 1),
            expected_color=DSLColor(ARCPixel(3))
        )
        expected = {
            "condition_type": "PixelMatchesCondition",
            "position": {"row": 1, "col": 1, "is_relative": False},
            "expected_color": {"value": 3}
        }
        self.assertEqual(cond.to_dict(), expected)

    def test_object_exists_condition_to_dict(self):
        cond = ObjectExistsCondition(
            selector=DSLObjectSelector(criteria={"color": ARCPixel(4)})
        )
        expected = {
            "condition_type": "ObjectExistsCondition",
            "selector": {"criteria": {"color": 4}, "select_all_matching": False}
        }
        self.assertEqual(cond.to_dict(), expected)

    def test_conditional_branch_to_dict(self):
        cond = ObjectExistsCondition(selector=DSLObjectSelector(criteria={"id": "A"}))
        prog_ops = [ChangeColorOp(DSLObjectSelector(criteria={"id":"A"}), DSLColor(ARCPixel(1)))]
        prog = DSLProgram(operations=prog_ops)
        branch = ConditionalBranch(condition=cond, program=prog)
        
        expected = {
            "condition": {
                "condition_type": "ObjectExistsCondition",
                "selector": {"criteria": {"id": "A"}, "select_all_matching": False}
            },
            "program": {
                "operations": [{
                    "operation_name": "ChangeColor",
                    "selector": {"criteria": {"id":"A"}, "select_all_matching": False},
                    "new_color": {"value": 1}
                }]
            }
        }
        self.assertEqual(branch.to_dict(), expected)

    def test_if_else_op_to_dict(self):
        # Main branch
        cond_main = PixelMatchesCondition(DSLPosition(0,0), DSLColor(ARCPixel(1)))
        prog_main_ops = [DeleteObjectOp(DSLObjectSelector(criteria={"id":"target"}))]
        prog_main = DSLProgram(operations=prog_main_ops)
        main_branch = ConditionalBranch(condition=cond_main, program=prog_main)

        # Else branch
        prog_else_ops = [CreateObjectOp(ARCGrid([[ARCPixel(2)]]), DSLPosition(1,1), DSLColor(ARCPixel(2)))]
        prog_else = DSLProgram(operations=prog_else_ops)

        op_if_else = IfElseOp(main_branch=main_branch, else_branch=prog_else)
        
        expected = {
            "operation_name": "IfElse",
            "main_branch": {
                "condition": {
                    "condition_type": "PixelMatchesCondition",
                    "position": {"row":0, "col":0, "is_relative":False},
                    "expected_color": {"value":1}
                },
                "program": {
                    "operations": [{
                        "operation_name": "DeleteObject",
                        "selector": {"criteria": {"id":"target"}, "select_all_matching":False}
                    }]
                }
            },
            "else_branch": {
                "operations": [{
                    "operation_name": "CreateObject",
                    "shape_data": [[2]],
                    "destination": {"row":1, "col":1, "is_relative":False},
                    "color": {"value":2}
                }]
            }
        }
        self.assertEqual(op_if_else.to_dict(), expected)

    def test_if_else_op_to_dict_no_else(self):
        cond_main = PixelMatchesCondition(DSLPosition(0,0), DSLColor(ARCPixel(1)))
        prog_main_ops = [DeleteObjectOp(DSLObjectSelector(criteria={"id":"target"}))]
        prog_main = DSLProgram(operations=prog_main_ops)
        main_branch = ConditionalBranch(condition=cond_main, program=prog_main)
        op_if_only = IfElseOp(main_branch=main_branch) # No else branch

        expected = {
            "operation_name": "IfElse",
            "main_branch": {
                "condition": {
                    "condition_type": "PixelMatchesCondition",
                    "position": {"row":0, "col":0, "is_relative":False},
                    "expected_color": {"value":1}
                },
                "program": {
                    "operations": [{
                        "operation_name": "DeleteObject",
                        "selector": {"criteria": {"id":"target"}, "select_all_matching":False}
                    }]
                }
            },
            "else_branch": None
        }
        self.assertEqual(op_if_only.to_dict(), expected)


if __name__ == '__main__':
    # This is so you can run the test script directly
    # You might need to adjust sys.path if ur_project is not in the Python path
    # For example, if your tests are in 'src/ur_project/core' and project root is 'src':
    # import sys
    # import os
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) # Adjust this path
    # sys.path.insert(0, project_root) 
    
    unittest.main()
