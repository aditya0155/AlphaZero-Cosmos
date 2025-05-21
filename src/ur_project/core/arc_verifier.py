from ur_project.core.verifier import BaseVerifier, VerificationResult
from ur_project.core.solver import Solution # To type hint solution parameter
from ur_project.data_processing.arc_types import ARCPuzzle, ARCGrid # ARCPuzzle is the Task

class ARCVerifier(BaseVerifier):
    """Verifies solutions for ARCPuzzles by comparing output grids."""
    
    def verify_solution(self, task: ARCPuzzle, solution: Solution) -> VerificationResult:
        if not isinstance(task, ARCPuzzle):
            return VerificationResult(
                task_id=task.id,
                is_correct=False,
                actual_solution=solution.parsed_answer,
                expected_solution=None, # Can't know expected if task type is wrong
                metadata={"error": "ARCVerifier received incompatible task type."}
            )

        if solution.parsed_answer is None or not isinstance(solution.parsed_answer, list):
            # This implies the ARCSolver could not parse the LLM output into a grid
            return VerificationResult(
                task_id=task.id,
                is_correct=False,
                actual_solution=solution.raw_answer, # Show raw if parsed is None or not a grid
                expected_solution=task.expected_output_grid,
                metadata={"error": "Solution was not a valid grid or could not be parsed."}
            )
        
        # Ensure the parsed answer is actually an ARCGrid (list of lists of ints)
        # This should ideally be guaranteed by the ARCSolver's parsing, but good to double check.
        # For simplicity, we assume ARCSolver produces a valid ARCGrid structure if not None.
        solver_grid: ARCGrid = solution.parsed_answer
        expected_grid: ARCGrid = task.expected_output_grid

        is_correct = False
        metadata = {}

        # Copy relevant DSL execution metadata from solution, if present
        if solution.metadata:
            dsl_status = solution.metadata.get("dsl_execution_status")
            if dsl_status:
                metadata["dsl_execution_status"] = dsl_status
                if "dsl_operations_executed" in solution.metadata:
                    metadata["dsl_operations_executed"] = solution.metadata["dsl_operations_executed"]
                if "dsl_execution_error_message" in solution.metadata:
                    metadata["dsl_execution_error_message"] = solution.metadata["dsl_execution_error_message"]

        # Compare dimensions first
        if len(solver_grid) != len(expected_grid):
            is_correct = False
            metadata["error"] = f"Dimension mismatch: Solver rows {len(solver_grid)}, Expected rows {len(expected_grid)}."
        elif solver_grid and expected_grid and (len(solver_grid[0]) != len(expected_grid[0])):
            is_correct = False
            metadata["error"] = f"Dimension mismatch: Solver cols {len(solver_grid[0]) if solver_grid else 0}, Expected cols {len(expected_grid[0]) if expected_grid else 0}."
        else:
            # Element-wise comparison
            match = True
            for r in range(len(expected_grid)):
                for c in range(len(expected_grid[0])):
                    if solver_grid[r][c] != expected_grid[r][c]:
                        match = False
                        metadata["first_mismatch_at"] = (r, c)
                        metadata["mismatch_values"] = (solver_grid[r][c], expected_grid[r][c])
                        break
                if not match:
                    break
            is_correct = match
            if not is_correct and not metadata.get("error"):
                 metadata["error"] = "Grid content mismatch."
        
        return VerificationResult(
            task_id=task.id,
            is_correct=is_correct,
            actual_solution=solver_grid, # The parsed grid from solver
            expected_solution=expected_grid,
            metadata=metadata
        )

# Example Usage:
# if __name__ == '__main__':
#     from ur_project.data_processing.arc_types import ARCPixel

#     # Dummy task and solutions for testing
#     sample_input_grid: ARCGrid = [[ARCPixel(1)]]
#     correct_output_grid: ARCGrid = [[ARCPixel(2), ARCPixel(2)], [ARCPixel(2), ARCPixel(2)]]
#     task_instance = ARCPuzzle(
#         id="test_arc_puzzle_01",
#         description="A test puzzle",
#         data=sample_input_grid,
#         expected_output_grid=correct_output_grid,
#         source_task_id="dummy_task",
#         source_pair_id="train_0"
#     )

#     verifier = ARCVerifier()

#     # Case 1: Correct solution
#     sol_correct = Solution(task_id=task_instance.id, solved_by="test_solver", raw_answer="[[2,2],[2,2]]", parsed_answer=correct_output_grid)
#     res1 = verifier.verify_solution(task_instance, sol_correct)
#     print(f"Result 1 (Correct): IsCorrect={res1.is_correct}, Metadata={res1.metadata}")

#     # Case 2: Incorrect solution (content mismatch)
#     incorrect_grid_content: ARCGrid = [[ARCPixel(3), ARCPixel(2)], [ARCPixel(2), ARCPixel(2)]]
#     sol_incorrect_content = Solution(task_id=task_instance.id, solved_by="test_solver", raw_answer="[[3,2],[2,2]]", parsed_answer=incorrect_grid_content)
#     res2 = verifier.verify_solution(task_instance, sol_incorrect_content)
#     print(f"Result 2 (Incorrect Content): IsCorrect={res2.is_correct}, Metadata={res2.metadata}")

#     # Case 3: Incorrect solution (dimension mismatch - rows)
#     incorrect_grid_rows: ARCGrid = [[ARCPixel(2), ARCPixel(2)]]
#     sol_incorrect_rows = Solution(task_id=task_instance.id, solved_by="test_solver", raw_answer="[[2,2]]", parsed_answer=incorrect_grid_rows)
#     res3 = verifier.verify_solution(task_instance, sol_incorrect_rows)
#     print(f"Result 3 (Incorrect Rows): IsCorrect={res3.is_correct}, Metadata={res3.metadata}")

#     # Case 4: Incorrect solution (dimension mismatch - cols)
#     incorrect_grid_cols: ARCGrid = [[ARCPixel(2)], [ARCPixel(2)]]
#     sol_incorrect_cols = Solution(task_id=task_instance.id, solved_by="test_solver", raw_answer="[[2],[2]]", parsed_answer=incorrect_grid_cols)
#     res4 = verifier.verify_solution(task_instance, sol_incorrect_cols)
#     print(f"Result 4 (Incorrect Cols): IsCorrect={res4.is_correct}, Metadata={res4.metadata}")

#     # Case 5: Solution not parsed (parsed_answer is None)
#     sol_not_parsed = Solution(task_id=task_instance.id, solved_by="test_solver", raw_answer="Could not parse LLM output", parsed_answer=None)
#     res5 = verifier.verify_solution(task_instance, sol_not_parsed)
#     print(f"Result 5 (Not Parsed): IsCorrect={res5.is_correct}, Metadata={res5.metadata}")

#     # Case 6: Empty grid solution for non-empty target
#     empty_solver_grid: ARCGrid = []
#     sol_empty = Solution(task_id=task_instance.id, solved_by="test_solver", raw_answer="[]", parsed_answer=empty_solver_grid)
#     res6 = verifier.verify_solution(task_instance, sol_empty)
#     print(f"Result 6 (Empty Solver Grid): IsCorrect={res6.is_correct}, Metadata={res6.metadata}")

#     # Case 7: Correct empty grid for empty target
#     empty_target_task = ARCPuzzle(
#         id="test_arc_puzzle_empty", description="Empty target", data=[[]],
#         expected_output_grid=[], source_task_id="dummy_empty", source_pair_id="train_0"
#     )
#     sol_correct_empty = Solution(task_id=empty_target_task.id, solved_by="test", raw_answer="[]", parsed_answer=[])
#     res7 = verifier.verify_solution(empty_target_task, sol_correct_empty)
#     print(f"Result 7 (Correct Empty Grid): IsCorrect={res7.is_correct}, Metadata={res7.metadata}") 