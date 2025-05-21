# src/ur_project/core/verifier.py

import logging # Added logging
from typing import Protocol, Optional, Dict, Any, List # Added List for ARCGrid comparison
import math # For comparing floats

# Assuming ARCPuzzle and ARCGrid will be imported for type hinting if not already.
# For now, using them directly as they are defined in arc_types.py and solver.py uses them.
from ur_project.data_processing.arc_types import ARCPuzzle, ARCGrid 
from ur_project.core.proposer import Task, SimpleArithmeticTask # Using Task protocol and specific task type
from ur_project.core.solver import Solution

class VerificationResult:
    def __init__(self, task_id: str, is_correct: bool, actual_solution: Any, expected_solution: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.is_correct = is_correct
        self.actual_solution = actual_solution # What the solver produced (parsed)
        self.expected_solution = expected_solution # The ground truth, if available
        self.metadata = metadata if metadata is not None else {}

class BaseVerifier(Protocol):
    def verify_solution(self, task: Task, solution: Solution) -> VerificationResult:
        ...

class SimpleArithmeticVerifier(BaseVerifier):
    """Verifies solutions for SimpleArithmeticTasks."""
    def __init__(self, float_tolerance: float = 1e-9):
        self.float_tolerance = float_tolerance

    def verify_solution(self, task: SimpleArithmeticTask, solution: Solution) -> VerificationResult:
        if not isinstance(task, SimpleArithmeticTask):
            return VerificationResult(
                task_id=task.id,
                is_correct=False,
                actual_solution=solution.parsed_answer,
                metadata={"error": "Verifier received incompatible task type."}
            )
        
        if solution.parsed_answer is None:
            return VerificationResult(
                task_id=task.id,
                is_correct=False,
                actual_solution=solution.raw_answer, # Show raw if parsed is None
                expected_solution=task.get_correct_answer(),
                metadata={"error": "Solution could not be parsed."}
            )

        expected_answer = task.get_correct_answer()
        
        is_correct = False
        # For floats, compare with tolerance
        if isinstance(solution.parsed_answer, (float, int)) and isinstance(expected_answer, (float, int)):
            if math.isclose(float(solution.parsed_answer), float(expected_answer), rel_tol=self.float_tolerance):
                is_correct = True
        # Could add checks for other types if tasks evolved

        return VerificationResult(
            task_id=task.id,
            is_correct=is_correct,
            actual_solution=solution.parsed_answer,
            expected_solution=expected_answer
        )

class ARCVerifier(BaseVerifier):
    """Verifies solutions for ARCPuzzles by comparing output grids."""
    task_type_expected = ARCPuzzle # For AZLoop compatibility check

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _compare_grids(self, grid1: Optional[ARCGrid], grid2: Optional[ARCGrid]) -> bool:
        """Compares two ARCGrids for equality."""
        if grid1 is None and grid2 is None:
            return True # Both are None, consider them "equal" in context of no solution vs no expectation
        if grid1 is None or grid2 is None:
            return False # One is None, the other isn't
        
        if len(grid1) != len(grid2):
            return False
        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        return True

    def verify_solution(self, task: ARCPuzzle, solution: Solution) -> VerificationResult:
        if not isinstance(task, ARCPuzzle):
            self.logger.error(f"ARCVerifier received incompatible task type '{type(task).__name__}' for task ID '{task.id}'. Expected ARCPuzzle.")
            return VerificationResult(
                task_id=task.id,
                is_correct=False,
                actual_solution=solution.parsed_answer,
                metadata={"error": f"ARCVerifier received incompatible task type: {type(task).__name__}"}
            )

        self.logger.info(f"Verifying solution for ARCPuzzle ID: {task.id}, Solver ID: {solution.solved_by}")

        # Log DSL execution journey
        self.logger.info(f"  Initial DSL Program: {solution.raw_dsl_program}")
        if solution.dsl_parse_error:
            self.logger.warning(f"  Initial DSL Parsing Failed: {solution.dsl_parse_error}")
        else:
            self.logger.info("  Initial DSL Parsing: OK")
            if solution.dsl_execution_error:
                self.logger.warning(f"  Initial DSL Execution Failed: {solution.dsl_execution_error}")
            elif solution.parsed_dsl_program: # Parsed OK, check if executed
                 self.logger.info(f"  Initial DSL Execution: OK (Produced grid: {solution.executed_grid is not None})")
            else: # Should not happen if parse_error is None, but defensive
                 self.logger.info("  Initial DSL Execution: Not run (e.g. parse error or no program)")


        if solution.retry_attempted:
            self.logger.info("  Retry Attempted:")
            self.logger.info(f"    Retry DSL Program: {solution.raw_dsl_program_retry}")
            if solution.dsl_parse_error_retry:
                self.logger.warning(f"    Retry DSL Parsing Failed: {solution.dsl_parse_error_retry}")
            else:
                self.logger.info("    Retry DSL Parsing: OK")
                if solution.dsl_execution_error_retry:
                    self.logger.warning(f"    Retry DSL Execution Failed: {solution.dsl_execution_error_retry}")
                elif solution.parsed_dsl_program_retry: # Parsed OK
                    self.logger.info(f"    Retry DSL Execution: OK (Produced grid: {solution.executed_grid_retry is not None})")
                else:
                    self.logger.info("    Retry DSL Execution: Not run (e.g. parse error or no program)")
        
        # Verification based on the final grid in solution.parsed_answer
        # ARCSolver sets solution.parsed_answer to the grid from the successful attempt (initial or retry)
        produced_grid = solution.parsed_answer
        expected_grid = task.expected_output_grid
        
        is_correct = self._compare_grids(produced_grid, expected_grid)
        
        if produced_grid is None and not (solution.dsl_parse_error or solution.dsl_execution_error or solution.dsl_parse_error_retry or solution.dsl_execution_error_retry):
            self.logger.info(f"  Verification for {task.id}: Solver produced no grid, but no DSL errors were reported. Assuming this means no solution found.")
        elif produced_grid is None:
             self.logger.info(f"  Verification for {task.id}: Solver produced no grid due to DSL errors.")


        self.logger.info(f"  Verification for {task.id}: Correct: {is_correct}")
        if not is_correct:
            self.logger.info(f"    Produced Grid: {produced_grid}")
            self.logger.info(f"    Expected Grid: {expected_grid}")

        # Populate metadata for VerificationResult
        verification_metadata: Dict[str, Any] = {
            "dsl_executed_successfully": produced_grid is not None,
            "initial_dsl_program": solution.raw_dsl_program,
            "retry_attempted": solution.retry_attempted,
        }
        final_dsl_error_summary = "OK"
        if solution.dsl_parse_error:
            verification_metadata["initial_parse_error"] = solution.dsl_parse_error
            final_dsl_error_summary = f"Initial parse error: {solution.dsl_parse_error}"
        if solution.dsl_execution_error:
            verification_metadata["initial_execution_error"] = solution.dsl_execution_error
            if final_dsl_error_summary == "OK": final_dsl_error_summary = f"Initial execution error: {solution.dsl_execution_error}"
        
        if solution.retry_attempted:
            verification_metadata["retry_dsl_program"] = solution.raw_dsl_program_retry
            if solution.dsl_parse_error_retry:
                verification_metadata["retry_parse_error"] = solution.dsl_parse_error_retry
                if final_dsl_error_summary == "OK" or "Initial" in final_dsl_error_summary: # If initial was ok or error, now retry failed
                    final_dsl_error_summary = f"Retry parse error: {solution.dsl_parse_error_retry}"
            if solution.dsl_execution_error_retry:
                verification_metadata["retry_execution_error"] = solution.dsl_execution_error_retry
                if final_dsl_error_summary == "OK" or "Initial" in final_dsl_error_summary:
                     final_dsl_error_summary = f"Retry execution error: {solution.dsl_execution_error_retry}"
        
        verification_metadata["final_dsl_error_summary"] = final_dsl_error_summary
        if produced_grid is None and final_dsl_error_summary == "OK":
            # This case means DSL was parsed and (presumably) executed without error, but produced None.
            # This might indicate a valid "no-op" or an issue in DSL command logic not raising error.
            verification_metadata["final_dsl_error_summary"] = "DSL processed OK, but no output grid was generated."


        return VerificationResult(
            task_id=task.id,
            is_correct=is_correct,
            actual_solution=produced_grid,
            expected_solution=expected_grid,
            metadata=verification_metadata
        )

# Example Usage (Conceptual - requires ARCPuzzle and Solution instances)
# if __name__ == "__main__":
#     # This example would require setting up mock ARCPuzzle and Solution objects
#     # with various DSL metadata states to test the ARCVerifier logging and metadata population.
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
#     logger.addHandler(stream_handler)

#     verifier = ARCVerifier()
    
#     # Test Case 1: Perfect match
#     mock_task1 = ARCPuzzle(id="arc_task_001", description="Test 1", data=[[1]], expected_output_grid=[[2]], source_task_id="src_001", source_pair_id="test_0")
#     mock_solution1 = Solution(task_id="arc_task_001", solved_by="MockSolver", raw_answer="DSL...", parsed_answer=[[2]])
#     mock_solution1.raw_dsl_program = "SOME_DSL_COMMAND_OK"
#     result1 = verifier.verify_solution(mock_task1, mock_solution1)
#     print(f"Result 1 Correct: {result1.is_correct}, Metadata: {result1.metadata}")

#     # Test Case 2: Mismatch
#     mock_task2 = ARCPuzzle(id="arc_task_002", description="Test 2", data=[[1]], expected_output_grid=[[3]], source_task_id="src_002", source_pair_id="test_0")
#     mock_solution2 = Solution(task_id="arc_task_002", solved_by="MockSolver", raw_answer="DSL...", parsed_answer=[[2]])
#     mock_solution2.raw_dsl_program = "SOME_DSL_COMMAND_WRONG_GRID"
#     result2 = verifier.verify_solution(mock_task2, mock_solution2)
#     print(f"Result 2 Correct: {result2.is_correct}, Metadata: {result2.metadata}")

#     # Test Case 3: DSL execution error, no grid produced by solver
#     mock_task3 = ARCPuzzle(id="arc_task_003", description="Test 3", data=[[1]], expected_output_grid=[[2]], source_task_id="src_003", source_pair_id="test_0")
#     mock_solution3 = Solution(task_id="arc_task_003", solved_by="MockSolver", raw_answer="DSL...", parsed_answer=None)
#     mock_solution3.raw_dsl_program = "INVALID_DSL_COMMAND"
#     mock_solution3.dsl_parse_error = "Unknown command: INVALID_DSL_COMMAND"
#     result3 = verifier.verify_solution(mock_task3, mock_solution3)
#     print(f"Result 3 Correct: {result3.is_correct}, Metadata: {result3.metadata}")

#     # Test Case 4: Retry attempt, successful
#     mock_task4 = ARCPuzzle(id="arc_task_004", description="Test 4", data=[[1]], expected_output_grid=[[5]], source_task_id="src_004", source_pair_id="test_0")
#     mock_solution4 = Solution(task_id="arc_task_004", solved_by="MockSolver", raw_answer="Retry response...", parsed_answer=[[5]]) # Final answer from retry
#     mock_solution4.raw_dsl_program = "INITIAL_BAD_DSL"
#     mock_solution4.dsl_execution_error = "Initial execution failed spectacularly"
#     mock_solution4.retry_attempted = True
#     mock_solution4.raw_dsl_program_retry = "RETRY_GOOD_DSL"
#     mock_solution4.executed_grid_retry = [[5]] # This is what parsed_answer would be from solver
#     result4 = verifier.verify_solution(mock_task4, mock_solution4)
#     print(f"Result 4 Correct: {result4.is_correct}, Metadata: {result4.metadata}")

#     # Test Case 5: Retry attempt, but retry also fails
#     mock_task5 = ARCPuzzle(id="arc_task_005", description="Test 5", data=[[1]], expected_output_grid=[[5]], source_task_id="src_005", source_pair_id="test_0")
#     mock_solution5 = Solution(task_id="arc_task_005", solved_by="MockSolver", raw_answer="Retry response failed...", parsed_answer=None) # No final answer
#     mock_solution5.raw_dsl_program = "INITIAL_BAD_DSL"
#     mock_solution5.dsl_parse_error = "Initial parse error"
#     mock_solution5.retry_attempted = True
#     mock_solution5.raw_dsl_program_retry = "RETRY_ALSO_BAD_DSL"
#     mock_solution5.dsl_execution_error_retry = "Retry execution also failed"
#     result5 = verifier.verify_solution(mock_task5, mock_solution5)
#     print(f"Result 5 Correct: {result5.is_correct}, Metadata: {result5.metadata}")