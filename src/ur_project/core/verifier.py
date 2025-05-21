# src/ur_project/core/verifier.py

from typing import Protocol, Optional, Dict, Any
import math # For comparing floats

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

# Example Usage:
# if __name__ == "__main__":
#     from ur_project.core.foundational_llm import BaseLLM, LLMResponse # For mock LLM
#     from ur_project.core.proposer import SimpleArithmeticProposer
#     from ur_project.core.solver import LLMSimpleArithmeticSolver

#     # Mock LLM (same as in solver example)
#     class MockLLM(BaseLLM):
#         def _load_model(self):
#             print("MockLLM loaded.")
#         def generate(self, prompt: str, **kwargs) -> LLMResponse:
#             if "5 + 3" in prompt: return LLMResponse("8.0")
#             if "10 / 4" in prompt: return LLMResponse("2.5")
#             if "7 * 6" in prompt: return LLMResponse("41.9999999999") # Test tolerance
#             if "1 / 3" in prompt: return LLMResponse("0.333") # Test tolerance
#             if "bad input" in prompt: return LLMResponse("I don\'t know")
#             return LLMResponse("NaN")
#         def batch_generate(self, prompts: list[str], **kwargs) -> list[LLMResponse]:
#             return [self.generate(p) for p in prompts]

#     mock_llm_instance = MockLLM(model_path_or_name="mock_model")
#     proposer = SimpleArithmeticProposer(max_number=10)
#     solver = LLMSimpleArithmeticSolver(llm=mock_llm_instance)
#     verifier = SimpleArithmeticVerifier()

#     test_cases = [
#         proposer.propose_task(), # Should be solvable
#         SimpleArithmeticTask("fixed_div_zero_test", 1, 0, '/'), # Proposer avoids this, but task can exist
#         SimpleArithmeticTask("fixed_float_test", 1, 3, '/'), # For 0.333... testing
#         SimpleArithmeticTask("fixed_text_output", 1, 1, '+') # Forcing LLM to give non-numeric
#     ]
#     test_cases[3].description = "bad input test" # To trigger specific mock LLM output

#     for task_to_solve in test_cases:
#         print(f"Task ID: {task_to_solve.id}, Description: {task_to_solve.description}")
#         try:
#             expected = task_to_solve.get_correct_answer()
#             print(f"  Expected by Verifier (from task): {expected}")
#         except ValueError as e:
#             print(f"  Expected by Verifier (from task): Error - {e}")
#             expected = None
        
#         solution = solver.solve_task(task_to_solve)
#         print(f"  Solver Raw: '{solution.raw_answer}', Parsed: {solution.parsed_answer}")
        
#         if isinstance(task_to_solve, SimpleArithmeticTask): # Verifier expects SimpleArithmeticTask
#             verification_result = verifier.verify_solution(task_to_solve, solution)
#             print(f"  Verification -> Correct: {verification_result.is_correct}, Actual: {verification_result.actual_solution}, Expected: {verification_result.expected_solution}")
#             if verification_result.metadata:
#                 print(f"  Verification Metadata: {verification_result.metadata}")
#         else:
#             print("  Skipping verification due to incompatible task type for SimpleArithmeticVerifier.")
#         print("---") 