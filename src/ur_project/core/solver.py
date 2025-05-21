from typing import Any, Dict, Protocol, Optional
from ur_project.core.foundational_llm import BaseLLM, LLMResponse
from ur_project.core.proposer import Task # Using the Task protocol
from .arc_dsl import DSLProgram # Import DSLProgram for type hinting

class Solution:
    def __init__(self, task_id: str, solved_by: str, raw_answer: Any, parsed_answer: Any = None, metadata: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.solved_by = solved_by # Identifier for the solver or model version
        self.raw_answer = raw_answer # The direct output from the LLM or solving process
        self.parsed_answer = parsed_answer # Potentially cleaned/structured version of the answer
        self.hypothesized_transformations: Optional[str] = None # For LLM-generated hypotheses
        self.raw_dsl_program: Optional[str] = None # For LLM-generated DSL program string
        self.parsed_dsl_program: Optional[DSLProgram] = None # For the parsed DSLProgram object
        self.metadata = metadata if metadata is not None else {}

class BaseSolver(Protocol):
    def solve_task(self, task: Task) -> Solution:
        ...

class LLMSimpleArithmeticSolver(BaseSolver):
    """Solves simple arithmetic tasks using a foundational LLM."""
    def __init__(self, llm: BaseLLM, solver_id: str = "LLMSimpleArithmeticSolver_v1"):
        self.llm = llm
        self.solver_id = solver_id

    def solve_task(self, task: Task) -> Solution:
        if task.expected_solution_type != "float": # From SimpleArithmeticTask
            # This solver is specialized, could raise error or return a specific "cannot solve" Solution
            return Solution(
                task_id=task.id,
                solved_by=self.solver_id,
                raw_answer="Task type not supported by this solver.",
                metadata={"error": "Unsupported task type"}
            )

        # Simple prompt engineering for arithmetic
        # For a task like "Calculate: 5 + 3", the task.description is already good.
        # We might want to add instructions for the output format.
        prompt = f"Solve the following arithmetic problem. Provide only the numerical answer.\nProblem: {task.description}"
        # An alternative could be to directly use task.data if the LLM is fine-tuned for structured input:
        # prompt = f"num1: {task.data['num1']}, operator: {task.data['operator']}, num2: {task.data['num2']}. Result: " 

        llm_response: LLMResponse = self.llm.generate(
            prompt,
            max_new_tokens=16, # Expecting short numerical answers
            temperature=0.0 # For deterministic arithmetic if possible
        )

        raw_answer_text = llm_response.text.strip()
        parsed_answer = None
        try:
            # Attempt to parse the LLM's answer as a float
            parsed_answer = float(raw_answer_text)
        except ValueError:
            # LLM might have produced non-numeric output
            print(f"Warning: Could not parse LLM output '{raw_answer_text}' as float for task {task.id}")
            # Keep raw_answer_text as is, parsed_answer remains None

        return Solution(
            task_id=task.id,
            solved_by=self.solver_id,
            raw_answer=raw_answer_text,
            parsed_answer=parsed_answer,
            metadata=llm_response.metadata
        )

# Example Usage (requires a BaseLLM implementation and instance):
# if __name__ == "__main__":
#     # This is a mock LLM for testing purposes
#     class MockLLM(BaseLLM):
#         def _load_model(self):
#             print("MockLLM loaded.")

#         def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, **kwargs) -> LLMResponse:
#             print(f"MockLLM received prompt: {prompt}")
#             if "5 + 3" in prompt:
#                 return LLMResponse("8")
#             elif "10 / 2" in prompt:
#                 return LLMResponse("5.0")
#             elif "7 * 6" in prompt:
#                 return LLMResponse("42")
#             return LLMResponse("NaN") # Default unknown answer
        
#         def batch_generate(self, prompts: list[str], **kwargs) -> list[LLMResponse]:
#             return [self.generate(p) for p in prompts]

#     mock_llm_instance = MockLLM(model_path_or_name="mock_model")
#     arithmetic_solver = LLMSimpleArithmeticSolver(llm=mock_llm_instance)
#     proposer = SimpleArithmeticProposer(max_number=10)

#     for _ in range(3):
#         task_to_solve = proposer.propose_task()
#         print(f"Attempting to solve Task ID: {task_to_solve.id}, Description: {task_to_solve.description}")
#         solution = arithmetic_solver.solve_task(task_to_solve)
#         print(f"  Solver ID: {solution.solved_by}")
#         print(f"  Raw Answer: '{solution.raw_answer}'")
#         print(f"  Parsed Answer: {solution.parsed_answer}")
#         # In a real loop, this solution would go to the Verifier
#         print("---") 