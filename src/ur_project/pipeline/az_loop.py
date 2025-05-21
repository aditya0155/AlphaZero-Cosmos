# src/ur_project/pipeline/az_loop.py

import time
import logging
from typing import List, Dict, Any, Optional

from ur_project.core.proposer import BaseProposer, Task
from ur_project.core.solver import BaseSolver, Solution
from ur_project.core.verifier import BaseVerifier, VerificationResult
from ur_project.core.reward_model import BaseRewardModel, RewardSignal
from ur_project.core.evaluator import BaseEvaluator, QualitativeFeedback, LLMQualitativeEvaluator # New import
# For initial testing with concrete implementations:
from ur_project.core.proposer import SimpleArithmeticProposer
from ur_project.core.solver import LLMSimpleArithmeticSolver
from ur_project.core.verifier import SimpleArithmeticVerifier
from ur_project.core.reward_model import SimpleBinaryRewardModel, ARCRewardModel # Added ARCRewardModel
from ur_project.core.foundational_llm import BaseLLM, LLMResponse # For MockLLM

# ARC specific components for example
from ur_project.core.arc_proposer import ARCProposer
from ur_project.core.arc_solver import ARCSolver
from ur_project.core.arc_verifier import ARCVerifier
from ur_project.data_processing.arc_types import ARCGrid, ARCPixel # For mock ARC LLM
from ur_project.core.perception import BasicARCFeatureExtractor # For ARCSolver

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AZLoop:
    """Orchestrates the Absolute Zero loop: Propose -> Solve -> Verify -> Evaluate -> Reward."""
    def __init__(
        self,
        proposer: BaseProposer,
        solver: BaseSolver,
        verifier: BaseVerifier,
        reward_model: BaseRewardModel,
        evaluator: Optional[BaseEvaluator] = None, # Added evaluator
        # Optional: experiment_tracker: BaseExperimentTracker (for future W&B/MLflow integration)
    ):
        self.proposer = proposer
        self.solver = solver
        self.verifier = verifier
        self.reward_model = reward_model
        self.evaluator = evaluator # Store evaluator instance
        # self.experiment_tracker = experiment_tracker

        self.history: List[Dict[str, Any]] = [] # To store records of each step in the loop

    def run_step(self, step_number: int) -> Dict[str, Any]:
        """Runs a single step of the AZ loop."""
        logging.info(f"--- AZ Loop Step {step_number} Starting ---")
        
        step_data = {
            "step_number": step_number,
            "task_id": None,
            "task_description": None,
            "task_data": None,
            "task_quality_assessment": None,
            "task_quality_reasoning": None,
            "solution_raw": None,
            "solution_parsed": None,
            "solver_id": None,
            "solution_plausibility_assessment": None,
            "solution_plausibility_reasoning": None,
            "verification_correct": None,
            "verification_actual": None,
            "verification_expected": None,
            "verification_metadata": None,
            "reward_value": None,
            "reward_reason": None,
            "reward_metadata": {},
            "timings": {}
        }

        # 1. Propose Task
        start_time = time.time()
        task = self.proposer.propose_task()
        propose_time = time.time() - start_time
        step_data["timings"]["propose"] = propose_time
        step_data["task_id"] = task.id
        step_data["task_description"] = task.description
        step_data["task_data"] = task.data
        logging.info(f"Step {step_number} [Propose]: Task ID {task.id} - '{task.description}' (took {propose_time:.3f}s)")

        # 1b. Assess Task Quality (Optional)
        task_quality_feedback: Optional[QualitativeFeedback] = None
        if self.evaluator:
            start_time = time.time()
            task_quality_feedback = self.evaluator.assess_task_quality(task)
            eval_task_time = time.time() - start_time
            step_data["timings"]["evaluate_task"] = eval_task_time
            if task_quality_feedback:
                logging.info(f"Step {step_number} [Evaluate Task]: Task ID {task.id} - Assessment: '{task_quality_feedback.assessment}', Reasoning: '{task_quality_feedback.reasoning}' (took {eval_task_time:.3f}s)")
                step_data["task_quality_assessment"] = task_quality_feedback.assessment
                step_data["task_quality_reasoning"] = task_quality_feedback.reasoning
                step_data["reward_metadata"].update({"task_quality_feedback": task_quality_feedback})
            else:
                logging.info(f"Step {step_number} [Evaluate Task]: Task ID {task.id} - No feedback provided.")

        # 2. Solve Task
        start_time = time.time()
        solution = self.solver.solve_task(task)
        solve_time = time.time() - start_time
        step_data["timings"]["solve"] = solve_time
        step_data["solution_raw"] = solution.raw_answer
        step_data["solution_parsed"] = solution.parsed_answer
        step_data["solver_id"] = solution.solved_by
        
        solve_log_message = f"Step {step_number} [Solve]: Task ID {task.id} - Solver {solution.solved_by} -> Raw: '{str(solution.raw_answer)[:100]}...', Parsed: {solution.parsed_answer} (took {solve_time:.3f}s)"
        if solution.metadata and "dsl_execution_status" in solution.metadata:
            solve_log_message += f" | DSL Status: {solution.metadata['dsl_execution_status']}"
            if "dsl_operations_executed" in solution.metadata:
                 solve_log_message += f" (Ops: {solution.metadata['dsl_operations_executed']})"
        logging.info(solve_log_message)

        # 2b. Assess Solution Plausibility (Optional)
        solution_plausibility_feedback: Optional[QualitativeFeedback] = None
        if self.evaluator:
            start_time = time.time()
            solution_plausibility_feedback = self.evaluator.assess_solution_plausibility(task, solution)
            eval_sol_time = time.time() - start_time
            step_data["timings"]["evaluate_solution"] = eval_sol_time
            if solution_plausibility_feedback:
                logging.info(f"Step {step_number} [Evaluate Solution]: Task ID {task.id} - Assessment: '{solution_plausibility_feedback.assessment}', Reasoning: '{solution_plausibility_feedback.reasoning}' (took {eval_sol_time:.3f}s)")
                step_data["solution_plausibility_assessment"] = solution_plausibility_feedback.assessment
                step_data["solution_plausibility_reasoning"] = solution_plausibility_feedback.reasoning
                step_data["reward_metadata"].update({"solution_plausibility_feedback": solution_plausibility_feedback})
            else:
                logging.info(f"Step {step_number} [Evaluate Solution]: Task ID {task.id} - No feedback provided.")

        # 3. Verify Solution
        start_time = time.time()
        if isinstance(task, getattr(self.verifier, 'task_type_expected', object)):
            verification_result = self.verifier.verify_solution(task, solution)
        else:
            logging.warning(f"Step {step_number} [Verify]: Verifier {type(self.verifier).__name__} may not be compatible with task type {type(task).__name__}.")
            try:
                verification_result = self.verifier.verify_solution(task, solution)
            except Exception as e:
                 logging.error(f"Step {step_number} [Verify]: Error during verification for task {task.id}: {e}")
                 verification_result = VerificationResult(
                    task_id=task.id, is_correct=False, actual_solution=solution.parsed_answer,
                    metadata={"error": "Verification failed due to verifier incompatibility or error", "exception": str(e)}
                )
        verify_time = time.time() - start_time
        step_data["timings"]["verify"] = verify_time
        step_data["verification_correct"] = verification_result.is_correct
        step_data["verification_actual"] = verification_result.actual_solution
        step_data["verification_expected"] = verification_result.expected_solution
        step_data["verification_metadata"] = verification_result.metadata
        logging.info(f"Step {step_number} [Verify]: Task ID {task.id} - Correct: {verification_result.is_correct}, Actual: {verification_result.actual_solution}, Expected: {verification_result.expected_solution} (took {verify_time:.3f}s)")
        if verification_result.metadata:
            logging.info(f"Step {step_number} [Verify]: Metadata: {verification_result.metadata}")

        # 4. Calculate Reward
        start_time = time.time()
        reward_signal = self.reward_model.calculate_reward(verification_result)
        reward_time = time.time() - start_time
        step_data["timings"]["reward"] = reward_time
        step_data["reward_value"] = reward_signal.reward_value
        step_data["reward_reason"] = reward_signal.reason
        # Ensure qualitative feedback (if any) is part of the reward metadata if not already added
        if "task_quality_feedback" not in step_data["reward_metadata"] and task_quality_feedback:
            step_data["reward_metadata"].update({"task_quality_feedback": task_quality_feedback})
        if "solution_plausibility_feedback" not in step_data["reward_metadata"] and solution_plausibility_feedback:
            step_data["reward_metadata"].update({"solution_plausibility_feedback": solution_plausibility_feedback})
        # Merge any existing reward signal metadata
        if reward_signal.metadata:
            step_data["reward_metadata"].update(reward_signal.metadata)

        logging.info(f"Step {step_number} [Reward]: Task ID {task.id} - Reward: {reward_signal.reward_value}, Reason: '{reward_signal.reason}' (took {reward_time:.3f}s)")
        if step_data["reward_metadata"]:
            logging.info(f"Step {step_number} [Reward]: Metadata: {step_data['reward_metadata']}")
        
        self.history.append(step_data)
        logging.info(f"--- AZ Loop Step {step_number} Finished ---")
        return step_data

    def run_loop(self, num_steps: int):
        """Runs the AZ loop for a specified number of steps."""
        logging.info(f"Starting AZ Loop for {num_steps} steps.")
        for i in range(num_steps):
            self.run_step(step_number=i + 1)
            # Small break to make logs more readable if running many steps quickly
            if num_steps > 10 and i < num_steps -1 :
                time.sleep(0.1)
        logging.info(f"AZ Loop finished after {num_steps} steps.")

# --- Example Usage ---
class MockLLMForLoop(BaseLLM):
    def __init__(self, model_path_or_name: str = "mock_solver_llm", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_path_or_name, config)
        self.eval_count = 0
    def _load_model(self):
        logging.info(f"MockLLMForLoop ({self.model_path_or_name}): Model loaded (mock).")
    def generate(self, prompt: str, max_new_tokens: int = 16, temperature: float = 0.0, **kwargs) -> LLMResponse:
        logging.info(f"MockLLMForLoop for {self.model_path_or_name} received prompt (first 100 chars): {prompt[:100]}...")
        if "Is this task well-formed" in prompt:
            return LLMResponse("Assessment: Well-formed (Mock) Justification: The task seems reasonable for mock evaluation.")
        if "Does this solution seem plausible" in prompt:
            return LLMResponse("Assessment: Plausible (Mock) Justification: The mock solution appears adequate.")
        
        # --- Mock LLM Logic for ARC Solver ---
        if "Input grid (data): [[1]]" in prompt or "[[1]]" in prompt and "ARC puzzle" in prompt : # Simplified check for ARC prompt
            # Mock output for a simple ARC task where input is [[1]]
            logging.info("MockLLMForLoop: Detected ARC prompt for [[1]], returning [[2]].")
            return LLMResponse(text="[[2]]") 
        elif "Input grid (data): [[1,0],[0,1]]" in prompt or "[[1, 0], [0, 1]]" in prompt and "ARC puzzle" in prompt:
            logging.info("MockLLMForLoop: Detected ARC prompt for [[1,0],[0,1]], returning [[0,1],[1,0]].")
            return LLMResponse(text="[[0, 1], [1, 0]]")
        # --- End Mock LLM Logic for ARC Solver ---

        # Fallback to arithmetic if not an ARC or evaluator prompt
        try:
            problem_line = prompt.split("Problem: Calculate: ")[-1]
            parts = problem_line.split()
            num1 = int(parts[0]); op = parts[1]; num2 = int(parts[2])
            result = "NaN"
            if op == '+': result = str(num1 + num2)
            elif op == '-': result = str(num1 - num2)
            elif op == '*': result = str(num1 * num2)
            elif op == '/': result = str(float(num1) / num2 if num2 != 0 else "Infinity")
            self.eval_count += 1
            if self.eval_count % 5 == 0: return LLMResponse(text="I am not sure about that.")
            if self.eval_count % 7 == 0 and op == '*': return LLMResponse(text=str(int(result) + random.randint(1,5))) # type: ignore
            return LLMResponse(text=result)
        except Exception: return LLMResponse(text="Error in mock arithmetic processing")
    def batch_generate(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        return [self.generate(p, **kwargs) for p in prompts]

if __name__ == "__main__":
    import random 
    import os # For ARC example path
    import json # For creating dummy ARC files
    import shutil # For cleaning up dummy ARC files

    # --- Configuration for the example run ---
    EXAMPLE_MODE = "ARC" # "ARITHMETIC" or "ARC"
    NUM_EXAMPLE_STEPS = 2
    # ---

    logging.info(f"Setting up AZ Loop for example run (Mode: {EXAMPLE_MODE})...")

    if EXAMPLE_MODE == "ARITHMETIC":
        proposer_impl = SimpleArithmeticProposer(max_number=10)
        mock_solver_llm = MockLLMForLoop(model_path_or_name="mock_llm_for_arith_solver")
        solver_impl = LLMSimpleArithmeticSolver(llm=mock_solver_llm)
        verifier_impl = SimpleArithmeticVerifier()
        # setattr(verifier_impl, 'task_type_expected', SimpleArithmeticTask) # No longer needed due to isinstance check
        reward_model_impl = SimpleBinaryRewardModel()
        mock_evaluator_llm = MockLLMForLoop(model_path_or_name="mock_llm_for_arith_evaluator")
        evaluator_impl = LLMQualitativeEvaluator(llm=mock_evaluator_llm)
        
        logging.info(f"Running AZ Loop for {NUM_EXAMPLE_STEPS} ARITHMETIC example steps with Evaluator...")

    elif EXAMPLE_MODE == "ARC":
        # Setup a dummy ARC task source if it doesn't exist
        dummy_arc_data_dir = "data/arc/training_dummy"
        if not os.path.exists(dummy_arc_data_dir):
            os.makedirs(dummy_arc_data_dir)
        dummy_task_file = os.path.join(dummy_arc_data_dir, "dummy_arc_task.json")
        if not os.path.exists(dummy_task_file):
            dummy_arc_content = {
                "train": [
                    {"input": [[1]], "output": [[2]]}
                ],
                "test": [
                    {"input": [[1,0],[0,1]], "output": [[0,1],[1,0]]} # Modified for a slightly more complex test
                ]
            }
            with open(dummy_task_file, 'w') as f:
                json.dump(dummy_arc_content, f)

        # Use a real LLM instance if configured, else MockLLMForLoop for ARC
        # For this example, always use MockLLMForLoop for ARC
        mock_arc_solver_llm = MockLLMForLoop(model_path_or_name="mock_llm_for_arc_solver")
        
        # ARCSolver needs a feature extractor
        feature_extractor_impl = BasicARCFeatureExtractor()
        
        proposer_impl = ARCProposer(task_dir=dummy_arc_data_dir) # Use dummy data
        solver_impl = ARCSolver(llm=mock_arc_solver_llm, feature_extractor=feature_extractor_impl)
        verifier_impl = ARCVerifier()
        reward_model_impl = ARCRewardModel() # Use ARCRewardModel
        
        # Evaluator for ARC (optional, can use the same mock or a specialized one)
        mock_arc_evaluator_llm = MockLLMForLoop(model_path_or_name="mock_llm_for_arc_evaluator")
        evaluator_impl = LLMQualitativeEvaluator(llm=mock_arc_evaluator_llm)

        logging.info(f"Running AZ Loop for {NUM_EXAMPLE_STEPS} ARC example steps with Evaluator and ARCRewardModel...")
        
    else:
        raise ValueError(f"Unknown EXAMPLE_MODE: {EXAMPLE_MODE}")

    az_loop_instance = AZLoop(
        proposer=proposer_impl,
        solver=solver_impl,
        verifier=verifier_impl,
        reward_model=reward_model_impl,
        evaluator=evaluator_impl # Pass the evaluator
    )
    logging.info("Example AZ Loop run with Evaluator complete.")

    if EXAMPLE_MODE == "ARC":
        # Clean up temporary ARC data directory
        if os.path.exists("./temp_arc_example_data"):
            shutil.rmtree("./temp_arc_example_data")
            logging.info("Cleaned up temporary ARC example data.") 