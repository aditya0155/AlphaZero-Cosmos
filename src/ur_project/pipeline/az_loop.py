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
from ur_project.core.proposer import ARCTaskProposer 
from ur_project.core.solver import ARCSolver
from ur_project.core.arc_verifier import ARCVerifier
from ur_project.data_processing.arc_types import ARCGrid, ARCPixel, ARCTask, ARCPuzzle 
# Updated perception, KB, and new Phase 3 components
from ur_project.core.perception import UNetARCFeatureExtractor 
from ur_project.core.knowledge_base import ARCKnowledgeBase 
from ur_project.core.symbol_grounding import SymbolGroundingModel
from ur_project.core.symbolic_learner import SimpleRuleInducer

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
            "timings": {},
            "rl_update_info": None # Added for RL fine-tuning step
        }

        # 1. Propose Task
        start_time = time.time()
        task_to_process: Task # This will be the object passed to solver and verifier (e.g. ARCPuzzle)
        original_task_id_for_logging = f"iter_{step_number}_raw_proposal"

        if isinstance(self.proposer, ARCTaskProposer):
            generated_task_id = f"arc_gen_iter_{step_number}"
            original_task_id_for_logging = generated_task_id # Use the ID passed to proposer
            try:
                # ARCTaskProposer's propose_task returns ARCTask, task_name_str, task_desc_str
                original_arc_task, task_name_str, task_desc_str = self.proposer.propose_task(task_id=generated_task_id) # Concept omitted for now
                logging.info(f"Step {step_number} [Propose]: ARCTaskProposer proposed ARCTask ID {original_arc_task.task_id}, Name: '{task_name_str}'.")

                if not original_arc_task.test_pairs:
                    logging.error(f"Step {step_number} [Propose]: ARCTask {original_arc_task.task_id} has no test pairs. Skipping.")
                    step_data["error"] = "Proposed ARCTask has no test pairs."
                    self.history.append(step_data)
                    return step_data # Early exit for this step if no test pairs
                
                first_test_pair = original_arc_task.test_pairs[0]
                
                # ARCTaskProposer is expected to parse task_name and task_description.
                # These are not part of ARCTask structure but could be in a metadata dict if proposer returns it,
                # or ARCTask could be augmented. For now, use generic description.
                # The ARCTaskProposer's _parse_llm_response_to_arctask creates metadata including task_name & desc.
                # task_name_str and task_desc_str are now directly available.

                task_to_process = ARCPuzzle(
                    id=f"{original_arc_task.task_id}_test_0",
                    description=f"{task_name_str}: {task_desc_str}", 
                    data=first_test_pair.input_grid,
                    expected_output_grid=first_test_pair.output_grid,
                    source_task_id=original_arc_task.task_id,
                    source_pair_id=first_test_pair.pair_id or "test_0", 
                    text_description=task_desc_str,
                    metadata={
                        "original_training_pairs_count": len(original_arc_task.training_pairs),
                        "original_task_name": task_name_str, 
                        "original_task_description": task_desc_str,
                        "full_arc_task_id": original_arc_task.task_id
                    }
                )
                logging.info(f"Step {step_number} [Propose]: Converted ARCTask {original_arc_task.task_id} to ARCPuzzle {task_to_process.id} ('{task_to_process.description}') for solving.")

            except Exception as e:
                logging.error(f"Step {step_number} [Propose]: Error during ARCTaskProposer proposal or ARCPuzzle conversion: {e}", exc_info=True)
                step_data["error"] = f"Task proposal/conversion failed: {e}"
                self.history.append(step_data)
                return step_data
        else: # For other proposers like SimpleArithmeticProposer
            # This branch handles non-ARCTaskProposer cases
            task_to_process = self.proposer.propose_task() # Existing generic call
            original_task_id_for_logging = task_to_process.id
            logging.info(f"Step {step_number} [Propose]: Task ID {task_to_process.id} - '{task_to_process.description}' proposed by {type(self.proposer).__name__}.")

        propose_time = time.time() - start_time
        step_data["timings"]["propose"] = propose_time
        step_data["task_id"] = task_to_process.id
        step_data["task_description"] = task_to_process.description # This is ARCPuzzle's description
        step_data["task_data"] = task_to_process.data 
        # Logging for task_id and description is now part of the if/else block

        # 1b. Assess Task Quality (Optional) - uses task_to_process (e.g. ARCPuzzle)
        task_quality_feedback: Optional[QualitativeFeedback] = None
        if self.evaluator:
            start_time = time.time()
            task_quality_feedback = self.evaluator.assess_task_quality(task_to_process)
            eval_task_time = time.time() - start_time
            step_data["timings"]["evaluate_task"] = eval_task_time
            if task_quality_feedback:
                logging.info(f"Step {step_number} [Evaluate Task]: Task ID {task_to_process.id} - Assessment: '{task_quality_feedback.assessment}', Reasoning: '{task_quality_feedback.reasoning}' (took {eval_task_time:.3f}s)")
                step_data["task_quality_assessment"] = task_quality_feedback.assessment
                step_data["task_quality_reasoning"] = task_quality_feedback.reasoning
                step_data["reward_metadata"].update({"task_quality_feedback": task_quality_feedback})
            else:
                logging.info(f"Step {step_number} [Evaluate Task]: Task ID {task_to_process.id} - No feedback provided.")

        # 2. Solve Task - uses task_to_process (e.g. ARCPuzzle)
        start_time = time.time()
        solution = self.solver.solve_task(task_to_process)
        solve_time = time.time() - start_time
        step_data["timings"]["solve"] = solve_time
        step_data["solution_raw"] = solution.raw_answer
        step_data["solution_parsed"] = solution.parsed_answer
        step_data["solver_id"] = solution.solved_by
        
        solve_log_message = f"Step {step_number} [Solve]: Task ID {task_to_process.id} - Solver {solution.solved_by} -> Raw: '{str(solution.raw_answer)[:100]}...', Parsed: {solution.parsed_answer} (took {solve_time:.3f}s)"
        if isinstance(self.solver, ARCSolver): # Check if it's an ARCSolver to expect ARC solution fields
            if solution.dsl_parse_error: solve_log_message += f" | Initial Parse Error: Yes"
            if solution.dsl_execution_error: solve_log_message += f" | Initial Exec Error: Yes"
            if solution.retry_attempted:
                solve_log_message += " | Retry: Yes"
                if solution.dsl_parse_error_retry: solve_log_message += f" | Retry Parse Error: Yes"
                if solution.dsl_execution_error_retry: solve_log_message += f" | Retry Exec Error: Yes"
            if solution.parsed_answer: solve_log_message += " | Final Grid: Yes"
            elif not (solution.dsl_parse_error or solution.dsl_execution_error or solution.dsl_parse_error_retry or solution.dsl_execution_error_retry):
                 solve_log_message += " | Final Grid: No (but no errors reported post-retry)"
        logging.info(solve_log_message)


        # 2b. Assess Solution Plausibility (Optional) - uses task_to_process
        solution_plausibility_feedback: Optional[QualitativeFeedback] = None
        if self.evaluator:
            start_time = time.time()
            solution_plausibility_feedback = self.evaluator.assess_solution_plausibility(task_to_process, solution)
            eval_sol_time = time.time() - start_time
            step_data["timings"]["evaluate_solution"] = eval_sol_time
            if solution_plausibility_feedback:
                logging.info(f"Step {step_number} [Evaluate Solution]: Task ID {task_to_process.id} - Assessment: '{solution_plausibility_feedback.assessment}', Reasoning: '{solution_plausibility_feedback.reasoning}' (took {eval_sol_time:.3f}s)")
                step_data["solution_plausibility_assessment"] = solution_plausibility_feedback.assessment
                step_data["solution_plausibility_reasoning"] = solution_plausibility_feedback.reasoning
                step_data["reward_metadata"].update({"solution_plausibility_feedback": solution_plausibility_feedback})
            else:
                logging.info(f"Step {step_number} [Evaluate Solution]: Task ID {task_to_process.id} - No feedback provided.")

        # 3. Verify Solution - uses task_to_process
        start_time = time.time()
        # Compatibility check based on the actual task_to_process type
        expected_verifier_task_type = getattr(self.verifier, 'task_type_expected', None)
        if expected_verifier_task_type and not isinstance(task_to_process, expected_verifier_task_type):
            logging.warning(f"Step {step_number} [Verify]: Verifier {type(self.verifier).__name__} (expects {expected_verifier_task_type.__name__}) may not be compatible with task type {type(task_to_process).__name__}.")
        
        try:
            verification_result = self.verifier.verify_solution(task_to_process, solution)
        except Exception as e:
            logging.error(f"Step {step_number} [Verify]: Error during verification for task {task_to_process.id}: {e}", exc_info=True)
            verification_result = VerificationResult(
                task_id=task_to_process.id, is_correct=False, actual_solution=solution.parsed_answer,
                metadata={"error": "Verification failed due to verifier incompatibility or error", "exception": str(e)}
            )
        verify_time = time.time() - start_time
        step_data["timings"]["verify"] = verify_time
        step_data["verification_correct"] = verification_result.is_correct
        step_data["verification_actual"] = verification_result.actual_solution
        step_data["verification_expected"] = verification_result.expected_solution
        step_data["verification_metadata"] = verification_result.metadata
        logging.info(f"Step {step_number} [Verify]: Task ID {task_to_process.id} - Correct: {verification_result.is_correct}, Actual: {verification_result.actual_solution}, Expected: {verification_result.expected_solution} (took {verify_time:.3f}s)")
        if verification_result.metadata:
            logging.info(f"Step {step_number} [Verify]: Metadata: {verification_result.metadata}")

        # 4. Calculate Reward
        start_time = time.time()
        
        # Consolidate qualitative feedback for the reward model
        qualitative_feedback_for_reward = {}
        if task_quality_feedback: # This is QualitativeFeedback object
            qualitative_feedback_for_reward["task_quality_assessment"] = task_quality_feedback.assessment
            qualitative_feedback_for_reward["task_quality_reasoning"] = task_quality_feedback.reasoning
        if solution_plausibility_feedback: # This is QualitativeFeedback object
            qualitative_feedback_for_reward["solution_plausibility_assessment"] = solution_plausibility_feedback.assessment
            qualitative_feedback_for_reward["solution_plausibility_reasoning"] = solution_plausibility_feedback.reasoning

        reward_signal = self.reward_model.calculate_reward(
            verification_result=verification_result,
            solution=solution, 
            task=task_to_process, 
            qualitative_feedback=qualitative_feedback_for_reward if qualitative_feedback_for_reward else None
        )
        reward_time = time.time() - start_time
        step_data["timings"]["reward"] = reward_time
        step_data["reward_value"] = reward_signal.reward_value
        step_data["reward_reason"] = reward_signal.reason
        
        # step_data["reward_metadata"] already contains task_quality_feedback and solution_plausibility_feedback (as objects)
        # Add reward signal's own metadata if any (e.g. verifier_metadata_summary from SimpleArcRLRewardModel)
        if reward_signal.metadata:
            step_data["reward_metadata"].update(reward_signal.metadata)

        logging.info(f"Step {step_number} [Reward]: Task ID {task_to_process.id} - Reward: {reward_signal.reward_value}, Reason: '{reward_signal.reason}' (took {reward_time:.3f}s)")
        if step_data["reward_metadata"]: 
            logging.info(f"Step {step_number} [Reward]: Full Metadata: {step_data['reward_metadata']}")

        # 5. RL Fine-tuning (Optional - if solver and LLM support it)
        start_time = time.time()
        rl_update_info_dict = None
        if isinstance(self.solver, ARCSolver) and isinstance(self.solver.llm, BaseLLM):
            prompt_text_for_rl = solution.prompt_for_llm
            generated_dsl_for_rl = None

            # Determine which DSL program led to the final successful answer
            if solution.parsed_answer is not None: # A successful grid was produced
                if solution.retry_attempted and solution.executed_grid_retry == solution.parsed_answer:
                    generated_dsl_for_rl = solution.raw_dsl_program_retry
                elif not solution.retry_attempted and solution.executed_grid == solution.parsed_answer:
                    generated_dsl_for_rl = solution.raw_dsl_program
                # If parsed_answer is set but doesn't match either executed_grid or executed_grid_retry
                # (which shouldn't happen with current solver logic but good to be aware),
                # then generated_dsl_for_rl might remain None or we might default to raw_dsl_program.
                # Current logic: only use DSL if it directly led to parsed_answer.
                if generated_dsl_for_rl is None and solution.raw_dsl_program: # Fallback for safety, though less direct
                    logging.warning(f"Step {step_number} [RL Fine-tune]: Parsed answer exists but doesn't match executed_grid or executed_grid_retry. Falling back to initial raw_dsl_program for RL if available.")
                    if not solution.retry_attempted: # only use initial if no retry was made or retry failed to produce parsed_answer
                         generated_dsl_for_rl = solution.raw_dsl_program


            if prompt_text_for_rl and generated_dsl_for_rl:
                try:
                    # Ensure the LLM instance from the solver is used
                    rl_update_info_dict = self.solver.llm.get_action_log_probs_and_train_step(
                        prompt_text=prompt_text_for_rl,
                        generated_text=generated_dsl_for_rl,
                        reward=float(reward_signal.reward_value) # Ensure reward is float
                    )
                    logging.info(f"Step {step_number} [RL Fine-tune]: Update info: {rl_update_info_dict}")
                    step_data["rl_update_info"] = rl_update_info_dict
                except Exception as e_rl:
                    logging.error(f"Step {step_number} [RL Fine-tune]: Error during RL update step: {e_rl}", exc_info=True)
                    step_data["rl_update_info"] = {"error": str(e_rl)}
            else:
                logging.info(f"Step {step_number} [RL Fine-tune]: Skipping RL update. Prompt or generated DSL not available. Prompt: {'OK' if prompt_text_for_rl else 'Missing'}, DSL: {'OK' if generated_dsl_for_rl else 'Missing'}")
        else:
            logging.info(f"Step {step_number} [RL Fine-tune]: Skipping RL update. Solver is not ARCSolver or its LLM is not BaseLLM.")
        
        rl_finetune_time = time.time() - start_time
        step_data["timings"]["rl_finetune"] = rl_finetune_time
        if rl_update_info_dict: # Log time only if attempted
             logging.info(f"Step {step_number} [RL Fine-tune]: RL fine-tuning attempt took {rl_finetune_time:.3f}s.")


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
        
        # Instantiate new Phase 3 components
        # UNetARCFeatureExtractor can be configured (e.g. model_path, use_unet_preprocessing)
        # For this example, use_unet_preprocessing=False means it acts like BasicARCFeatureExtractor but includes advanced algorithmic features.
        feature_extractor_impl = UNetARCFeatureExtractor(use_unet_preprocessing=False) 
        symbol_grounder_impl = SymbolGroundingModel()
        rule_inducer_impl = SimpleRuleInducer()
        # ARCKnowledgeBase is instantiated inside ARCSolver, so not needed here.
        
        # ARCTaskProposer needs an LLM.
        mock_arc_task_proposer_llm = MockLLMForLoop(model_path_or_name="mock_llm_for_arc_task_proposer")
        proposer_impl = ARCTaskProposer(llm=mock_arc_task_proposer_llm)
        
        # Update ARCSolver instantiation
        solver_impl = ARCSolver(
            llm=mock_arc_solver_llm, 
            feature_extractor=feature_extractor_impl,
            symbol_grounder=symbol_grounder_impl,
            rule_inducer=rule_inducer_impl
        )
        verifier_impl = ARCVerifier()
        
        from ur_project.core.reward_model import SimpleArcRLRewardModel 
        reward_model_impl = SimpleArcRLRewardModel() 
        
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