import unittest
import logging
from typing import Any, Dict, List, Optional

# Imports from ur_project
from src.ur_project.core.foundational_llm import BaseLLM, LLMResponse
from src.ur_project.core.proposer import ARCTaskProposer
from src.ur_project.core.solver import ARCSolver
from src.ur_project.core.perception import BasicARCFeatureExtractor
from src.ur_project.core.knowledge_base import ARCKnowledgeBase
from src.ur_project.core.arc_verifier import ARCVerifier
from src.ur_project.core.reward_model import SimpleArcRLRewardModel
from src.ur_project.core.evaluator import LLMQualitativeEvaluator
from src.ur_project.pipeline.az_loop import AZLoop
from src.ur_project.data_processing.arc_types import ARCPixel # For mock task generation

# Disable most logging for cleaner test output, can be re-enabled for debugging
logging.disable(logging.CRITICAL)

# --- Mock LLM for Integration Test ---
class MockLLMForIntegrationTest(BaseLLM):
    def __init__(self, model_path_or_name: str = "mock_integration_llm", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_path_or_name, config)
        self.proposer_call_count = 0
        self.solver_call_count = 0
        self.evaluator_call_count = 0
        self.rl_update_call_count = 0

    def _load_model(self):
        # No actual model loading needed for mock
        self.model = "mock_model_instance" # Satisfy checks
        self.tokenizer = "mock_tokenizer_instance" # Satisfy checks
        if self.config.get("distributed_strategy") != "fsdp":
             self.optimizer = "mock_optimizer_instance" # Satisfy checks if not FSDP
        else:
            self.optimizer = None


    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, **kwargs) -> LLMResponse:
        # Proposer prompt detection (heuristic)
        if "Please generate a new ARC task." in prompt or "TASK_NAME:" in prompt and "TRAIN_PAIRS:" in prompt and "TEST_PAIRS:" in prompt and not "Error Encountered:" in prompt:
            self.proposer_call_count += 1
            # Return a parsable ARC task structure
            mock_task_name = f"Mock Generated Task {self.proposer_call_count}"
            mock_task_desc = f"A simple mock task involving color {self.proposer_call_count}."
            # Ensure ARCPixel is not used directly here as it's for grid data, not the string representation
            # The _format_grid_for_prompt in proposer will handle list of lists of ints
            response_text = f"""TASK_NAME: {mock_task_name}
TASK_DESCRIPTION: {mock_task_desc}
TRAIN_PAIRS:
[
    {{
        "input_grid": [[{self.proposer_call_count % 10}, 0], [0, 0]],
        "output_grid": [[0, 0], [0, {self.proposer_call_count % 10}]]
    }}
]
TEST_PAIRS:
[
    {{
        "input_grid": [[0, {(self.proposer_call_count + 1) % 10}], [0, 0]],
        "output_grid": [[0, 0], [{(self.proposer_call_count + 1) % 10}, 0]]
    }}
]
"""
            return LLMResponse(text=response_text)

        # Solver prompt detection (heuristic) - initial attempt
        elif "Current Test Input Grid:" in prompt and "DSL_PROGRAM:" in prompt and not "Error Encountered:" in prompt:
            self.solver_call_count += 1
            # Return thoughts and a simple DSL program
            response_text = f"""THOUGHTS:
This is solver call #{self.solver_call_count}.
The task seems to involve moving a pixel or filling a small area.
Let's try to fill a 1x1 rectangle with color 3 at (0,0).
DSL_PROGRAM:
FILL_RECTANGLE(top_left=DSLPosition(0,0), bottom_right=DSLPosition(0,0), color=DSLColor(3))
"""
            return LLMResponse(text=response_text)
        
        # Solver retry prompt detection
        elif "Error Encountered:" in prompt and "Corrected DSL_PROGRAM" in prompt:
            self.solver_call_count += 1 # Also counts as a solver call
            # Return a slightly different or corrected DSL program for retry
            response_text = f"""THOUGHTS:
The previous attempt failed. This is retry for solver call #{self.solver_call_count}.
The error might have been due to incorrect parameters.
Let's try changing a different color, e.g. color 1 to 2.
DSL_PROGRAM:
CHANGE_COLOR(selector=DSLObjectSelector(criteria={{'old_color':1}}), new_color=DSLColor(2))
"""
            return LLMResponse(text=response_text)

        # Evaluator prompt detection (heuristic)
        elif "Is this task well-formed and solvable?" in prompt or "Does this solution seem plausible?" in prompt:
            self.evaluator_call_count += 1
            response_text = "Assessment: Plausible (Mock) Justification: The mock evaluation deems this acceptable."
            return LLMResponse(text=response_text)
        
        # Fallback for unexpected prompts
        return LLMResponse(text="Unknown prompt type received by mock LLM.")

    def batch_generate(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        return [self.generate(p, **kwargs) for p in prompts]

    def get_action_log_probs_and_train_step(self, prompt_text: str, generated_text: str, reward: float, learning_rate: float = 1e-5) -> Dict[str, Any]:
        self.rl_update_call_count += 1
        if self.optimizer is None and self.config.get("distributed_strategy") == "fsdp":
             return {"loss": 0.0, "log_probs": 0.0, "message": "Optimizer externally managed. Forward/backward/step handled outside."}
        return {"loss": 0.0, "log_probs": 0.0, "message": "Placeholder RL update."}


class TestAZLoopARCIntegration(unittest.TestCase):

    def setUp(self):
        # Mock LLMs
        self.mock_proposer_llm = MockLLMForIntegrationTest(model_path_or_name="mock_proposer")
        self.mock_solver_llm = MockLLMForIntegrationTest(model_path_or_name="mock_solver")
        self.mock_evaluator_llm = MockLLMForIntegrationTest(model_path_or_name="mock_evaluator")

        # Core Components
        self.proposer = ARCTaskProposer(llm=self.mock_proposer_llm)
        self.feature_extractor = BasicARCFeatureExtractor()
        self.knowledge_base = ARCKnowledgeBase()
        self.solver = ARCSolver(
            llm=self.mock_solver_llm,
            perception_engine=self.feature_extractor,
            knowledge_base=self.knowledge_base
        )
        self.verifier = ARCVerifier()
        self.reward_model = SimpleArcRLRewardModel()
        self.evaluator = LLMQualitativeEvaluator(llm=self.mock_evaluator_llm)

        # AZLoop Instance
        self.az_loop_instance = AZLoop(
            proposer=self.proposer,
            solver=self.solver,
            verifier=self.verifier,
            reward_model=self.reward_model,
            evaluator=self.evaluator
        )
        # Reset call counts for mocks for each test method
        self.mock_proposer_llm.proposer_call_count = 0
        self.mock_solver_llm.solver_call_count = 0
        self.mock_evaluator_llm.evaluator_call_count = 0
        self.mock_solver_llm.rl_update_call_count = 0


    def test_arc_az_loop_runs_few_steps(self):
        num_steps = 2
        self.az_loop_instance.run_loop(num_steps=num_steps)

        self.assertEqual(len(self.az_loop_instance.history), num_steps, "Loop history length mismatch.")

        for i, step_data in enumerate(self.az_loop_instance.history):
            with self.subTest(step=i):
                self.assertIsNotNone(step_data["task_id"], "Task ID is None.")
                self.assertIsNotNone(step_data["task_description"], "Task description is None.")
                self.assertIsNotNone(step_data["solution_raw"], "Raw solution is None.")
                self.assertEqual(step_data["solver_id"], self.solver.solver_id, "Solver ID mismatch.")
                self.assertIn(step_data["verification_correct"], [True, False], "Verification result invalid.")
                self.assertIsInstance(step_data["reward_value"], float, "Reward value is not a float.")
                
                self.assertIsNotNone(step_data["rl_update_info"], f"RL update info is None for step {i}")
                self.assertIn("message", step_data["rl_update_info"], f"RL update info missing 'message' for step {i}")
                # Check for specific message based on whether optimizer is mocked (it is for non-FSDP here)
                self.assertTrue(
                    step_data["rl_update_info"]["message"] == "Placeholder RL update." or \
                    step_data["rl_update_info"]["message"] == "Optimizer externally managed. Forward/backward/step handled outside.",
                    f"Unexpected RL update message: {step_data['rl_update_info']['message']}"
                )

                self.assertIsNone(step_data.get("error"), f"Step {i} encountered an error: {step_data.get('error')}")

                # Optional checks for DSL execution success
                # This depends on the mock solver's DSL being simple enough
                # The first DSL is FILL_RECTANGLE, second (retry) is CHANGE_COLOR
                # If the first one fails parsing/execution, a retry happens.
                # The mock solver's first DSL is simple and should execute.
                if step_data["verification_metadata"]:
                     # If there was no error in initial DSL, it should succeed
                    if not step_data["verification_metadata"].get("initial_dsl_parse_error") and \
                       not step_data["verification_metadata"].get("initial_dsl_execution_error"):
                        self.assertTrue(step_data["verification_metadata"].get("dsl_execution_success", False) or \
                                        step_data["verification_metadata"].get("dsl_execution_success_retry", False),
                                        f"DSL execution success not marked true in step {i}, metadata: {step_data['verification_metadata']}")
                        self.assertIsNotNone(step_data["solution_parsed"], f"Parsed solution grid is None despite DSL success in step {i}")
                
        # Check if LLMs were called
        self.assertGreaterEqual(self.mock_proposer_llm.proposer_call_count, num_steps, "Proposer LLM not called enough times.")
        self.assertGreaterEqual(self.mock_solver_llm.solver_call_count, num_steps, "Solver LLM not called enough times.")
        self.assertGreaterEqual(self.mock_evaluator_llm.evaluator_call_count, num_steps * 2, "Evaluator LLM not called enough times (task + solution).") # Task quality + solution plausibility
        self.assertGreaterEqual(self.mock_solver_llm.rl_update_call_count, num_steps, "RL update method on solver LLM not called enough times.")


if __name__ == '__main__':
    # Re-enable logging for manual runs if desired
    # logging.disable(logging.NOTSET)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
