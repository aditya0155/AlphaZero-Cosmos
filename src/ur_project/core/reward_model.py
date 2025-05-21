from typing import Protocol, Dict, Any, Optional
from ur_project.core.verifier import VerificationResult

class RewardSignal:
    def __init__(self, task_id: str, reward_value: float, reason: str, metadata: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.reward_value = reward_value
        self.reason = reason # e.g., "Correct solution", "Incorrect solution", "Parse error"
        self.metadata = metadata if metadata is not None else {}

from ur_project.core.solver import Solution # Import Solution for type hinting
from typing import List # For List type hint

class BaseRewardModel(Protocol):
    def calculate_reward(
        self, 
        verification_result: VerificationResult,
        solution: Optional[Solution] = None, 
        task: Optional[Any] = None, 
        qualitative_feedback: Optional[Dict[str, Any]] = None
    ) -> RewardSignal:
        ...

class SimpleBinaryRewardModel(BaseRewardModel):
    """Assigns a binary reward based on correctness."""
    def __init__(self, correct_reward: float = 1.0, incorrect_reward: float = -1.0):
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward

    def calculate_reward(
        self, 
        verification_result: VerificationResult,
        solution: Optional[Solution] = None,
        task: Optional[Any] = None,
        qualitative_feedback: Optional[Dict[str, Any]] = None
    ) -> RewardSignal:
        if verification_result.is_correct:
            reward_value = self.correct_reward
            reason = "Correct solution"
        else:
            reward_value = self.incorrect_reward
            reason = "Incorrect solution"
            if verification_result.metadata and "error" in verification_result.metadata:
                reason += f" (Details: {verification_result.metadata['error']})"
        
        return RewardSignal(
            task_id=verification_result.task_id,
            reward_value=reward_value,
            reason=reason,
            metadata={"verifier_metadata": verification_result.metadata}
        )

class ARCRewardModel(BaseRewardModel):
    """Assigns a more nuanced reward for ARC tasks, considering DSL execution status."""
    def __init__(
        self,
        correct_dsl_success_reward: float = 1.0, # Correct solution via successful DSL execution
        correct_llm_fallback_reward: float = 0.7, # Correct solution via LLM direct output (DSL failed or not used)
        
        incorrect_dsl_produced_wrong_grid: float = -0.8, # DSL executed successfully but grid was wrong
        incorrect_dsl_execution_error: float = -0.7,     # DSL parsed but failed during execution
        incorrect_dsl_returned_none: float = -0.6,       # DSL executed but returned None (e.g. op not fully implemented)
        incorrect_dsl_parsing_failed: float = -0.9,      # DSL string was malformed / could not be parsed
        incorrect_dsl_no_ops: float = -0.5,              # DSL parsed but had no operations
        incorrect_llm_fallback_reward: float = -0.5      # Incorrect solution, and DSL was not a factor or failed early
    ):
        self.correct_dsl_success_reward = correct_dsl_success_reward
        self.correct_llm_fallback_reward = correct_llm_fallback_reward
        self.incorrect_dsl_produced_wrong_grid = incorrect_dsl_produced_wrong_grid
        self.incorrect_dsl_execution_error = incorrect_dsl_execution_error
        self.incorrect_dsl_returned_none = incorrect_dsl_returned_none
        self.incorrect_dsl_parsing_failed = incorrect_dsl_parsing_failed
        self.incorrect_dsl_no_ops = incorrect_dsl_no_ops
        self.incorrect_llm_fallback_reward = incorrect_llm_fallback_reward

    def calculate_reward(
        self, 
        verification_result: VerificationResult,
        solution: Optional[Solution] = None, 
        task: Optional[Any] = None, 
        qualitative_feedback: Optional[Dict[str, Any]] = None
    ) -> RewardSignal:
        reward_value: float
        reason: str
        
        # ARCVerifier populates metadata like "dsl_executed_successfully", "final_dsl_error_summary"
        dsl_executed_successfully = verification_result.metadata.get("dsl_executed_successfully", False)
        final_dsl_error_summary = verification_result.metadata.get("final_dsl_error_summary", "Unknown DSL status")

        if verification_result.is_correct:
            if dsl_executed_successfully:
                reward_value = self.correct_dsl_success_reward
                reason = "Correct solution via successful DSL program execution."
            else:
                # If correct but DSL didn't execute successfully, it might be a non-DSL solution or an issue.
                reward_value = self.correct_llm_fallback_reward 
                reason = f"Correct solution. DSL status: {final_dsl_error_summary}."
        else: # Incorrect solution
            if dsl_executed_successfully: 
                reward_value = self.incorrect_dsl_produced_wrong_grid
                reason = "Incorrect solution: DSL program executed successfully but produced the wrong grid."
            elif "parse error" in final_dsl_error_summary.lower():
                reward_value = self.incorrect_dsl_parsing_failed
                reason = f"Incorrect solution: DSL parsing failed ({final_dsl_error_summary})."
            elif "execution error" in final_dsl_error_summary.lower():
                reward_value = self.incorrect_dsl_execution_error
                reason = f"Incorrect solution: DSL execution failed ({final_dsl_error_summary})."
            elif final_dsl_error_summary == "DSL processed OK, but no output grid was generated.":
                reward_value = self.incorrect_dsl_returned_none
                reason = "Incorrect solution: DSL executed but returned no grid."
            elif solution and (not solution.raw_dsl_program and not solution.raw_dsl_program_retry) and \
                 not (verification_result.metadata.get("initial_parse_error") or \
                      verification_result.metadata.get("initial_execution_error") or \
                      verification_result.metadata.get("retry_parse_error") or \
                      verification_result.metadata.get("retry_execution_error")):
                 # This condition checks if no DSL was provided AND no other DSL errors occurred.
                 # (If raw_dsl_program was empty and caused a parse error, that's handled by "parse error" above)
                 reward_value = self.incorrect_llm_fallback_reward - 0.2 # Specific penalty for no DSL at all
                 reason = "Incorrect solution: No DSL program was attempted by the solver."
            else: 
                reward_value = self.incorrect_llm_fallback_reward
                reason = f"Incorrect solution. DSL status: {final_dsl_error_summary}."
            
            # Avoid duplicating error messages if final_dsl_error_summary already captured it.
            if verification_result.metadata and "error" in verification_result.metadata and \
               final_dsl_error_summary not in verification_result.metadata["error"] :
                reason += f" Verifier detail: {verification_result.metadata['error']}."

        return RewardSignal(
            task_id=verification_result.task_id,
            reward_value=reward_value,
            reason=reason,
            metadata={"verifier_metadata": verification_result.metadata} 
        )

# --- SimpleArcRLRewardModel ---
DEFAULT_REWARD_CONFIG = {
    "correct_solution_bonus": 1.0,
    "dsl_executed_bonus": 0.2,          # DSL ran successfully but wrong grid
    "successful_parse_bonus": 0.1,      # Parsed OK, but execution failed (initial)
    "successful_retry_parse_bonus": 0.05, # Parsed OK on retry, but execution failed
    "initial_parse_error_penalty": -0.3,
    "initial_execution_error_penalty": -0.2,
    "retry_parse_error_penalty": -0.15, 
    "retry_execution_error_penalty": -0.1, 
    "no_dsl_attempt_penalty": -0.5, 
    "good_task_bonus": 0.05,
    "bad_task_penalty": -0.05,
    "good_solution_plausibility_bonus": 0.05,
    "bad_solution_plausibility_penalty": -0.05,
}

class SimpleArcRLRewardModel(BaseRewardModel):
    """
    Calculates rewards for ARC tasks based on DSL processing success and qualitative feedback.
    """
    def __init__(self, reward_config: Optional[Dict[str, float]] = None):
        self.config = DEFAULT_REWARD_CONFIG.copy()
        if reward_config:
            self.config.update(reward_config)

    def calculate_reward(
        self, 
        verification_result: VerificationResult,
        solution: Optional[Solution] = None, 
        task: Optional[Any] = None, 
        qualitative_feedback: Optional[Dict[str, Any]] = None
    ) -> RewardSignal:
        
        reward = 0.0
        reason_parts: List[str] = []

        if solution is None: 
            # This case should ideally be prevented by AZLoop ensuring a Solution object is always passed.
            return RewardSignal(verification_result.task_id, -1.0, "Critical error: Solution object not provided to reward model.", {})

        # 1. Correctness (Primary Reward)
        if verification_result.is_correct:
            reward += self.config["correct_solution_bonus"]
            reason_parts.append(f"Correct solution (+{self.config['correct_solution_bonus']:.2f}).")
        
        # 2. DSL Processing Rewards/Penalties
        # Uses metadata from VerificationResult, which ARCVerifier populates from the Solution object.
        initial_parse_error = verification_result.metadata.get("initial_parse_error")
        initial_exec_error = verification_result.metadata.get("initial_execution_error")
        retry_attempted = verification_result.metadata.get("retry_attempted", False)
        retry_parse_error = verification_result.metadata.get("retry_parse_error")
        retry_exec_error = verification_result.metadata.get("retry_execution_error")
        dsl_executed_successfully = verification_result.metadata.get("dsl_executed_successfully", False)

        if not verification_result.is_correct: 
            if dsl_executed_successfully: # DSL ran but produced wrong grid
                reward += self.config["dsl_executed_bonus"]
                reason_parts.append(f"DSL executed (but result incorrect) (+{self.config['dsl_executed_bonus']:.2f}).")
            else: # DSL did not execute successfully to produce a final grid.
                # Penalize based on the first point of failure.
                if initial_parse_error:
                    reward += self.config["initial_parse_error_penalty"]
                    reason_parts.append(f"Initial DSL parse error ({self.config['initial_parse_error_penalty']:.2f}).")
                    if retry_attempted and retry_parse_error: # Additional penalty if retry also fails parsing
                        reward += self.config["retry_parse_error_penalty"] # This is an additional penalty
                        reason_parts.append(f"Retry DSL also failed to parse ({self.config['retry_parse_error_penalty']:.2f}).")
                elif initial_exec_error: # Parsed OK, but execution failed
                    reward += self.config["initial_execution_error_penalty"]
                    reason_parts.append(f"Initial DSL execution error ({self.config['initial_execution_error_penalty']:.2f}).")
                    reward += self.config["successful_parse_bonus"] 
                    reason_parts.append(f"Initial DSL parsed correctly (+{self.config['successful_parse_bonus']:.2f}).")
                    if retry_attempted and retry_exec_error: # Retry also failed execution
                         reward += self.config["retry_execution_error_penalty"]
                         reason_parts.append(f"Retry DSL execution error ({self.config['retry_execution_error_penalty']:.2f}).")
                         if not retry_parse_error: # Parsed on retry
                            reward += self.config["successful_retry_parse_bonus"]
                            reason_parts.append(f"Retry DSL parsed correctly (+{self.config['successful_retry_parse_bonus']:.2f}).")
                    elif retry_attempted and retry_parse_error: # Retry failed parsing (after initial exec error)
                        reward += self.config["retry_parse_error_penalty"]
                        reason_parts.append(f"Retry DSL parse error after initial exec error ({self.config['retry_parse_error_penalty']:.2f}).")

                # If no DSL program was provided by the solver at all (check Solution object)
                # This applies if there were no parse/exec errors because there was nothing to parse/execute.
                if not solution.raw_dsl_program and not initial_parse_error and not initial_exec_error:
                    # Only apply if not already penalized for parse/exec error (which implies some DSL was present)
                    reward += self.config["no_dsl_attempt_penalty"]
                    reason_parts.append(f"No DSL program attempted by solver ({self.config['no_dsl_attempt_penalty']:.2f}).")

        # 3. Qualitative Feedback (Optional)
        if qualitative_feedback:
            task_assessment = qualitative_feedback.get("task_quality_assessment")
            solution_assessment = qualitative_feedback.get("solution_plausibility_assessment")

            if task_assessment:
                if any(kw in task_assessment.lower() for kw in ["good", "well-formed", "clear"]):
                    reward += self.config["good_task_bonus"]
                    reason_parts.append(f"Good task quality (+{self.config['good_task_bonus']:.2f}).")
                elif any(kw in task_assessment.lower() for kw in ["bad", "poor", "unclear"]):
                    reward += self.config["bad_task_penalty"]
                    reason_parts.append(f"Bad task quality ({self.config['bad_task_penalty']:.2f}).")
            
            if solution_assessment:
                if any(kw in solution_assessment.lower() for kw in ["good", "plausible", "logical"]):
                    reward += self.config["good_solution_plausibility_bonus"]
                    reason_parts.append(f"Good solution plausibility (+{self.config['good_solution_plausibility_bonus']:.2f}).")
                elif any(kw in solution_assessment.lower() for kw in ["bad", "implausible", "illogical"]):
                    reward += self.config["bad_solution_plausibility_penalty"]
                    reason_parts.append(f"Bad solution plausibility ({self.config['bad_solution_plausibility_penalty']:.2f}).")
        
        final_reason = "; ".join(reason_parts) if reason_parts else "No specific reward conditions met."
        if not reason_parts and reward == 0 and not verification_result.is_correct:
            final_reason = "Incorrect solution with no specific DSL penalties/bonuses or qualitative feedback."
        elif not reason_parts and reward != 0: 
            final_reason = f"Reward accumulated to {reward:.2f} without specific reasons recorded."

        return RewardSignal(
            task_id=verification_result.task_id,
            reward_value=round(reward, 4), 
            reason=final_reason,
            metadata={ 
                "verifier_metadata_summary": {
                    "dsl_executed_successfully": dsl_executed_successfully,
                    "final_dsl_error_summary": verification_result.metadata.get("final_dsl_error_summary"),
                    "retry_attempted": retry_attempted
                },
                "qualitative_feedback_applied": qualitative_feedback is not None
            }
        )

# Example Usage:
# if __name__ == "__main__":
#     # Assume we have some VerificationResult instances
#     result_correct = VerificationResult(task_id="task1", is_correct=True, actual_solution=5.0, expected_solution=5.0)
#     result_incorrect = VerificationResult(task_id="task2", is_correct=False, actual_solution=3.0, expected_solution=7.0)
#     result_parse_error = VerificationResult(
#         task_id="task3", 
#         is_correct=False, 
#         actual_solution="NaN", 
#         expected_solution=10.0, 
#         metadata={"error": "Solution could not be parsed."}
#     )

#     # Example of how SimpleArcRLRewardModel might be used (requires Solution object)
#     # mock_solution_obj = Solution(task_id="task_arc_1", solved_by="test", raw_answer="dsl") 
#     # mock_solution_obj.raw_dsl_program = "FILL_RECTANGLE(...)"
#     # mock_verification_result_arc = VerificationResult(
#     #    task_id="task_arc_1", 
#     #    is_correct=True, 
#     #    actual_solution=[[1]], 
#     #    expected_solution=[[1]],
#     #    metadata={"dsl_executed_successfully": True, "final_dsl_error_summary": "OK", "retry_attempted": False}
#     # )
#     # arc_rl_reward_model = SimpleArcRLRewardModel()
#     # reward_arc = arc_rl_reward_model.calculate_reward(mock_verification_result_arc, mock_solution_obj)
#     # print(f"Task: {reward_arc.task_id}, Reward: {reward_arc.reward_value}, Reason: {reward_arc.reason}")


#     reward_model = SimpleBinaryRewardModel() # Keep this for existing non-ARC tests if any

#     reward1 = reward_model.calculate_reward(result_correct) # solution, task, qualitative_feedback are optional
#     print(f"Task: {reward1.task_id}, Reward: {reward1.reward_value}, Reason: {reward1.reason}")

#     reward2 = reward_model.calculate_reward(result_incorrect)
#     print(f"Task: {reward2.task_id}, Reward: {reward2.reward_value}, Reason: {reward2.reason}")
    
#     reward3 = reward_model.calculate_reward(result_parse_error)
#     print(f"Task: {reward3.task_id}, Reward: {reward3.reward_value}, Reason: {reward3.reason}")
#     if reward3.metadata:
#         print(f"  Reward Metadata: {reward3.metadata}")