from typing import Protocol, Dict, Any, Optional
from ur_project.core.verifier import VerificationResult

class RewardSignal:
    def __init__(self, task_id: str, reward_value: float, reason: str, metadata: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.reward_value = reward_value
        self.reason = reason # e.g., "Correct solution", "Incorrect solution", "Parse error"
        self.metadata = metadata if metadata is not None else {}

class BaseRewardModel(Protocol):
    def calculate_reward(self, verification_result: VerificationResult) -> RewardSignal:
        ...

class SimpleBinaryRewardModel(BaseRewardModel):
    """Assigns a binary reward based on correctness."""
    def __init__(self, correct_reward: float = 1.0, incorrect_reward: float = -1.0):
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward

    def calculate_reward(self, verification_result: VerificationResult) -> RewardSignal:
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

    def calculate_reward(self, verification_result: VerificationResult) -> RewardSignal:
        reward_value: float
        reason: str
        
        dsl_status = verification_result.metadata.get("dsl_execution_status")

        if verification_result.is_correct:
            if dsl_status == "success":
                reward_value = self.correct_dsl_success_reward
                reason = "Correct solution via successful DSL program execution."
            else:
                # Correct, but DSL was not successful or not present. This means LLM direct output was correct.
                reward_value = self.correct_llm_fallback_reward
                reason = f"Correct solution (LLM direct output). DSL status: {dsl_status if dsl_status else 'N/A'}."
        else: # Incorrect solution
            if dsl_status == "success": # DSL ran successfully but produced the wrong grid
                reward_value = self.incorrect_dsl_produced_wrong_grid
                reason = "Incorrect solution: DSL program executed successfully but produced the wrong grid."
            elif dsl_status == "execution_error":
                reward_value = self.incorrect_dsl_execution_error
                reason = "Incorrect solution: DSL program parsed but failed during execution."
            elif dsl_status == "execution_returned_none":
                reward_value = self.incorrect_dsl_returned_none
                reason = "Incorrect solution: DSL program executed but returned no grid (e.g., op not fully implemented or error)."
            elif dsl_status == "parsing_failed":
                reward_value = self.incorrect_dsl_parsing_failed
                reason = "Incorrect solution: Failed to parse DSL program string."
            elif dsl_status == "no_operations_to_execute":
                reward_value = self.incorrect_dsl_no_ops
                reason = "Incorrect solution: DSL program parsed but contained no operations."
            else: # DSL was not a factor or failed before execution status was set in a specific way
                reward_value = self.incorrect_llm_fallback_reward
                reason = f"Incorrect solution (LLM direct output or unclear DSL state). DSL status: {dsl_status if dsl_status else 'N/A'}."
            
            # Append verifier's specific error if available
            if verification_result.metadata and "error" in verification_result.metadata:
                reason += f" Verifier detail: {verification_result.metadata['error']}."
            if verification_result.metadata and "dsl_execution_error_message" in verification_result.metadata:
                 reason += f" DSL Error: {verification_result.metadata['dsl_execution_error_message']}."

        return RewardSignal(
            task_id=verification_result.task_id,
            reward_value=reward_value,
            reason=reason,
            metadata={"verifier_metadata": verification_result.metadata} # Keep original verifier metadata
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

#     reward_model = SimpleBinaryRewardModel()

#     reward1 = reward_model.calculate_reward(result_correct)
#     print(f"Task: {reward1.task_id}, Reward: {reward1.reward_value}, Reason: {reward1.reason}")

#     reward2 = reward_model.calculate_reward(result_incorrect)
#     print(f"Task: {reward2.task_id}, Reward: {reward2.reward_value}, Reason: {reward2.reason}")
    
#     reward3 = reward_model.calculate_reward(result_parse_error)
#     print(f"Task: {reward3.task_id}, Reward: {reward3.reward_value}, Reason: {reward3.reason}")
#     if reward3.metadata:
#         print(f"  Reward Metadata: {reward3.metadata}") 