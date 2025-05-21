from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Protocol
from dataclasses import dataclass, field

from ur_project.core.foundational_llm import BaseLLM, LLMResponse
from ur_project.core.proposer import Task # Assuming Task protocol from proposer.py
from ur_project.core.solver import Solution # Assuming Solution class from solver.py

@dataclass
class QualitativeFeedback:
    """Stores qualitative feedback from an LLM-based evaluator."""
    assessment: str # e.g., "Plausible", "Seems well-formed", "Potentially problematic"
    reasoning: Optional[str] = None # LLM's justification for the assessment
    confidence: Optional[float] = None # If the LLM can provide a confidence score
    raw_llm_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseEvaluator(Protocol):
    """Protocol for an evaluator component that provides qualitative assessments."""

    def assess_task_quality(self, task: Task) -> Optional[QualitativeFeedback]:
        """Assesses the quality or interestingness of a proposed task."""
        ...

    def assess_solution_plausibility(self, task: Task, solution: Solution) -> Optional[QualitativeFeedback]:
        """Assesses the plausibility or quality of a proposed solution to a task."""
        ...

class LLMQualitativeEvaluator(BaseEvaluator):
    """Uses a foundational LLM to provide qualitative evaluations."""

    def __init__(self, llm: BaseLLM, evaluator_id: str = "LLMQualitativeEvaluator_v1"):
        self.llm = llm
        self.evaluator_id = evaluator_id
        # TODO: Define specific prompts for task quality and solution plausibility
        self.task_quality_prompt_template = (
            "Consider the following puzzle task:\n"
            "Task ID: {task_id}\n"
            "Description: {task_description}\n"
            # Future: Could include compact representation of task.data if useful (e.g., grid dimensions for ARC)
            "Is this task well-formed, interesting, and unambiguous for a reasoning benchmark? "
            "Provide a brief assessment (e.g., Well-formed, Ambiguous, Too simple, Interesting concept) "
            "and a short justification. Format your response as: Assessment: [Your Assessment] Justification: [Your Justification]"
        )
        self.solution_plausibility_prompt_template = (
            "Consider the following puzzle task and a proposed solution:\n"
            "Task ID: {task_id}\n"
            "Task Description: {task_description}\n"
            "Proposed Solution (raw output from solver): {raw_solution}\n"
            # Future: Could include task.data or parsed_solution if more helpful
            "Does this solution seem plausible for solving the task? Consider its coherence and relevance. "
            "Provide a brief assessment (e.g., Plausible, Implausible, Partially plausible, Unclear) "
            "and a short justification. Format your response as: Assessment: [Your Assessment] Justification: [Your Justification]"
        )

    def _parse_llm_assessment_response(self, response_text: str) -> QualitativeFeedback:
        """Rudimentary parsing of the LLM's formatted assessment string."""
        assessment = "Could not parse LLM response"
        justification = "Raw response: " + response_text
        try:
            # Attempt to find "Assessment:" and "Justification:"
            assessment_part = None
            justification_part = None

            if "Assessment:" in response_text:
                assessment_split = response_text.split("Assessment:", 1)[1]
                if "Justification:" in assessment_split:
                    assessment_part = assessment_split.split("Justification:", 1)[0].strip()
                    justification_part = assessment_split.split("Justification:", 1)[1].strip()
                else:
                    assessment_part = assessment_split.strip()
            elif "Justification:" in response_text: # Only justification found
                 justification_part = response_text.split("Justification:", 1)[1].strip()
            
            if assessment_part:
                assessment = assessment_part
            if justification_part:
                justification = justification_part
            
        except Exception as e:
            print(f"Error parsing LLM assessment response ('{response_text}'): {e}")
            # Fallback to using the raw text if parsing fails badly
            assessment = "Parsing error"
            justification = response_text 

        return QualitativeFeedback(
            assessment=assessment,
            reasoning=justification,
            raw_llm_response=response_text
        )

    def assess_task_quality(self, task: Task) -> Optional[QualitativeFeedback]:
        prompt = self.task_quality_prompt_template.format(
            task_id=task.id,
            task_description=task.description
        )
        try:
            llm_response: LLMResponse = self.llm.generate(
                prompt,
                max_new_tokens=100, # Adjust as needed
                temperature=0.5 # Encourage factual but slightly creative assessment
            )
            return self._parse_llm_assessment_response(llm_response.text)
        except Exception as e:
            print(f"Error during LLM call for task quality assessment ({task.id}): {e}")
            return QualitativeFeedback(assessment="Error during LLM call", reasoning=str(e))

    def assess_solution_plausibility(self, task: Task, solution: Solution) -> Optional[QualitativeFeedback]:
        prompt = self.solution_plausibility_prompt_template.format(
            task_id=task.id,
            task_description=task.description,
            raw_solution=solution.raw_answer
        )
        try:
            llm_response: LLMResponse = self.llm.generate(
                prompt,
                max_new_tokens=100, # Adjust as needed
                temperature=0.5 
            )
            return self._parse_llm_assessment_response(llm_response.text)
        except Exception as e:
            print(f"Error during LLM call for solution plausibility assessment ({task.id}, {solution.solved_by}): {e}")
            return QualitativeFeedback(assessment="Error during LLM call", reasoning=str(e))


# Example Usage (for illustration, requires mock/real LLM and task/solution objects)
# if __name__ == '__main__':
#     # Mock LLM that returns predictable assessment strings
#     class MockEvaluatorLLM(BaseLLM):
#         def _load_model(self):
#             print("MockEvaluatorLLM loaded.")
#         def generate(self, prompt: str, **kwargs) -> LLMResponse:
#             if "Is this task well-formed" in prompt:
#                 return LLMResponse("Assessment: Well-formed Justification: The task description is clear and objective seems achievable.")
#             elif "Does this solution seem plausible" in prompt:
#                 return LLMResponse("Assessment: Plausible Justification: The solution directly addresses the arithmetic operation.")
#             return LLMResponse("Assessment: Unknown query Justification: The prompt was not recognized by mock.")
#         def batch_generate(self, prompts: list[str], **kwargs) -> list[LLMResponse]:
#             return [self.generate(p) for p in prompts]

#     # Mock Task and Solution for testing
#     class MockTask(Task):
#         def __init__(self, task_id, description):
#             self.id = task_id
#             self.description = description
#             self.data = {}
#             self.expected_solution_type = "text"
    
#     class MockSolution(Solution):
#         def __init__(self, task_id, raw_answer):
#             super().__init__(task_id=task_id, solved_by="mock_solver", raw_answer=raw_answer)


#     mock_llm = MockEvaluatorLLM(model_path_or_name="mock_eval_llm")
#     evaluator = LLMQualitativeEvaluator(llm=mock_llm)

#     test_task = MockTask(task_id="eval_task_01", description="Calculate 2 + 2")
#     test_solution = MockSolution(task_id="eval_task_01", raw_answer="4")

#     print("--- Assessing Task Quality ---")
#     task_fb = evaluator.assess_task_quality(test_task)
#     if task_fb:
#         print(f"  Assessment: {task_fb.assessment}")
#         print(f"  Reasoning: {task_fb.reasoning}")
#         print(f"  Raw LLM: {task_fb.raw_llm_response}")

#     print("\n--- Assessing Solution Plausibility ---")
#     solution_fb = evaluator.assess_solution_plausibility(test_task, test_solution)
#     if solution_fb:
#         print(f"  Assessment: {solution_fb.assessment}")
#         print(f"  Reasoning: {solution_fb.reasoning}")
#         print(f"  Raw LLM: {solution_fb.raw_llm_response}")

#     # Test with a more complex response that might be harder to parse
#     class MockLLMComplexResponse(BaseLLM):
#         def _load_model(self): pass
#         def generate(self, prompt: str, **kwargs) -> LLMResponse:
#             return LLMResponse("Assessment: This is a Test. Justification: Because I said so. And other things.")
#         def batch_generate(self, prompts: list[str], **kwargs) -> list[LLMResponse]: pass
    
#     evaluator_complex = LLMQualitativeEvaluator(llm=MockLLMComplexResponse("mock"))
#     task_fb_complex = evaluator_complex.assess_task_quality(test_task)
#     print("\n--- Assessing Task Quality (Complex Response) ---")
#     if task_fb_complex:
#         print(f"  Assessment: {task_fb_complex.assessment}")
#         print(f"  Reasoning: {task_fb_complex.reasoning}") 