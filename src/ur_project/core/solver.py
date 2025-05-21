from typing import Any, Dict, Protocol, Optional
from ur_project.core.foundational_llm import BaseLLM, LLMResponse
from ur_project.core.proposer import Task # Using the Task protocol
from .arc_dsl import DSLProgram # Import DSLProgram for type hinting

class Solution:
    def __init__(self, task_id: str, solved_by: str, raw_answer: Any, parsed_answer: Any = None, metadata: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.solved_by = solved_by
        self.raw_answer = raw_answer # Raw LLM response for the first attempt
        
        # Fields for the initial DSL attempt
        self.hypothesized_transformations: Optional[str] = None
        self.raw_dsl_program: Optional[str] = None
        self.parsed_dsl_program: Optional[DSLProgram] = None
        self.dsl_parse_error: Optional[str] = None
        self.dsl_execution_error: Optional[str] = None
        self.executed_grid: Optional[ARCGrid] = None # Result of 1st successful execution

        # Fields for the retry mechanism
        self.retry_attempted: bool = False
        self.retry_prompt: Optional[str] = None
        self.raw_llm_response_retry: Optional[str] = None # Raw LLM response for the retry
        self.thoughts_retry: Optional[str] = None
        self.raw_dsl_program_retry: Optional[str] = None
        self.parsed_dsl_program_retry: Optional[DSLProgram] = None
        self.dsl_parse_error_retry: Optional[str] = None
        self.dsl_execution_error_retry: Optional[str] = None
        self.executed_grid_retry: Optional[ARCGrid] = None # Result of retry successful execution

        # Final outcome
        self.parsed_answer = parsed_answer # Final successfully executed grid, if any
        
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

# --- ARC Specific Solver ---
from ur_project.data_processing.arc_types import ARCPuzzle, ARCGrid, ARCPixel, ARCPair
from ur_project.core.perception import BaseFeatureExtractor # Assuming this protocol exists
from ur_project.core.knowledge_base import ARCKnowledgeBase

class ARCSolver(BaseSolver):
    """
    Solves ARC tasks by leveraging a foundational LLM, symbolic knowledge base,
    and few-shot prompting.
    """
    def __init__(self, 
                 llm: BaseLLM, 
                 perception_engine: BaseFeatureExtractor, 
                 knowledge_base: ARCKnowledgeBase,
                 solver_id: str = "ARCSolver_v1"):
        self.llm = llm
        self.perception = perception_engine
        self.kb = knowledge_base
        self.solver_id = solver_id
        self.few_shot_examples = self._load_few_shot_examples()
        # Import and instantiate the DSL interpreter
        from .arc_dsl_interpreter import ARCDSLInterpreter, DSLParsingError, DSLExecutionError # Import errors
        self.interpreter = ARCDSLInterpreter()


    def _format_grid_for_prompt(self, grid: ARCGrid, grid_name: str) -> str:
        """Formats an ARCGrid into a string for inclusion in a prompt."""
        if not grid:
            return f"{grid_name}:\n(empty grid)\n"
        
        grid_str = f"{grid_name}:\n"
        for row in grid:
            row_str = " ".join([str(int(pixel)) for pixel in row])
            grid_str += row_str + "\n"
        return grid_str

    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """
        Loads hardcoded few-shot examples.
        Later, this could load from a JSON file.
        """
        # TODO: Populate with 2-3 diverse examples as specified
        # For now, returning a placeholder structure.
        # Ensure ARCPixel is used where grid data is defined, or cast appropriately.
        return [
            {
                "task_name": "example_task_color_change",
                "task_description": "Changes the color of the largest object.",
                "input_grids": [ # Corresponds to ARCTask.training_pairs[0].input_grid
                    [[ARCPixel(1), ARCPixel(0)], [ARCPixel(0), ARCPixel(1)]] 
                ],
                "output_grids": [ # Corresponds to ARCTask.training_pairs[0].output_grid
                    [[ARCPixel(2), ARCPixel(0)], [ARCPixel(0), ARCPixel(2)]]
                ],
                "solution_input_grid": [[ARCPixel(3), ARCPixel(3)], [ARCPixel(0), ARCPixel(3)]], # Test input
                "solution_output_grid": [[ARCPixel(5), ARCPixel(5)], [ARCPixel(0), ARCPixel(5)]], # Test output
                "thoughts": "The task is to change the color of all non-background pixels. In the example, color 1 becomes 2. In the test input, color 3 should become 5.",
                "dsl_program": "CHANGE_COLOR(source_color=3, target_color=5)"
            },
            # Add 1-2 more diverse examples here
        ]

    def solve_task(self, task: ARCPuzzle) -> Solution:
        """
        Solves a given ARC puzzle.
        - Populates the Knowledge Base (KB) with features from the task's grids.
        - Constructs a prompt with few-shot examples, KB insights, and task data.
        - Calls the LLM to get a DSL program and thoughts.
        - Parses the response into a Solution object.
        """
        self.kb.clear_task_context() # Clear KB from previous task

        # --- 1. Populate Knowledge Base ---
        # For an ARCPuzzle, `task.data` is the input grid for the current test instance.
        # `task.source_task_id` and `task.source_pair_id` can be used if we need to
        # refer back to the original ARCTask and its training pairs for more context,
        # but the primary input for *this specific puzzle* is task.data.

        # Process the main input grid for the puzzle
        input_grid_features = self.perception.extract_features(task.data, "input_grid_main")
        if input_grid_features:
            grid_entity_id = f"{task.id}_input_grid"
            self.kb.add_grid_features_as_entity(input_grid_features, grid_id=grid_entity_id)
            for arc_obj in input_grid_features.objects:
                self.kb.add_arc_object_as_entity(arc_obj, grid_context=grid_entity_id)
        
        # TODO: If ARCPuzzle includes training pairs, process them as well for the KB
        # For now, focusing on the direct input grid `task.data`.
        # Example: if task.metadata and 'training_pairs' in task.metadata:
        #   for i, pair_data in enumerate(task.metadata['training_pairs']):
        #       train_input_feats = self.perception.extract_features(pair_data['input'], f"train_input_{i}")
        #       if train_input_feats: ... add to KB ...
        #       train_output_feats = self.perception.extract_features(pair_data['output'], f"train_output_{i}")
        #       if train_output_feats: ... add to KB ...

        # Infer relationships
        # TODO: Make relation names constants or configurable
        self.kb.infer_spatial_relationships(relation_name="is_left_of") 
        self.kb.infer_spatial_relationships(relation_name="is_above")
        # Add other relevant inferences if available in KB

        # --- 2. Format Symbolic Information for Prompt ---
        kb_object_props = self.kb.get_all_object_properties_for_prompt()
        kb_relationships = self.kb.get_all_relationships_for_prompt()
        
        kb_insights_str = "KNOWLEDGE_BASE_INSIGHTS:\n"
        if kb_object_props:
            kb_insights_str += "\n".join(kb_object_props) + "\n"
        else:
            kb_insights_str += "No object properties identified.\n"
        if kb_relationships:
            kb_insights_str += "\n".join(kb_relationships) + "\n"
        else:
            kb_insights_str += "No relationships identified.\n"

        # --- 3. Format Few-Shot Examples ---
        few_shot_prompt_parts = ["Here are some examples of how to solve ARC tasks:\n"]
        for i, ex in enumerate(self.few_shot_examples):
            example_str = f"Example {i+1}: {ex['task_name']}\n"
            example_str += f"Task Description: {ex['task_description']}\n"
            
            # Format training pair grids for the example
            if ex.get("input_grids") and ex.get("output_grids"):
                example_str += "Input Grid(s) for demonstration:\n"
                for j, ig in enumerate(ex["input_grids"]):
                    example_str += self._format_grid_for_prompt(ARCGrid(ig), f"example_{i+1}_train_input_{j}")
                example_str += "Output Grid(s) for demonstration:\n"
                for j, og in enumerate(ex["output_grids"]):
                    example_str += self._format_grid_for_prompt(ARCGrid(og), f"example_{i+1}_train_output_{j}")
            
            example_str += "\nFor the following input:\n"
            example_str += self._format_grid_for_prompt(ARCGrid(ex["solution_input_grid"]), f"example_{i+1}_solution_input")
            example_str += "The desired output is:\n"
            example_str += self._format_grid_for_prompt(ARCGrid(ex["solution_output_grid"]), f"example_{i+1}_solution_output")
            
            example_str += f"\nTHOUGHTS:\n{ex['thoughts']}\n"
            example_str += f"DSL_PROGRAM:\n{ex['dsl_program']}\n---\n"
            few_shot_prompt_parts.append(example_str)
        
        few_shot_section = "".join(few_shot_prompt_parts)

        # --- 4. Construct the Main Task Prompt ---
        # The ARCPuzzle contains the specific test input grid in `task.data`
        # and the expected output in `task.expected_output_grid`.
        
        current_task_prompt_parts = ["Now, solve the following ARC task:\n"]
        current_task_prompt_parts.append(f"Task ID: {task.id}\n")
        if task.text_description: # Optional textual description from the source JSON
             current_task_prompt_parts.append(f"Task Description from source: {task.text_description}\n")

        # Display training pairs if they are part of the ARCPuzzle structure
        # (Assuming ARCPuzzle might be extended to hold its parent ARCTask's train pairs)
        if hasattr(task, 'source_task_training_pairs') and task.source_task_training_pairs:
            current_task_prompt_parts.append("Training Pairs (Input -> Output):\n")
            for i, pair_info in enumerate(task.source_task_training_pairs): # pair_info is ARCPair
                current_task_prompt_parts.append(f"Training Pair {i+1} Input:\n")
                current_task_prompt_parts.append(self._format_grid_for_prompt(pair_info.input_grid, f"train_input_{i}"))
                current_task_prompt_parts.append(f"Training Pair {i+1} Output:\n")
                current_task_prompt_parts.append(self._format_grid_for_prompt(pair_info.output_grid, f"train_output_{i}"))
        
        current_task_prompt_parts.append("Current Test Input Grid:\n")
        current_task_prompt_parts.append(self._format_grid_for_prompt(task.data, "test_input_grid"))
        current_task_prompt_parts.append("Expected Test Output Grid:\n")
        current_task_prompt_parts.append(self._format_grid_for_prompt(task.expected_output_grid, "test_output_grid"))

        current_task_prompt_parts.append("\nBased on the examples, the knowledge base insights, and the task grids, provide your reasoning and the DSL program to solve this task.\n")
        current_task_prompt_parts.append("THOUGHTS:\n(Provide a step-by-step thought process for how to arrive at the solution. Analyze the input and output grids, identify patterns, and explain the transformations needed. Refer to KB insights if they are helpful.)\n")
        current_task_prompt_parts.append("DSL_PROGRAM:\n(Provide the DSL program that transforms the input grid to the output grid. Use one DSL command per line. Refer to the DSL documentation for available commands and syntax.)\n")
        
        main_task_section = "".join(current_task_prompt_parts)

        # --- 5. Assemble Final Prompt ---
        final_prompt = few_shot_section + "\n" + kb_insights_str + "\n" + main_task_section
        
        # --- 6. Call LLM ---
        # print(f"DEBUG: Final Prompt for LLM:\n{final_prompt}") # For debugging
        
        llm_response: LLMResponse = self.llm.generate(
            final_prompt,
            # Adjust max_new_tokens based on expected DSL length and thoughts
            max_new_tokens=1024, 
            temperature=0.1 # Lower temperature for more deterministic DSL generation
        )
        
        raw_answer_text = llm_response.text.strip()
        
        # --- 7. Parse LLM Response and Create Solution ---
        # Basic parsing: split by "THOUGHTS:" and "DSL_PROGRAM:"
        thoughts_content = ""
        dsl_content = ""

        if "DSL_PROGRAM:" in raw_answer_text:
            parts = raw_answer_text.split("DSL_PROGRAM:", 1)
            thoughts_part = parts[0]
            dsl_content = parts[1].strip()
            if "THOUGHTS:" in thoughts_part:
                thoughts_content = thoughts_part.split("THOUGHTS:", 1)[1].strip()
            else: # If THOUGHTS: marker is missing but DSL_PROGRAM: is present
                thoughts_content = thoughts_part.strip()
        elif "THOUGHTS:" in raw_answer_text: # Only THOUGHTS found
             thoughts_content = raw_answer_text.split("THOUGHTS:", 1)[1].strip()
        else: # No clear markers, assume the whole output is relevant for one or the other
            # This might need more sophisticated parsing depending on LLM output structure
            # For now, if no DSL_PROGRAM marker, assume it might be all thoughts, or all DSL.
            # Let's tentatively put it in DSL if no markers, as that's the primary goal.
            dsl_content = raw_answer_text 


        # --- 7. Parse LLM Response and Create Solution ---
        solution = Solution(
            task_id=task.id,
            solved_by=self.solver_id,
            raw_answer=raw_answer_text, # Initial LLM response
            metadata=llm_response.metadata
        )
        solution.hypothesized_transformations = thoughts_content
        solution.raw_dsl_program = dsl_content

        # Attempt to parse and execute the initial DSL
        try:
            solution.parsed_dsl_program = self.interpreter.parse_dsl(dsl_content)
            # Execute the parsed program using a copy of the input grid
            # to avoid modifying the original task.data if execution is partial.
            current_grid_for_execution = [row[:] for row in task.data] # Deep copy
            solution.executed_grid = self.interpreter.execute_program(
                solution.parsed_dsl_program, 
                current_grid_for_execution
            )
            solution.parsed_answer = solution.executed_grid # Success on first try
        except DSLParsingError as e_parse:
            solution.dsl_parse_error = str(e_parse)
        except DSLExecutionError as e_exec:
            solution.dsl_execution_error = str(e_exec)
            # Potentially store partially executed grid if interpreter supports/returns it
            # For now, solution.executed_grid would be None or the state before error
        except Exception as e_generic: # Catch any other unexpected errors
            solution.dsl_execution_error = f"Unexpected error during DSL processing: {str(e_generic)}"

        # --- 8. Retry Logic (if parsing or execution failed) ---
        if solution.dsl_parse_error or solution.dsl_execution_error:
            solution.retry_attempted = True
            
            error_message_for_prompt = solution.dsl_parse_error or solution.dsl_execution_error
            
            retry_prompt_parts = [
                "The previous DSL program attempt resulted in an error.\n"
                "Original Task Input was:\n"
            ]
            # Add original task input grids to prompt (test input, and training pairs if used)
            if hasattr(task, 'source_task_training_pairs') and task.source_task_training_pairs:
                retry_prompt_parts.append("Training Pairs (Input -> Output):\n")
                for i, pair_info in enumerate(task.source_task_training_pairs):
                    retry_prompt_parts.append(self._format_grid_for_prompt(pair_info.input_grid, f"train_input_{i}"))
                    retry_prompt_parts.append(self._format_grid_for_prompt(pair_info.output_grid, f"train_output_{i}"))
            retry_prompt_parts.append("Current Test Input Grid:\n")
            retry_prompt_parts.append(self._format_grid_for_prompt(task.data, "test_input_grid"))
            
            retry_prompt_parts.append("\nPrevious DSL Program:\n")
            retry_prompt_parts.append(solution.raw_dsl_program if solution.raw_dsl_program else "No DSL program was generated.")
            retry_prompt_parts.append(f"\n\nError Encountered:\n{error_message_for_prompt}\n")
            retry_prompt_parts.append(kb_insights_str) # Add KB insights again
            retry_prompt_parts.append("\nPlease analyze the error and provide a corrected DSL_PROGRAM.\n")
            retry_prompt_parts.append("Your new THOUGHTS should explain the correction.\n")
            retry_prompt_parts.append("THOUGHTS:\n")
            retry_prompt_parts.append("DSL_PROGRAM:\n")

            solution.retry_prompt = "".join(retry_prompt_parts)

            # Call LLM for retry
            llm_response_retry: LLMResponse = self.llm.generate(
                solution.retry_prompt,
                max_new_tokens=1024,
                temperature=0.1 # Keep temperature low
            )
            solution.raw_llm_response_retry = llm_response_retry.text.strip()

            # Parse retry response
            thoughts_retry_content = ""
            dsl_retry_content = ""
            if "DSL_PROGRAM:" in solution.raw_llm_response_retry:
                parts = solution.raw_llm_response_retry.split("DSL_PROGRAM:", 1)
                thoughts_part_retry = parts[0]
                dsl_retry_content = parts[1].strip()
                if "THOUGHTS:" in thoughts_part_retry:
                    thoughts_retry_content = thoughts_part_retry.split("THOUGHTS:", 1)[1].strip()
                else:
                    thoughts_retry_content = thoughts_part_retry.strip()
            elif "THOUGHTS:" in solution.raw_llm_response_retry:
                 thoughts_retry_content = solution.raw_llm_response_retry.split("THOUGHTS:", 1)[1].strip()
            else:
                dsl_retry_content = solution.raw_llm_response_retry
            
            solution.thoughts_retry = thoughts_retry_content
            solution.raw_dsl_program_retry = dsl_retry_content

            # Attempt to parse and execute the retry DSL
            try:
                solution.parsed_dsl_program_retry = self.interpreter.parse_dsl(dsl_retry_content)
                current_grid_for_execution_retry = [row[:] for row in task.data] # Deep copy
                solution.executed_grid_retry = self.interpreter.execute_program(
                    solution.parsed_dsl_program_retry,
                    current_grid_for_execution_retry
                )
                solution.parsed_answer = solution.executed_grid_retry # Success on retry
                # Clear previous errors if retry was successful
                solution.dsl_parse_error = None
                solution.dsl_execution_error = None
            except DSLParsingError as e_parse_retry:
                solution.dsl_parse_error_retry = str(e_parse_retry)
            except DSLExecutionError as e_exec_retry:
                solution.dsl_execution_error_retry = str(e_exec_retry)
            except Exception as e_generic_retry:
                solution.dsl_execution_error_retry = f"Unexpected error during DSL processing on retry: {str(e_generic_retry)}"
        
        self.kb.clear_task_context()
        return solution

# Example Usage (Conceptual - requires actual instances of LLM, Perception, KB)
# if __name__ == "__main__":
#     # Mock components
#     class MockLLM(BaseLLM):
#         def _load_model(self): pass
#         def generate(self, prompt: str, **kwargs) -> LLMResponse:
#             print("\n--- MockLLM Received Prompt ---")
#             # print(prompt) 
#             print("--- End MockLLM Prompt ---")
#             # Simulate an LLM response for the example task
#             response_text = """THOUGHTS:
# The example task shows changing color 3 to 5. The input grid has objects of color 3.
# So, I need to change all pixels of color 3 to color 5.
# DSL_PROGRAM:
# CHANGE_COLOR(source_color=3, target_color=5)
# """
#             return LLMResponse(text=response_text)
#         def batch_generate(self, prompts: list[str], **kwargs) -> list[LLMResponse]:
#             return [self.generate(p) for p in prompts]

#     from ur_project.core.perception import BasicARCFeatureExtractor
#     from ur_project.data_processing.arc_types import ARCPair

#     mock_llm = MockLLM("mock_model")
#     perception_eng = BasicARCFeatureExtractor()
#     kb_instance = ARCKnowledgeBase()
    
#     arc_solver = ARCSolver(llm=mock_llm, perception_engine=perception_eng, knowledge_base=kb_instance)

#     # Create a dummy ARCPuzzle for testing
#     # This would normally come from a dataset loader / task proposer
#     dummy_train_pair = ARCPair(
#         input_grid=[[ARCPixel(1), ARCPixel(0)], [ARCPixel(0), ARCPixel(1)]],
#         output_grid=[[ARCPixel(2), ARCPixel(0)], [ARCPixel(0), ARCPixel(2)]]
#     )
#     dummy_puzzle = ARCPuzzle(
#         id="dummy_task_01",
#         description="A simple color change task.",
#         data=[[ARCPixel(3), ARCPixel(3)], [ARCPixel(0), ARCPixel(3)]], # Test input
#         expected_output_grid=[[ARCPixel(5), ARCPixel(5)], [ARCPixel(0), ARCPixel(5)]], # Test output
#         source_task_id="dummy_source_task",
#         source_pair_id="test_0",
#         text_description="Change color 3 to 5.",
#         # If your ARCPuzzle includes training pairs for context in the prompt:
#         # metadata={"source_task_training_pairs": [dummy_train_pair]} 
#     )
#     # If your ARCPuzzle might have this structure:
#     # setattr(dummy_puzzle, 'source_task_training_pairs', [dummy_train_pair])


#     solution_obj = arc_solver.solve_task(dummy_puzzle)
#     print(f"\n--- Solver Output for Task: {solution_obj.task_id} ---")
#     print(f"Solved By: {solution_obj.solved_by}")
#     print(f"Thoughts:\n{solution_obj.hypothesized_transformations}")
#     print(f"Raw DSL Program:\n{solution_obj.raw_dsl_program}")
#     print(f"Raw LLM Answer:\n{solution_obj.raw_answer}")
#     # print(f"Parsed DSL Program: {solution_obj.parsed_dsl_program}") # Requires DSL parser
#     print(f"Metadata: {solution_obj.metadata}")
#     print("---")