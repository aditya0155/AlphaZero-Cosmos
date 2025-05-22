import json
import os
import shutil
import logging
from typing import Dict, Any, Optional, List

import torch # Required by symbol_grounding and perception even if not directly used here for tensors

# Setup basic logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow direct import of ur_project modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ur_project.core.foundational_llm import BaseLLM, LLMResponse
from ur_project.data_processing.arc_types import ARCPuzzle, ARCGrid, ARCPixel
from ur_project.core.perception import UNetARCFeatureExtractor, GridFeatures, ARCObject
from ur_project.core.symbol_grounding import SymbolGroundingModel
from ur_project.core.symbolic_learner import SimpleRuleInducer
from ur_project.core.arc_solver import ARCSolver
from ur_project.core.knowledge_base import ARCKnowledgeBase # For inspection

# --- Mock LLM for testing ---
class CapturingMockLLM(BaseLLM):
    def __init__(self, model_path_or_name: str = "capturing_mock_llm", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_path_or_name, config)
        self.last_prompt: Optional[str] = None
        self.response_map: Dict[str, str] = {} # To map parts of prompt to responses

    def _load_model(self):
        logging.info(f"CapturingMockLLM ({self.model_path_or_name}): Model loaded (mock).")

    def set_response_for_prompt_containing(self, substring: str, response_text: str):
        self.response_map[substring] = response_text
    
    def generate(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.1, **kwargs) -> LLMResponse:
        self.last_prompt = prompt
        logging.info(f"CapturingMockLLM received prompt (first 200 chars): {prompt[:200]}...")

        for key_substring, response in self.response_map.items():
            if key_substring in prompt:
                logging.info(f"CapturingMockLLM: Found key '{key_substring}', returning mapped response.")
                return LLMResponse(text=response, metadata={"prompt_used": prompt})
        
        # Default response if no mapping found
        default_response_text = (
            "THOUGHTS:\nLLM reasoning based on the provided rich prompt.\n"
            "HYPOTHESIZED_TRANSFORMATIONS_OR_RULES:\nSome transformation hypothesis.\n"
            "DSL_PROGRAM:\n[NONE]\n" # Default to no DSL to avoid parsing errors in test
            "OUTPUT_GRID:\n[[0]]" # Default fallback grid
        )
        logging.info("CapturingMockLLM: No specific mapping found, returning default response.")
        return LLMResponse(text=default_response_text, metadata={"prompt_used": prompt})

    def get_action_log_probs_and_train_step(self, sequence_log_probs: torch.Tensor, rewards: torch.Tensor, learning_rate: Optional[float] = None) -> Dict[str, Any]:
        # This is the newer signature for RL-fine-tuning, compatible with foundational_llm.py
        # For this test, we don't need to implement actual RL logic.
        return {"loss": 0.0, "mean_reward": rewards.mean().item() if rewards.numel() > 0 else 0.0, "updated_by": "mock_rl_step"}


def create_arc_grid(raw_grid: List[List[int]]) -> ARCGrid:
    return [[ARCPixel(p) for p in row] for row in raw_grid]

def test_phase3_integration_flow():
    logging.info("--- Starting Phase 3 Integration Test ---")

    # 1. Setup Mock LLM
    mock_llm = CapturingMockLLM()
    # Define some mock responses if needed, e.g., for specific prompts
    mock_llm.set_response_for_prompt_containing(
        "Input Grid (raw, list of lists of integers):\n[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", # Training input
        ("THOUGHTS:\nThe input is a diagonal line of color 1. The output is a diagonal line of color 2. This suggests a color change.\n"
         "HYPOTHESIZED_TRANSFORMATIONS_OR_RULES:\nChange color of all objects from 1 to 2.\n"
         "DSL_PROGRAM:\n[{\"operation_name\": \"ChangeColor\", \"selector\": {\"criteria\": {\"color\": 1}}, \"new_color\": {\"value\": 2}}]\n"
         "OUTPUT_GRID:\n[[2, 0, 0], [0, 2, 0], [0, 0, 2]]")
    )
    mock_llm.set_response_for_prompt_containing(
        "Input Grid (raw, list of lists of integers):\n[[3, 0, 0], [0, 3, 0], [0, 0, 3]]", # Test input
        ("THOUGHTS:\nThe input is a diagonal line of color 3. Based on induced rules, objects of one color change to another specific color. "
         "If a rule like 'color 1 -> color 2' was learned, and if this task implies a similar pattern (e.g. color X -> color Y), then color 3 might change to color 4.\n"
         "HYPOTHESIZED_TRANSFORMATIONS_OR_RULES:\nChange color of all objects from 3 to 4 (hypothetical extension of learned pattern).\n"
         "DSL_PROGRAM:\n[{\"operation_name\": \"ChangeColor\", \"selector\": {\"criteria\": {\"color\": 3}}, \"new_color\": {\"value\": 4}}]\n"
         "OUTPUT_GRID:\n[[4, 0, 0], [0, 4, 0], [0, 0, 4]]")
    )


    # 2. Setup Components
    # use_unet_preprocessing=False means U-Net arch is there but won't run if no model path,
    # but UNetARCFeatureExtractor still provides advanced algorithmic features.
    feature_extractor = UNetARCFeatureExtractor(use_unet_preprocessing=False)
    symbol_grounder = SymbolGroundingModel()
    rule_inducer = SimpleRuleInducer()
    
    arc_solver = ARCSolver(
        llm=mock_llm,
        feature_extractor=feature_extractor,
        symbol_grounder=symbol_grounder,
        rule_inducer=rule_inducer
    )
    kb_instance = arc_solver.kb # Get reference to the solver's KB for inspection

    # 3. Define Dummy ARC Task Data
    task_id = "test_task_phase3"
    
    # Training Pair
    train_example_id = "train_0"
    train_input_grid_raw = [[1,0,0],[0,1,0],[0,0,1]]
    train_output_grid_raw = [[2,0,0],[0,2,0],[0,0,2]]
    training_puzzle = ARCPuzzle(
        id=f"{task_id}_{train_example_id}",
        description="Training example: diagonal line color change 1 to 2.",
        data=create_arc_grid(train_input_grid_raw),
        expected_output_grid=create_arc_grid(train_output_grid_raw),
        source_task_id=task_id,
        source_pair_id=train_example_id
    )

    # Test Pair
    test_example_id = "test_0"
    test_input_grid_raw = [[3,0,0],[0,3,0],[0,0,3]]
    test_puzzle = ARCPuzzle(
        id=f"{task_id}_{test_example_id}",
        description="Test example: diagonal line of 3s.",
        data=create_arc_grid(test_input_grid_raw),
        expected_output_grid=None, # No output for test, LLM should predict
        source_task_id=task_id,
        source_pair_id=test_example_id
    )

    # --- 4. Process Training Example ---
    logging.info(f"\n--- Processing Training Example: {training_puzzle.id} ---")
    train_solution = arc_solver.solve_task(training_puzzle)
    
    # Log/Assert training phase outputs
    logging.info("KB state after training example:")
    train_input_context = [task_id, train_example_id, "input"]
    train_entities_input = kb_instance.query_entities(context_tags=train_input_context)
    logging.info(f"  Number of entities for input context '{train_input_context}': {len(train_entities_input)}")
    assert len(train_entities_input) > 0 # Should have grid entity + object entities

    # Sample of rich features (check one object)
    if train_entities_input:
        first_obj_entity = next((e for e in train_entities_input if e.entity_type == "arc_object"), None)
        if first_obj_entity:
            logging.info(f"  Sample rich features for entity {first_obj_entity.id}:")
            for prop in first_obj_entity.properties:
                if "symmetry_" in prop.name or "shape_signature" in prop.name:
                     logging.info(f"    - {prop.name}: {prop.value.value}")

    # Sample of grounded symbols
    logging.info("  Sample grounded symbols (from KB relationships for input context):")
    grounded_rels_train = kb_instance.query_relationships(rel_type="grounded_symbolic_predicate", context_tags=train_input_context)
    for rel in grounded_rels_train[:2]: # Log first 2
        prob = rel.attributes.get("probability", SymbolicValue("float", 0.0)).value
        logging.info(f"    - {rel.subject.id} {rel.name} {rel.object.id if rel.object else ''} (Prob: {prob:.2f})")
    if not grounded_rels_train: logging.info("    - No grounded symbolic relationships found for input context.")
    
    # Induced rules
    # Rules are tagged [task_id, example_id, "induced", type] and also [task_id, "induced_rule_pool"] if SimpleRuleInducer is updated
    # For now, let's query by [task_id, train_example_id, "induced"]
    induced_rules_for_train_example = kb_instance.get_rules(context_tags=[task_id, train_example_id, "induced"])
    logging.info(f"  Number of rules induced from {train_example_id}: {len(induced_rules_for_train_example)}")
    assert len(induced_rules_for_train_example) > 0
    for rule in induced_rules_for_train_example[:1]: # Log first rule
        logging.info(f"    - Sample Induced Rule ID: {rule.id}, Desc: {rule.description}")

    # Captured prompt for training example
    training_prompt = train_solution.metadata.get("prompt_used", "Prompt not captured in training solution metadata.")
    logging.info(f"\n--- Prompt for Training Example ({training_puzzle.id}) (first 500 chars) ---")
    logging.info(training_prompt[:500] + "...\n")


    # --- 5. Process Test Example ---
    logging.info(f"\n--- Processing Test Example: {test_puzzle.id} ---")
    # Before solving test, ensure KB isn't cleared of task-general induced rules.
    # ARCSolver's solve_task clears context [current_task_id, current_example_id].
    # Rules from "train_0" were tagged [task_id, "train_0", "induced", type].
    # If SimpleRuleInducer also adds a general task tag like [task_id, "task_rule_pool"],
    # these rules would persist across example-specific context clearing.
    # For this test, let's assume rules from train_0 are available for test_0 if they share task_id.
    # The current rule query in ARCSolver is: get_rules(context_tags=[current_task_id, "induced_rule_pool"])
    # This means SimpleRuleInducer needs to add this tag.
    # Let's manually add this tag to the rules induced from training for this test to work.
    for rule in induced_rules_for_train_example:
        if "induced_rule_pool" not in rule.context_tags: # Avoid duplicate tags
             rule.context_tags.append("induced_rule_pool") 
             # kb.add_rule(rule) # Re-add to update if needed, but list modification might be enough if objects are refs

    test_solution = arc_solver.solve_task(test_puzzle)

    # Log/Assert test phase outputs
    test_prompt = test_solution.metadata.get("prompt_used", "Prompt not captured in test solution metadata.")
    logging.info(f"\n--- Prompt for Test Example ({test_puzzle.id}) (first 700 chars) ---")
    logging.info(test_prompt[:700] + "...\n")
    
    # Check if induced rules were in the prompt for the test example
    rule_segment_in_test_prompt = "Potentially Relevant Induced Rules"
    assert rule_segment_in_test_prompt in test_prompt, f"'{rule_segment_in_test_prompt}' segment not found in test prompt."
    # Check if a specific rule detail appears (based on the dummy rule induced)
    # Example: "If ?subject_1 has color=ARCPixel(1)"
    assert "color=ARCPixel(1)" in test_prompt or "color=Color(1)" in test_prompt, "Detail from induced rule not found in test prompt."

    logging.info(f"  LLM Response for Test Example (parsed grid): {test_solution.parsed_answer}")
    # Example assertion based on mock LLM behavior
    expected_test_output = create_arc_grid([[4,0,0],[0,4,0],[0,0,4]])
    assert test_solution.parsed_answer == expected_test_output, \
        f"Test output {test_solution.parsed_answer} did not match expected {expected_test_output}"

    logging.info("--- Phase 3 Integration Test Completed Successfully ---")

if __name__ == "__main__":
    # Create a dummy model path for UNet if it checks for it, even if not loading weights
    dummy_model_dir = "temp_test_models"
    os.makedirs(dummy_model_dir, exist_ok=True)
    dummy_unet_path = os.path.join(dummy_model_dir, "dummy_unet.pth")
    # Create an empty file as a placeholder model state dict
    torch.save({}, dummy_unet_path)
    
    # Run the test
    try:
        test_phase3_integration_flow()
    finally:
        # Clean up
        if os.path.exists(dummy_model_dir):
            shutil.rmtree(dummy_model_dir)

```
**Note on `SimpleRuleInducer` Tagging:**
The test script assumes that rules induced from training examples for a task are tagged in a way that they can be retrieved when processing a test example of the *same task*. The current `SimpleRuleInducer` tags rules like `[task_id, example_id, "induced", type]`.
The `ARCSolver`'s prompt formatting logic queries for rules using `context_tags=[current_task_id, "induced_rule_pool"]`.
To bridge this, the test script manually adds the `"induced_rule_pool"` tag to rules for the sake of the test.
A proper fix would be to update `SimpleRuleInducer` to add this more general task-level tag to all rules it induces, e.g., `rule.context_tags.append("induced_rule_pool")` in addition to the example-specific tags. I will make this small change to `SimpleRuleInducer`.
