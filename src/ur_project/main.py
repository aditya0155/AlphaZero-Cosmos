# src/ur_project/main.py

import logging
import argparse
import os # For ARC data path

from ur_project.core.proposer import SimpleArithmeticProposer
from ur_project.core.solver import LLMSimpleArithmeticSolver
from ur_project.core.verifier import SimpleArithmeticVerifier, SimpleArithmeticTask # Task type for verifier
from ur_project.core.reward_model import SimpleBinaryRewardModel
from ur_project.core.foundational_llm import HuggingFaceLLM # For actual LLM use
from ur_project.core.evaluator import LLMQualitativeEvaluator # New import
from ur_project.pipeline.az_loop import AZLoop, MockLLMForLoop # MockLLM for default/testing
# from ur_project.utils.config import load_config # To be implemented

# ARC specific components
from ur_project.core.arc_proposer import ARCProposer
from ur_project.core.arc_solver import ARCSolver
from ur_project.core.arc_verifier import ARCVerifier
from ur_project.core.perception import BasicARCFeatureExtractor # For ARCSolver
# from ur_project.utils.config import load_config # To be implemented

# Setup basic logging (if not already set by AZLoop or other modules)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Universal Reasoner (UR) Main Execution Script")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of AZ loop steps to run.")
    parser.add_argument("--llm_model_path", type=str, default=None, help="Path or HuggingFace ID for the foundational LLM. If None, uses MockLLM.")
    parser.add_argument("--task_type", type=str, default="arithmetic", choices=["arithmetic", "arc"], help="Type of tasks for the AZ loop.")
    parser.add_argument("--arc_data_dir", type=str, default="data/arc/training", help="Path to ARC data directory (used if task_type is 'arc').")
    # Add arguments for Proposer, Solver, Verifier, RewardModel types later if needed
    # parser.add_argument("--config_file", type=str, default="configs/base_config.yaml", help="Path to the configuration file.")

    args = parser.parse_args()

    # config = load_config(args.config_file) # Load configuration
    # For now, directly initialize components based on args or defaults

    logging.info(f"Starting Universal Reasoner with {args.num_steps} steps for task type: {args.task_type}.")

    # --- Initialize Components ---
    # Common components
    reward_model = SimpleBinaryRewardModel()
    evaluator_instance = None
    llm = None

    if args.llm_model_path:
        logging.info(f"Using HuggingFaceLLM with model: {args.llm_model_path}")
        # llm_config = config.get("llm_config", {}) # Get from loaded config
        llm_config = {
            "device_map": "auto",
            "torch_dtype": "auto", # Use "auto" for transformers to infer, suitable for AWQ models
            "trust_remote_code": True, # Often needed for custom model code
            # If you were using BitsAndBytes for a non-AWQ model, you'd set quantization_config_bnb here:
            # "quantization_config_bnb": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        }
        # Potentially load more specific llm_config from a config file later
        # For example, if base_config.yaml has an llm section:
        # loaded_llm_cfg = get_config_value(config, "llm.config_args", {})
        # llm_config.update(loaded_llm_cfg) # Merge/override defaults

        try:
            llm = HuggingFaceLLM(model_path_or_name=args.llm_model_path, config=llm_config)
            evaluator_instance = LLMQualitativeEvaluator(llm=llm, evaluator_id="gemma_3_27b_evaluator") # Instantiate evaluator with the loaded LLM
        except Exception as e:
            logging.error(f"Failed to load HuggingFaceLLM with model {args.llm_model_path}: {e}")
            logging.error("Falling back to MockLLM for solver, and no LLM evaluator.")
            llm = MockLLMForLoop(model_path_or_name="mock_fallback_llm_solver")
            # Optionally, could use a mock evaluator here too if desired
            # mock_eval_llm = MockLLMForLoop(model_path_or_name="mock_fallback_llm_evaluator")
            # evaluator_instance = LLMQualitativeEvaluator(llm=mock_eval_llm, evaluator_id="mock_evaluator")
    else:
        logging.info("No LLM model path provided. Using MockLLM for solver, and no LLM evaluator.")
        llm = MockLLMForLoop(model_path_or_name="mock_default_llm_solver")
        # Optionally, could use a mock evaluator here too
        # mock_eval_llm = MockLLMForLoop(model_path_or_name="mock_default_llm_evaluator")
        # evaluator_instance = LLMQualitativeEvaluator(llm=mock_eval_llm, evaluator_id="mock_evaluator")
    
    # Task-specific components
    if args.task_type == "arithmetic":
        proposer = SimpleArithmeticProposer(max_number=100)
        if llm is None: # If no real LLM, use mock for arithmetic solver
            llm = MockLLMForLoop(model_path_or_name="mock_arith_solver_llm")
        solver = LLMSimpleArithmeticSolver(llm=llm)
        verifier = SimpleArithmeticVerifier()
        # setattr(verifier, 'task_type_expected', SimpleArithmeticTask) # No longer needed
    elif args.task_type == "arc":
        if not os.path.isdir(args.arc_data_dir) or not os.listdir(args.arc_data_dir):
            logging.error(f"ARC data directory {args.arc_data_dir} is empty or not found. Please provide valid ARC data for --task_type arc.")
            logging.error("You can download and prepare ARC data by running 'python scripts/download_arc.py' from the project root,")
            logging.error("then ensure data/arc/training or data/arc/evaluation contains .json files.")
            return
        
        proposer = ARCProposer(arc_tasks_directory=args.arc_data_dir)
        if llm is None: # If no real LLM, use mock for ARC solver
            # MockLLMForLoop in az_loop.py has been updated to crudely handle ARC prompts
            llm = MockLLMForLoop(model_path_or_name="mock_arc_solver_llm")
        
        feature_extractor = BasicARCFeatureExtractor()
        solver = ARCSolver(llm=llm, feature_extractor=feature_extractor)
        verifier = ARCVerifier()
        logging.info(f"Initialized ARC components using data from: {args.arc_data_dir}")
    else:
        logging.error(f"Unsupported task type: {args.task_type}")
        return
    
    # If evaluator was not set up with a real LLM but task needs it, ensure it uses the task-specific mock LLM
    if evaluator_instance is None and llm is not None: # llm here would be the one for the solver
        if args.llm_model_path is None: # Only create mock evaluator if no real LLM was specified at all
            logging.info(f"Using the solver's MockLLM for the LLMQualitativeEvaluator for task type {args.task_type}.")
            evaluator_instance = LLMQualitativeEvaluator(llm=llm, evaluator_id=f"mock_evaluator_for_{args.task_type}")
        # If a real LLM path was given, evaluator_instance should have been created with it.
        # If it failed, it remains None, and we won't use an evaluator in that case.

    # --- Initialize AZLoop ---
    az_loop = AZLoop(
        proposer=proposer,
        solver=solver,
        verifier=verifier,
        reward_model=reward_model,
        evaluator=evaluator_instance # Pass the evaluator to the loop
    )

    # --- Run Loop ---
    az_loop.run_loop(num_steps=args.num_steps)

    logging.info("\n--- Final Loop History (Summary) ---")
    for i, record in enumerate(az_loop.history[-min(5, len(az_loop.history)):]): # Show last 5 records
        logging.info(f"Record {i+1} (Step {record['step_number']}): Task='{record['task_description']}', ParsedSol='{record['solution_parsed']}', Correct={record['verification_correct']}, Reward={record['reward_value']}")

    logging.info("Universal Reasoner run complete.")

if __name__ == "__main__":
    main() 