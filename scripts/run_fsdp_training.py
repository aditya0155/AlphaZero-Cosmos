import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2Block # For auto_wrap_policy
import functools # For auto_wrap_policy

# Assuming src.ur_project.core.foundational_llm is in PYTHONPATH
# For this script, let's add it to sys.path if not found
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ur_project.core.foundational_llm import HuggingFaceLLM


def setup_distributed(rank, world_size):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Make sure this port is free
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank}/{world_size} initialized. Using GPU: {torch.cuda.current_device()}")

def cleanup_distributed():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()
    print("Distributed environment cleaned up.")

class MockDataset(Dataset):
    def __init__(self, tokenizer, num_samples=1000, max_length=50):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        # Create some dummy sentences
        self.data = ["This is sample sentence " + str(i) for i in range(num_samples)]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # .squeeze() to remove batch dim added by tokenizer
        return {"input_ids": tokens.input_ids.squeeze(0), "attention_mask": tokens.attention_mask.squeeze(0)}


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        setup_distributed(local_rank, world_size)
    else: # Single GPU/CPU execution
        print("Running in single process mode (no distributed setup).")
        # For CPU debugging if no GPUs
        if not torch.cuda.is_available():
             print("CUDA not available. Running on CPU (FSDP will not be used).")


    # --- Model and Tokenizer Setup ---
    model_name = 'gpt2' # Using a small model for testing
    # Ensure model is downloaded on rank 0 first if necessary (from_pretrained handles this)
    if local_rank == 0:
        print(f"Rank 0 ensuring model '{model_name}' is downloaded...")
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForCausalLM.from_pretrained(model_name)
    if world_size > 1:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: # GPT2 specific
        tokenizer.pad_token = tokenizer.eos_token

    # --- HuggingFaceLLM Instantiation ---
    llm_config = {
        "distributed_strategy": "fsdp",
        # device_map will be effectively ignored/overridden by FSDP's device_id logic
        "device_map": "auto", 
        # FSDP will handle mixed precision. Load model in higher precision first.
        "torch_dtype": "float32", 
        # Make sure a default LR is available if not overridden in train_step
        "default_rl_learning_rate": 1e-5, 
    }
    
    # Instantiate our LLM class.
    # _load_model within HuggingFaceLLM will run. It sets self.distributed_strategy="fsdp",
    # which prevents internal optimizer creation.
    # The actual model and tokenizer it loads will be replaced.
    llm_instance = HuggingFaceLLM(model_path_or_name=model_name, config=llm_config)
    llm_instance.tokenizer = tokenizer # Set our tokenizer

    # Load the underlying Hugging Face model on CPU before FSDP wrapping
    # FSDP will then manage moving parts of it to the local_rank GPU
    model_load_kwargs = {}
    if torch.cuda.is_available(): # FSDP requires CUDA
         # FSDP typically expects the model to be on CPU before wrapping, 
         # or on 'meta' device for very large models (requires from_config then).
         # For 'gpt2', loading on CPU is fine.
        model_load_kwargs['torch_dtype'] = torch.float32 # Load in full precision
    
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)
    
    if torch.cuda.is_available() and world_size > 0 : # FSDP specific section
        # --- FSDP Wrapping ---
        # Define auto_wrap_policy for Transformer blocks
        # For GPT2, the block class is GPT2Block
        gpt2_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={GPT2Block}
        )
        # Mixed precision for FSDP
        mp_policy = MixedPrecision(
            param_dtype=torch.float16, # Parameters stored in float16
            reduce_dtype=torch.float16, # Gradient communication in float16
            buffer_dtype=torch.float16  # Buffers in float16
        )

        fsdp_model = FSDP(
            hf_model,
            auto_wrap_policy=gpt2_auto_wrap_policy,
            mixed_precision=mp_policy, # Enable mixed precision
            device_id=torch.cuda.current_device(), # local_rank
            sharding_strategy="FULL_SHARD", # or HYBRID_SHARD, SHARD_GRAD_OP
            # cpu_offload=CPUOffload(offload_params=True) # If memory is extremely tight
        )
        print(f"Rank {local_rank}: FSDP model wrapped.")
        llm_instance.model = fsdp_model # Assign FSDP-wrapped model to our LLM class instance
    else: # Not using FSDP (e.g. CPU or single GPU without FSDP)
        if torch.cuda.is_available():
            hf_model = hf_model.to(local_rank)
            print(f"Rank {local_rank}: Model moved to GPU {local_rank} (not using FSDP).")
        else:
            print(f"Rank {local_rank}: Model on CPU (not using FSDP).")
        llm_instance.model = hf_model


    # --- Optimizer Setup ---
    # Optimizer must be created *after* model is wrapped with FSDP.
    # Use parameters of the FSDP wrapped model.
    # The learning rate here is a default; it can be overridden per step in get_action_log_probs_and_train_step
    optimizer_lr = llm_config.get("default_rl_learning_rate", 1e-5)
    optimizer = optim.AdamW(llm_instance.model.parameters(), lr=optimizer_lr)
    
    # Pass the externally created optimizer to the HuggingFaceLLM instance
    llm_instance.optimizer = optimizer
    if local_rank == 0:
        print(f"Optimizer created with LR: {optimizer_lr} and assigned to LLM instance.")

    # --- Data Preparation ---
    mock_dataset = MockDataset(tokenizer, num_samples=100, max_length=32) # Smaller for faster testing
    if world_size > 1:
        sampler = DistributedSampler(mock_dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
        dataloader = DataLoader(mock_dataset, batch_size=4, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        dataloader = DataLoader(mock_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    if local_rank == 0:
        print("DataLoader prepared.")

    # --- Training Loop ---
    num_epochs = 2
    # Ensure model is in training mode
    # For FSDP, .train() should be called on the FSDP-wrapped model.
    llm_instance.model.train() 
    if local_rank == 0:
        print("Starting training loop...")

    for epoch in range(num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch) # Important for shuffling with DistributedSampler
        
        for step, batch in enumerate(dataloader):
            # Data is already { "input_ids": tensor, "attention_mask": tensor }
            # Move batch to device if not using FSDP for model inputs
            # FSDP handles input placement automatically based on model sharding
            # However, explicit .to(local_rank) for inputs is generally safe and clear.
            
            # For this mock training, we need `sequence_log_probs` and `rewards`.
            # Let's mock them. `sequence_log_probs` should conceptually come from a forward pass.
            # To make them require gradients if they were real, we'd do something like:
            #   outputs = llm_instance.model(input_ids, attention_mask, labels=...)
            #   log_probs = calculate_log_probs_from_logits(outputs.logits, labels) 
            # For now, a simple random tensor that requires grad.
            current_batch_size = batch['input_ids'].size(0)
            
            # Mock sequence_log_probs (these would normally be output of a model's forward pass)
            # Ensure requires_grad=True so that loss.backward() has something to work on.
            # This is a key simplification for this script to get the FSDP mechanics working.
            mock_sequence_log_probs = torch.randn(
                current_batch_size, 
                device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu", 
                requires_grad=True
            )
            
            # Mock rewards
            mock_rewards = torch.randn(
                current_batch_size, 
                device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            )

            # RL Step (Backward Pass & Optimization)
            # The learning_rate can be optionally passed here to override the optimizer's default
            # If llm_instance.optimizer is set (which it is here), 
            # get_action_log_probs_and_train_step will use it.
            train_info = llm_instance.get_action_log_probs_and_train_step(
                sequence_log_probs=mock_sequence_log_probs,
                rewards=mock_rewards
                # learning_rate=0.00001 # Optionally override LR for this step
            )

            if local_rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}, "
                      f"Loss: {train_info['loss']:.4f}, Mean Reward: {train_info['mean_reward']:.4f}, "
                      f"Updated by: {train_info['updated_by']}")

            if step >= 10: # Limiting steps per epoch for quick testing
                if local_rank == 0: print("Reached step limit for this epoch.")
                break 
        
        if local_rank == 0: print(f"Epoch {epoch+1} completed.")


    # --- Cleanup ---
    if world_size > 1:
        cleanup_distributed()
    
    if local_rank == 0:
        print("Training script finished.")

if __name__ == "__main__":
    # WORLD_SIZE and LOCAL_RANK are expected to be set by torchrun
    # Example: torchrun --nproc_per_node=2 scripts/run_fsdp_training.py
    main()
