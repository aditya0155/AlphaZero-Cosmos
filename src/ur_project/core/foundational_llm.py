# src/ur_project/core/foundational_llm.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# A simple dataclass or NamedTuple could be used for LLM responses if they have a common structure.
class LLMResponse:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata if metadata is not None else {}

    def __str__(self):
        return self.text

class BaseLLM(ABC):
    """Abstract Base Class for a foundational Large Language Model."""

    def __init__(self, model_path_or_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the LLM.

        Args:
            model_path_or_name (str): Identifier for the model (e.g., path to local weights or Hugging Face model name).
            config (Optional[Dict[str, Any]]): Model-specific configuration options.
        """
        self.model_path_or_name = model_path_or_name
        self.config = config if config is not None else {}
        self.model = None # Placeholder for the actual loaded model
        self.tokenizer = None # Placeholder for the tokenizer
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Loads the model and tokenizer. Specific to each LLM implementation."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        # Add other common generation parameters as needed
    ) -> LLMResponse:
        """
        Generates text based on a given prompt.

        Args:
            prompt (str): The input prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_k (Optional[int]): Top-k sampling.
            top_p (Optional[float]): Top-p (nucleus) sampling.

        Returns:
            LLMResponse: The generated text and any associated metadata.
        """
        pass

    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        # Add other common generation parameters as needed
    ) -> List[LLMResponse]:
        """
        Generates text for a batch of prompts.

        Args:
            prompts (List[str]): A list of input prompts.
            max_new_tokens (int): Maximum number of new tokens to generate per prompt.
            temperature (float): Sampling temperature.

        Returns:
            List[LLMResponse]: A list of LLMResponse objects, one for each prompt.
        """
        pass

# Example of a concrete implementation (e.g., using Hugging Face Transformers)
# This would require `transformers` and `torch` (or `tensorflow`/`jax`) to be installed.
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HuggingFaceLLM(BaseLLM):
    def _load_model(self):
        print(f"Loading Hugging Face model: {self.model_path_or_name}")
        device_map = self.config.get("device_map", "auto") 
        # For AWQ models loaded with from_pretrained(..., torch_dtype="auto"),
        # a separate bitsandbytes quantization_config is usually not needed.
        # Transformers handles AWQ based on the model's own config files.
        # However, if a BitsAndBytesConfig is explicitly provided, we should use it.
        quantization_config_bnb = self.config.get("quantization_config_bnb", None) # Specific for BitsAndBytes

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_or_name)
        if self.tokenizer.pad_token is None:
             self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_dtype_str = self.config.get("torch_dtype", "float16") # Default to float16 if not auto
        torch_dtype_resolved = torch.float32 # Fallback default

        if model_dtype_str == "auto":
            # Pass "auto" directly to from_pretrained, which is suitable for AWQ models
            # or when Transformers should infer the dtype.
            torch_dtype_resolved = "auto"
        elif model_dtype_str == "float16":
            torch_dtype_resolved = torch.float16
        elif model_dtype_str == "bfloat16":
            torch_dtype_resolved = torch.bfloat16
        elif model_dtype_str == "float32":
            torch_dtype_resolved = torch.float32
        else:
            print(f"Warning: Unrecognized torch_dtype '{model_dtype_str}'. Defaulting to float32.")
            torch_dtype_resolved = torch.float32

        model_load_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype_resolved,
            "trust_remote_code": self.config.get("trust_remote_code", True)
        }
        
        # Only add quantization_config if it's a BitsAndBytes config and explicitly provided
        if quantization_config_bnb:
             model_load_kwargs["quantization_config"] = quantization_config_bnb
             print("Applying BitsAndBytes quantization config.")
        elif torch_dtype_resolved == "auto":
            print(f"Loading model with torch_dtype='{torch_dtype_resolved}'. AWQ quantization (if applicable) will be handled by Transformers.")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path_or_name,
            **model_load_kwargs
        )
        self.model.eval() 
        print(f"Model {self.model_path_or_name} loaded. Main device: {self.model.device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> LLMResponse:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call _load_model() first or ensure successful initialization.")

        # Ensure inputs are on the same device as the model expects, esp. for first layer with device_map
        # For device_map="auto", the inputs should typically be on CPU, and Transformers handles placement.
        # If model is on a single device (e.g. 'cuda:0'), inputs must be moved there.
        # model_device = self.model.device # This might be tricky with device_map='auto'
        # For now, let's assume device_map will handle it or model is on one device and inputs will be moved by .to()
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 1.0, # Temp 0 can be problematic for some samplers
            "pad_token_id": self.tokenizer.pad_token_id
        }
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        
        if temperature <= 0: # Greedy decoding for temperature 0 or less
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # Remove prompt tokens from the output
        # For batch decoding, this needs to be done per item in the batch
        input_token_len = inputs.input_ids.shape[1]
        output_text = self.tokenizer.decode(output_ids[0][input_token_len:], skip_special_tokens=True)
        return LLMResponse(text=output_text)

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = None, 
        top_p: Optional[float] = None, 
    ) -> List[LLMResponse]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded.")

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        if temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            
        with torch.no_grad():
            output_ids_batch = self.model.generate(**inputs, **gen_kwargs)

        responses = []
        for i in range(len(prompts)):
            input_token_len = inputs.input_ids[i].ne(self.tokenizer.pad_token_id).sum().item()
            # Ensure we don't try to decode from a negative slice if output is shorter than input (e.g. error or empty gen)
            # This can happen if max_new_tokens is too small or generation fails.
            actual_output_ids = output_ids_batch[i][input_token_len:]
            output_text = self.tokenizer.decode(actual_output_ids, skip_special_tokens=True)
            responses.append(LLMResponse(text=output_text))
        return responses

# To use this later (after deciding on a model and installing transformers/torch):
# from ur_project.core.foundational_llm import HuggingFaceLLM
# llm_config = {
#     "device_map": "auto", 
#     "torch_dtype": "auto", # Recommended for AWQ models like "gaunernst/gemma-3-27b-it-int4-awq"
#                            # or "float16"/"bfloat16" for other models.
#     "trust_remote_code": True, 
#     # For standard BitsAndBytes 4-bit quantization (NOT for AWQ models loaded with torch_dtype="auto"):
#     # from transformers import BitsAndBytesConfig
#     # "quantization_config_bnb": BitsAndBytesConfig(
#     #     load_in_4bit=True,
#     #     bnb_4bit_compute_dtype=torch.bfloat16 
#     # ), 
# }
# # model_path_or_name should be the local directory where snapshot_download placed the model files,
# # e.g., "./gemma-3-27b-int4" if downloaded to that relative path.
# foundational_model = HuggingFaceLLM(model_path_or_name="./gemma-3-27b-int4", config=llm_config)
# # Or, for a non-quantized Hugging Face Hub model:
# # foundational_model = HuggingFaceLLM(model_path_or_name="google/gemma-2b-it", config={"torch_dtype": "float16", "device_map": "auto"})
# response = foundational_model.generate("Explain the theory of relativity in simple terms.")
# print(response.text) 