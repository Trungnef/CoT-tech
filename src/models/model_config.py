"""
Model configuration settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configurations with default parameters
MODEL_CONFIGS = {
    "llama": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "context_window": 4096,
        "model_path": os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Llama-3.3-70B-Instruct"),
        "tokenizer_path": os.environ.get("LLAMA_TOKENIZER_PATH")
    },
    "qwen": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "context_window": 8192,
        "model_path": os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-72B-Instruct"),
        "tokenizer_path": os.environ.get("QWEN_TOKENIZER_PATH")
    },
    "gemini": {
        "max_new_tokens": 2048,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "model_name": "gemini-1.5-flash",
        "api_key": os.environ.get("GEMINI_API_KEY")
    }
}

def get_model_config(model_name: str, param_name: str = None, default=None):
    """
    Get model configuration or a specific parameter.
    
    Args:
        model_name: Name of the model
        param_name: Name of the parameter to get (if None, returns full config)
        default: Default value if parameter not found
        
    Returns:
        Parameter value or full config dict
    """
    model_name = model_name.lower()
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    if param_name is None:
        return MODEL_CONFIGS[model_name]
    
    return MODEL_CONFIGS[model_name].get(param_name, default) 