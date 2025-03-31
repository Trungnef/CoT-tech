"""
Model loader for loading and optimizing LLMs with efficient multi-GPU usage.
"""

import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from accelerate import dispatch_model, infer_auto_device_map
import google.generativeai as genai
import threading
from functools import lru_cache
from pathlib import Path
import time
import logging
from typing import Dict, Any, Tuple, Optional, List
from threading import RLock
from collections import defaultdict

from .model_config import get_model_config

# Set up logging
logger = logging.getLogger(__name__)

# Global variables
_LOADED_MODELS = {}
_MODEL_LOADING_LOCKS = {}
_CACHE_DIR = Path("./model_cache")
_CACHE_DIR.mkdir(exist_ok=True)

# Model cache settings
_MODEL_CACHE = {}  # Cached models: {model_key: (model, tokenizer, last_used_time)}
_MODEL_USAGE_COUNT = defaultdict(int)  # Usage count: {model_key: count}
_MODEL_LOADING_LOCKS = defaultdict(RLock)  # Locks for model loading: {model_key: lock}
_MAX_CACHED_MODELS = 3  # Maximum number of models to keep in cache
_MIN_FREE_GPU_MEMORY = 2 * 1024 * 1024 * 1024  # Minimum 2GB free GPU memory to keep

def optimize_memory_config():
    """Optimize memory configuration for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_device_map="auto",
        llm_int8_enable_fp32_cpu_offload=True
    )

def clear_memory():
    """Free GPU and CPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Memory cache cleared")

def check_gpu_memory():
    """Check and display GPU memory information."""
    info = []
    info.append("\nGPU Memory Information:")
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free = total - allocated
        total_memory += total
        
        info.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        info.append(f"  - Total: {total:.2f} GB")
        info.append(f"  - Allocated: {allocated:.2f} GB")
        info.append(f"  - Reserved: {reserved:.2f} GB")
        info.append(f"  - Free: {free:.2f} GB")
    info.append(f"\nTotal GPU Memory: {total_memory:.2f} GB")
    
    return "\n".join(info)

def create_optimal_device_map(model_size_gb=70, system_reserve=2.5, cpu_offload=24):
    """
    Create an optimal device map for model distribution across GPUs.
    
    Args:
        model_size_gb (float): Approximate model size in GB
        system_reserve (float): Amount of GPU memory to reserve for system
        cpu_offload (float): Amount of memory to offload to CPU
        
    Returns:
        dict: Device map configuration
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return "cpu"
    
    # Calculate available memory per GPU
    gpu_memories = []
    for i in range(num_gpus):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        available = total - allocated - system_reserve
        gpu_memories.append(available)
    
    total_available = sum(gpu_memories)
    
    if total_available < model_size_gb:
        logger.warning(f"Total available GPU memory ({total_available:.2f}GB) is less than model size ({model_size_gb}GB)")
        logger.warning("Will use CPU offloading")
        return "auto"
    
    # Create memory map
    memory_map = {}
    for i in range(num_gpus):
        memory_map[f"cuda:{i}"] = f"{int(gpu_memories[i])}GiB"
    memory_map["cpu"] = f"{cpu_offload}GiB"
    
    return memory_map

def build_model_key(model_name: str, gpu_id: int, use_4bit: bool) -> str:
    """
    Build a unique key for model caching.
    
    Args:
        model_name: Name of the model
        gpu_id: GPU ID
        use_4bit: Whether 4-bit quantization is used
        
    Returns:
        str: Unique model key
    """
    return f"{model_name}_{gpu_id}_{use_4bit}"

def get_free_gpu_memory(gpu_id: int) -> int:
    """
    Get the amount of free memory on the specified GPU.
    
    Args:
        gpu_id: GPU ID
        
    Returns:
        int: Free memory in bytes
    """
    if not torch.cuda.is_available():
        return 0
    
    try:
        torch.cuda.empty_cache()  # Clear cache to get accurate free memory
        free_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        free_memory -= torch.cuda.memory_allocated(gpu_id)
        free_memory -= torch.cuda.memory_reserved(gpu_id)
        return free_memory
    except Exception as e:
        logger.warning(f"Error getting free GPU memory: {e}")
        return 0

def get_or_load_model(
    model_name: str, 
    gpu_id: int, 
    use_4bit: bool = True
) -> Tuple[Any, Any]:
    """
    Get a cached model or load it if not cached.
    
    Args:
        model_name: Name of the model
        gpu_id: GPU ID for model loading
        use_4bit: Whether to use 4-bit quantization
        
    Returns:
        tuple: (tokenizer, model)
    """
    model_key = build_model_key(model_name, gpu_id, use_4bit)
    
    # Get lock for this specific model
    lock = _MODEL_LOADING_LOCKS[model_key]
    
    with lock:
        # Check if model is already loaded
        if model_key in _MODEL_CACHE:
            tokenizer, model, _ = _MODEL_CACHE[model_key]
            logger.info(f"Using cached model: {model_name} on GPU {gpu_id}")
            
            # Update usage statistics
            _MODEL_USAGE_COUNT[model_key] += 1
            _MODEL_CACHE[model_key] = (tokenizer, model, time.time())
            
            return tokenizer, model
        
        # Check if we need to free up cache based on GPU memory
        if torch.cuda.is_available():
            free_memory = get_free_gpu_memory(gpu_id)
            if free_memory < _MIN_FREE_GPU_MEMORY and _MODEL_CACHE:
                cleanup_model_cache(required_memory=_MIN_FREE_GPU_MEMORY)
        
        # Get model and tokenizer paths from config
        model_config = get_model_config(model_name)
        model_path = model_config.get("model_path")
        tokenizer_path = model_config.get("tokenizer_path", model_path)
        
        if not model_path:
            raise ValueError(f"Model path not defined for {model_name}. Check your .env file.")
        
        logger.info(f"Loading model: {model_name} on GPU {gpu_id}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Tokenizer path: {tokenizer_path}")
        
        # Setup device
        device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
            
            # Configure quantization if needed
            quantization_config = None
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            
            # Add model to cache
            _MODEL_CACHE[model_key] = (tokenizer, model, time.time())
            _MODEL_USAGE_COUNT[model_key] = 1
            
            # Clean up cache if necessary
            if len(_MODEL_CACHE) > _MAX_CACHED_MODELS:
                cleanup_model_cache()
            
            return tokenizer, model
        
        except Exception as e:
            logger.error(f"Error loading model {model_name} on GPU {gpu_id}: {str(e)}")
            raise

def cleanup_model_cache(required_memory: int = None) -> None:
    """
    Clean up model cache based on usage patterns and time.
    
    Args:
        required_memory: If specified, try to free at least this much memory
    """
    if not _MODEL_CACHE:
        return
    
    logger.info(f"Cleaning up model cache (current size: {len(_MODEL_CACHE)})")
    
    # If we have a specific memory requirement, remove models until we have enough free memory
    if required_memory is not None and torch.cuda.is_available():
        models_to_remove = []
        
        # Sort models by last used time (oldest first)
        sorted_models = sorted(
            _MODEL_CACHE.items(),
            key=lambda x: x[1][2]  # Sort by last_used_time
        )
        
        for model_key, (_, _, _) in sorted_models:
            # Parse GPU ID from model key
            gpu_id = int(model_key.split('_')[1])
            
            # Check if we have enough free memory now
            if get_free_gpu_memory(gpu_id) >= required_memory:
                break
            
            models_to_remove.append(model_key)
        
        # Remove selected models
        for model_key in models_to_remove:
            logger.info(f"Removing model {model_key} from cache to free GPU memory")
            remove_model_from_cache(model_key)
    else:
        # Otherwise, just remove the oldest or least used model
        least_used_model = min(
            _MODEL_CACHE.items(), 
            key=lambda x: (_MODEL_USAGE_COUNT[x[0]], x[1][2])
        )[0]
        
        logger.info(f"Removing least used model {least_used_model} from cache")
        remove_model_from_cache(least_used_model)

def remove_model_from_cache(model_key: str) -> None:
    """
    Remove a specific model from the cache.
    
    Args:
        model_key: Model key to remove
    """
    if model_key not in _MODEL_CACHE:
        return
    
    tokenizer, model, _ = _MODEL_CACHE[model_key]
    
    # Cleanup model
    try:
        if hasattr(model, 'to'):
            model.to('cpu')
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.warning(f"Error removing model from cache: {e}")
    
    # Remove from cache dictionaries
    _MODEL_CACHE.pop(model_key, None)
    _MODEL_USAGE_COUNT.pop(model_key, None)

def load_gemini_model() -> Any:
    """
    Load and configure Gemini model.
    
    Returns:
        Gemini model
    """
    # Get Gemini configuration
    config = get_model_config("gemini")
    model_name = config.get("model_name", "gemini-1.5-flash")
    api_key = config.get("api_key")
    
    try:
        # Initialize Gemini
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env file")
        
        genai.configure(api_key=api_key)
        
        # Get model
        gemini_model = genai.GenerativeModel(model_name)
        logger.info(f"Gemini model '{model_name}' configured successfully")
        
        return gemini_model
    
    except Exception as e:
        logger.error(f"Error loading Gemini model: {str(e)}")
        raise 