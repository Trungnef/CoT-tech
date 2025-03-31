"""
Model management functionality for loading models and generating text.
"""

from .model_loader import (
    get_or_load_model,
    load_gemini_model,
    clear_memory,
    check_gpu_memory,
    create_optimal_device_map
)

from .text_generation import generate_text_with_model
from .model_config import MODEL_CONFIGS

__all__ = [
    'get_or_load_model',
    'load_gemini_model',
    'clear_memory',
    'check_gpu_memory',
    'create_optimal_device_map',
    'generate_text_with_model',
    'MODEL_CONFIGS'
]
