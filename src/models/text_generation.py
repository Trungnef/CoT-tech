"""
Text generation utilities for different model types.
"""

import time
import logging
import torch
import google.generativeai as genai
from typing import Dict, Any, Tuple, Optional

from .model_config import get_model_config

# Setup logging
logger = logging.getLogger(__name__)

def generate_text_with_model(
    prompt: str, 
    model_type: str = "local",
    model: Any = None,
    tokenizer: Any = None,
    model_name: str = None,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate text using the specified model.
    
    Args:
        prompt: Input prompt text
        model_type: Type of model ("local" or "gemini")
        model: Model object
        tokenizer: Tokenizer object (for local models)
        model_name: Name of the model for loading config
        **kwargs: Additional parameters to override defaults
        
    Returns:
        tuple: (generated_text, metadata)
    """
    if model is None:
        raise ValueError("Model must be provided")
    
    start_time = time.time()
    
    if model_type == "local":
        response, metadata = generate_with_local_model(prompt, model, tokenizer, model_name, **kwargs)
    elif model_type == "gemini":
        response, metadata = generate_with_gemini(prompt, model, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Calculate time metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Add timing info to metadata
    metadata["elapsed_time"] = elapsed_time
    if "tokens_generated" in metadata:
        metadata["tokens_per_second"] = metadata["tokens_generated"] / elapsed_time if elapsed_time > 0 else 0
    
    return response, metadata

def generate_with_local_model(
    prompt: str,
    model: Any,
    tokenizer: Any,
    model_name: str = None,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate text with a local HuggingFace model.
    
    Args:
        prompt: Input prompt
        model: Model object
        tokenizer: Tokenizer object
        model_name: Name of the model for config
        **kwargs: Override parameters
        
    Returns:
        tuple: (generated_text, metadata)
    """
    if model_name:
        # Get default config for this model
        config = get_model_config(model_name)
        
        # Set defaults if not provided in kwargs
        max_new_tokens = kwargs.get("max_new_tokens", config.get("max_new_tokens", 1024))
        temperature = kwargs.get("temperature", config.get("temperature", 0.7))
        top_p = kwargs.get("top_p", config.get("top_p", 0.9))
        top_k = kwargs.get("top_k", config.get("top_k", 50))
        repetition_penalty = kwargs.get("repetition_penalty", config.get("repetition_penalty", 1.1))
    else:
        # Fallback defaults
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 50)
        repetition_penalty = kwargs.get("repetition_penalty", 1.1)
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output tokens
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output text
        decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if output_text.startswith(decoded_prompt):
            output_text = output_text[len(decoded_prompt):].strip()
        
        # Create metadata
        metadata = {
            "input_tokens": len(inputs["input_ids"][0]),
            "tokens_generated": len(outputs[0]) - len(inputs["input_ids"][0]),
            "model_type": "local"
        }
        
        return output_text, metadata
    
    except Exception as e:
        logger.error(f"Error generating text with local model: {str(e)}")
        return f"Error generating text: {str(e)}", {"error": str(e)}

def generate_with_gemini(
    prompt: str,
    model: Any,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate text with Gemini model.
    
    Args:
        prompt: Input prompt
        model: Gemini model client
        **kwargs: Override parameters
        
    Returns:
        tuple: (generated_text, metadata)
    """
    # Get default config for Gemini
    config = get_model_config("gemini")
    
    # Set defaults if not provided in kwargs
    max_output_tokens = kwargs.get("max_new_tokens", config.get("max_new_tokens", 2048))
    temperature = kwargs.get("temperature", config.get("temperature", 0.9))
    top_p = kwargs.get("top_p", config.get("top_p", 0.95))
    top_k = kwargs.get("top_k", config.get("top_k", 40))
    
    try:
        # Configure generation parameters
        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract text from response
        if hasattr(response, "text"):
            output_text = response.text
        else:
            output_text = response.candidates[0].content.parts[0].text
        
        # Create metadata
        # Note: Gemini API doesn't provide token count directly
        estimated_tokens = len(prompt.split()) + len(output_text.split())
        
        metadata = {
            "model_type": "gemini",
            "estimated_tokens": estimated_tokens,
            "tokens_generated": len(output_text.split())
        }
        
        return output_text, metadata
    
    except Exception as e:
        logger.error(f"Error generating text with Gemini: {str(e)}")
        return f"Error generating text: {str(e)}", {"error": str(e)} 