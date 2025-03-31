"""
Text generation functionality for different model types.
"""

import time
import logging
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def generate_text_with_model(
    prompt: str, 
    model_type: str = "local", 
    max_tokens: int = 1024, 
    temperature: float = 0.7,
    model=None,
    tokenizer=None
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate text using a language model.
    
    Args:
        prompt (str): The prompt to generate text from
        model_type (str): Type of model to use ("local", "gemini")
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        model: The model object
        tokenizer: The tokenizer object (only required for local models)
    
    Returns:
        tuple: (generated_text, metadata)
    """
    start_time = time.time()
    
    # Record prompt length for metrics
    prompt_length = len(prompt.split())
    
    # Initialize metadata
    metadata = {
        "model_type": model_type,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_length": prompt_length,
    }
    
    try:
        # Generate text based on model type
        if model_type == "local":
            if model is None or tokenizer is None:
                raise ValueError("Local model generation requires model and tokenizer objects")
            
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_length = len(inputs["input_ids"][0])
            
            # Set generation parameters
            gen_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 50,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": temperature > 0.2
            }
            
            # Generate text
            with torch.no_grad():
                output = model.generate(**inputs, **gen_config)
            
            # Decode and clean the output
            generated_text = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
            
            # Calculate stats
            output_length = len(generated_text.split())
            tokens_per_second = output_length / (time.time() - start_time) if time.time() > start_time else 0
            
            # Add stats to metadata
            metadata.update({
                "input_tokens": input_length,
                "output_tokens": output_length,
                "tokens_per_second": tokens_per_second,
                "generation_time": time.time() - start_time,
            })
            
        elif model_type == "gemini":
            if model is None:
                raise ValueError("Gemini model generation requires model object")
            
            # Setup Gemini generation parameters
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 50,
            }
            
            # Generate text
            result = model.generate_content(prompt, generation_config=generation_config)
            
            if hasattr(result, 'text') and result.text:
                generated_text = result.text
            else:
                generated_text = ""
                
            # Calculate stats
            output_length = len(generated_text.split())
            tokens_per_second = output_length / (time.time() - start_time) if time.time() > start_time else 0
            
            # Add stats to metadata
            metadata.update({
                "output_tokens": output_length,
                "tokens_per_second": tokens_per_second,
                "generation_time": time.time() - start_time,
            })
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    except Exception as e:
        # Log the error
        logger.error(f"Error generating text: {str(e)}")
        
        # Return error information
        generated_text = f"Error: {str(e)}"
        metadata["error"] = str(e)
        metadata["generation_time"] = time.time() - start_time
    
    return generated_text, metadata

def parallel_generate(
    prompts: List[str], 
    model_info: Dict[str, Any], 
    max_tokens: int = 1024, 
    temperature: float = 0.7,
    max_workers: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate text for multiple prompts in parallel.
    
    Args:
        prompts (list): List of prompts to generate text from
        model_info (dict): Model information dictionary
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        max_workers (int): Maximum number of worker threads
    
    Returns:
        list: List of generation results
    """
    model_type = model_info.get("type", "local")
    model = model_info.get("model")
    tokenizer = model_info.get("tokenizer")
    
    results = []
    
    # Create a thread-local storage to prevent concurrent access to models
    thread_local = threading.local()
    
    def _generate_text(prompt):
        """
        Thread worker function for text generation.
        """
        try:
            generated_text, metadata = generate_text_with_model(
                prompt, 
                model_type=model_type, 
                max_tokens=max_tokens, 
                temperature=temperature,
                model=model,
                tokenizer=tokenizer
            )
            
            return {
                "prompt": prompt,
                "response": generated_text,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error in parallel text generation: {str(e)}")
            return {
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each prompt to a worker thread
        for result in executor.map(_generate_text, prompts):
            results.append(result)
    
    return results 