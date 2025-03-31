"""
Parallel evaluator for model evaluation.
"""

import os
import time
import json
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from multiprocessing import Manager
import threading
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from src.evaluators.evaluator import ModelEvaluator
from src.evaluators.metrics import evaluate_answer_by_prompt_type
from src.models import get_or_load_model, load_gemini_model, generate_text_with_model
from src.prompts import standard_prompt, chain_of_thought_prompt, hybrid_cot_prompt, zero_shot_cot_prompt
from src.utils import print_status, save_json, save_dataframe, save_config

# Set up logging
logger = logging.getLogger(__name__)

def parse_gpu_allocation(
    gpu_allocation_str: List[str], 
    models: List[str], 
    default_gpu_ids: List[int]
) -> Dict[str, List[int]]:
    """
    Parse GPU allocation string into a model-to-GPU mapping.
    
    Args:
        gpu_allocation_str: List of allocation strings (e.g., ["llama:0", "qwen:1"])
        models: List of model names
        default_gpu_ids: Default list of GPU IDs to use
        
    Returns:
        dict: Mapping of model names to lists of GPU IDs
    """
    gpu_allocation = {model: [] for model in models}
    
    # If allocation string is provided, parse it
    if gpu_allocation_str:
        try:
            for alloc in gpu_allocation_str:
                model, gpu_id = alloc.split(":")
                gpu_allocation[model].append(int(gpu_id))
        except ValueError:
            print_status("error", "Invalid GPU allocation format. Using default allocation.")
            # Fall back to default allocation below
    
    # Fill in any models that don't have allocated GPUs
    next_gpu_idx = 0
    for model in models:
        if not gpu_allocation[model]:
            # For API models like Gemini, use -1 (no GPU needed)
            if model.lower() == "gemini":
                gpu_allocation[model] = [-1]
            else:
                # Use the next available GPU
                gpu_allocation[model] = [default_gpu_ids[next_gpu_idx % len(default_gpu_ids)]]
                next_gpu_idx += 1
    
    return gpu_allocation

def evaluate_question(
    question: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    model_name: str,
    model_type: str,
    prompt_fn: Any,
    prompt_type: str
) -> Dict[str, Any]:
    """
    Evaluate a single question with a specific model.
    
    Args:
        question: Question dictionary
        model: Model object
        tokenizer: Tokenizer (for local models)
        model_name: Name of the model
        model_type: Type of model ("local" or "gemini")
        prompt_fn: Function to generate prompt
        prompt_type: Type of prompt
        
    Returns:
        dict: Evaluation result
    """
    question_text = question["question"]
    # Support both solution and answer fields
    expected_solution = question.get("solution", question.get("answer", ""))
    question_type = question.get("type", "classical_problem")
    
    # Generate prompt
    prompt = prompt_fn(question_text, question_type)
    
    # Track metrics
    start_time = time.time()
    
    # Generate response with retries for Gemini
    max_retries = 3 if model_type == "gemini" else 1
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if model_type == "local":
                response, metadata = generate_text_with_model(
                    prompt, 
                    model_type="local",
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name
                )
                break
            elif model_type == "gemini":
                response, metadata = generate_text_with_model(
                    prompt, 
                    model_type="gemini",
                    model=model,
                    model_name=model_name
                )
                break
            else:
                response = f"Unsupported model type: {model_type}"
                metadata = {"error": "Unsupported model type"}
                break
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                response = f"Error after {max_retries} retries: {str(e)}"
                metadata = {"error": str(e)}
            else:
                print_status("warning", f"Retry {retry_count}/{max_retries} due to error: {str(e)}")
                time.sleep(2)  # Wait before retry
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Evaluate response
    evaluation_results = evaluate_answer_by_prompt_type(
        question_text, 
        response, 
        expected_solution, 
        prompt_type
    )
    
    # Create result dictionary
    result = {
        "model_name": model_name,
        "prompt_type": prompt_type,
        "question_id": question.get("id", ""),
        "question": question_text,
        "expected_solution": expected_solution,
        "response": response,
        "elapsed_time": elapsed_time,
        "tokens_per_second": metadata.get("tokens_per_second", 0),
        "response_length": len(response.split()),
        "has_error": "error" in metadata,
        "error_type": metadata.get("error", "")
    }
    
    # Add evaluation metrics
    result.update(evaluation_results)
    
    # Add question metadata if available
    if "difficulty" in question:
        result["difficulty"] = question["difficulty"]
    if "category" in question:
        result["category"] = question["category"]
    if "type" in question:
        result["question_type"] = question["type"]
    if "tags" in question:
        result["tags"] = question["tags"]
    
    return result

def evaluate_single_model_process(
    questions: List[Dict[str, Any]],
    model_name: str,
    prompt_types: List[str],
    gpu_id: int,
    batch_size: int,
    use_4bit: bool,
    shared_results: List,
    output_dir: str,
    timestamp: str
) -> List[Dict[str, Any]]:
    """
    Function to evaluate a single model on all prompt types in a separate process.
    
    Args:
        questions: List of question dictionaries
        model_name: Name of the model to evaluate
        prompt_types: List of prompt types to evaluate
        gpu_id: GPU ID to use (-1 for API models)
        batch_size: Number of questions per batch
        use_4bit: Whether to use 4-bit quantization
        shared_results: Shared list for results
        output_dir: Directory to save results
        timestamp: Timestamp for the run
        
    Returns:
        list: Evaluation results
    """
    results = []
    
    # Set up prompt functions
    prompt_mapping = {
        "standard": standard_prompt,
        "cot": chain_of_thought_prompt,
        "hybrid_cot": hybrid_cot_prompt,
        "zero_shot_cot": zero_shot_cot_prompt
    }
    
    print_status("info", f"Process {os.getpid()}: Evaluating {model_name} with {prompt_types} on GPU {gpu_id}")
    
    # Determine model type
    model_type = "gemini" if model_name.lower() == "gemini" or gpu_id == -1 else "local"
    
    # Load model
    if model_type == "gemini":
        model = load_gemini_model()
        tokenizer = None
    else:
        tokenizer, model = get_or_load_model(model_name, gpu_id, use_4bit=use_4bit)
    
    # Process each prompt type
    for prompt_type in prompt_types:
        prompt_fn = prompt_mapping[prompt_type]
        print_status("info", f"Evaluating {model_name} with {prompt_type} prompt on {len(questions)} questions")
        
        # Process questions in batches
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            batch_results = []
            
            # Process each question in the batch
            for question in tqdm(batch, desc=f"Batch {i//batch_size + 1}/{len(questions)//batch_size + 1}"):
                result = evaluate_question(
                    question,
                    model,
                    tokenizer,
                    model_name,
                    model_type,
                    prompt_fn,
                    prompt_type
                )
                batch_results.append(result)
            
            # Save batch results
            results.extend(batch_results)
            shared_results.extend(batch_results)
            
            # Save partial results to disk
            save_batch_results(batch_results, output_dir, model_name, prompt_type, timestamp)
    
    return results

def save_batch_results(
    batch_results: List[Dict[str, Any]],
    output_dir: str,
    model_name: str,
    prompt_type: str,
    timestamp: str
) -> None:
    """
    Save batch results to disk.
    
    Args:
        batch_results: List of result dictionaries
        output_dir: Directory to save results
        model_name: Name of the model
        prompt_type: Type of prompt
        timestamp: Timestamp for the run
    """
    if not batch_results:
        return
    
    # Create result directory
    result_dir = Path(output_dir) / timestamp
    model_dir = result_dir / f"{model_name}_{prompt_type}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    batch_timestamp = time.strftime("%H%M%S")
    json_path = model_dir / f"batch_results_{batch_timestamp}.json"
    
    # Save as JSON
    save_json(batch_results, json_path)
    
    logger.info(f"Saved batch results with {len(batch_results)} entries to {json_path}")

class ParallelEvaluator:
    """
    Class for parallel evaluation of models using multiple processes and GPUs.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the parallel evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamped directory for this evaluation run
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.result_dir = Path(output_dir) / self.timestamp
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Create plots directory
        self.plots_dir = self.result_dir / "plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Configuration dictionary
        self.config = {
            "timestamp": self.timestamp,
            "models": [],
            "prompt_types": [],
            "num_questions": 0,
            "batch_size": 0,
            "parallel": True,
            "gpu_allocation": {}
        }
        
        logger.info(f"Initialized ParallelEvaluator with output directory: {self.result_dir}")
    
    def evaluate_models_in_parallel(
        self,
        questions: List[Dict[str, Any]],
        models: List[str],
        prompt_types: List[str],
        gpu_ids: List[int] = [0],
        gpu_allocation: Optional[List[str]] = None,
        batch_size: int = 10,
        use_4bit: bool = True,
        max_questions: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate models in parallel using multiple processes and GPUs.
        
        Args:
            questions: List of question dictionaries
            models: List of model names to evaluate
            prompt_types: List of prompt types to evaluate
            gpu_ids: List of available GPU IDs
            gpu_allocation: Optional custom GPU allocation strings
            batch_size: Number of questions per batch
            use_4bit: Whether to use 4-bit quantization
            max_questions: Maximum number of questions to evaluate
        
        Returns:
            dict: Results info dictionary
        """
        # Update configuration
        self.config["models"] = models
        self.config["prompt_types"] = prompt_types
        self.config["batch_size"] = batch_size
        self.config["parallel"] = True
        
        # Limit number of questions if specified
        if max_questions and max_questions < len(questions):
            eval_questions = questions[:max_questions]
            self.config["num_questions"] = max_questions
        else:
            eval_questions = questions
            self.config["num_questions"] = len(questions)
        
        # Parse GPU allocation
        model_to_gpu = parse_gpu_allocation(gpu_allocation, models, gpu_ids)
        self.config["gpu_allocation"] = {model: str(gpus) for model, gpus in model_to_gpu.items()}
        
        print_status("info", f"Starting parallel evaluation with {len(eval_questions)} questions")
        print_status("info", f"Models: {', '.join(models)}")
        print_status("info", f"Prompt types: {', '.join(prompt_types)}")
        print_status("info", f"GPU allocation: {model_to_gpu}")
        
        # Save configuration
        save_config(self.config, self.result_dir)
        
        # Use a Manager to share results between processes
        with Manager() as manager:
            # Create a list that can be shared between processes
            shared_results = manager.list()
            
            # Create evaluation processes
            processes = []
            
            for model_name in models:
                for gpu_id in model_to_gpu[model_name]:
                    processes.append((
                        eval_questions,
                        model_name,
                        prompt_types,
                        gpu_id,
                        batch_size,
                        use_4bit,
                        shared_results,
                        self.output_dir,
                        self.timestamp
                    ))
            
            start_time = time.time()
            
            # Execute processes in parallel
            with ProcessPoolExecutor(max_workers=len(processes)) as executor:
                futures = [
                    executor.submit(
                        evaluate_single_model_process,
                        *process_args
                    )
                    for process_args in processes
                ]
                
                # Wait for all processes to complete
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error in evaluation process: {str(e)}")
                        print_status("error", f"Process error: {str(e)}")
            
            # Convert shared_results to list
            all_results = list(shared_results)
            
            # Save combined results
            results_df = pd.DataFrame(all_results)
            
            # Save as CSV
            csv_path = self.result_dir / "results.csv"
            save_dataframe(results_df, csv_path, format="csv")
            
            # Save as JSON
            json_path = self.result_dir / "results.json"
            save_json(all_results, json_path)
            
            elapsed_time = time.time() - start_time
            print_status("success", f"Parallel evaluation completed in {elapsed_time:.2f} seconds")
            print_status("success", f"Saved results with {len(all_results)} entries")
            
            return {
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "result_dir": str(self.result_dir),
                "num_results": len(all_results)
            } 