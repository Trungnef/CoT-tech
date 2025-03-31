"""
Core evaluator class for model evaluation.
"""

import os
import json
import time
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.evaluators.metrics import evaluate_answer_by_prompt_type
from src.models import generate_text_with_model
from src.prompts import (
    standard_prompt,
    chain_of_thought_prompt,
    hybrid_cot_prompt,
    zero_shot_cot_prompt
)
from src.utils import (
    create_timestamp_directory,
    save_json,
    save_dataframe,
    save_config,
    print_status
)

# Set up logging
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Class for evaluating and comparing different LLM performances.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamped directory for this evaluation run
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.result_dir = create_timestamp_directory(output_dir)
        
        # Create plots directory
        self.plots_dir = Path(self.result_dir) / "plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Register available prompt types
        self.prompt_types = {
            "standard": standard_prompt,
            "cot": chain_of_thought_prompt,
            "hybrid_cot": hybrid_cot_prompt,
            "zero_shot_cot": zero_shot_cot_prompt
        }
        
        # Initialize results dictionary
        self.results = []
        
        # Configuration dictionary
        self.config = {
            "timestamp": self.timestamp,
            "models": [],
            "prompt_types": [],
            "num_questions": 0,
            "batch_size": 0,
            "parallel": False,
            "gpu_allocation": {}
        }
        
        logger.info(f"Initialized ModelEvaluator with output directory: {self.result_dir}")
    
    def load_questions(self, questions_file: str) -> List[Dict[str, Any]]:
        """
        Load questions from a JSON file.
        
        Args:
            questions_file: Path to JSON file containing questions
            
        Returns:
            list: List of question dictionaries
        """
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if data is a dictionary with 'questions' key or a direct list
            if isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
            else:
                questions = data
                
            # Update config
            self.config["num_questions"] = len(questions)
            
            print_status("success", f"Loaded {len(questions)} questions from {questions_file}")
            return questions
        except Exception as e:
            logger.error(f"Error loading questions: {str(e)}")
            print_status("error", f"Failed to load questions: {str(e)}")
            return []
    
    def evaluate_single_question(
        self, 
        question: Dict[str, Any],
        model_info: Dict[str, Any], 
        prompt_type: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single question with a specific model and prompt type.
        
        Args:
            question: Question dictionary with 'question' and 'solution' keys
            model_info: Model information dictionary with 'name', 'type', etc.
            prompt_type: Type of prompt to use
            
        Returns:
            dict: Evaluation results
        """
        model_name = model_info["name"]
        model_type = model_info["type"]
        prompt_fn = self.prompt_types[prompt_type]
        
        question_text = question["question"]
        # Check for both "solution" and "answer" keys for compatibility
        expected_solution = question.get("solution", question.get("answer", ""))
        
        # Create prompt
        question_type = question.get("type", "classical_problem")
        prompt = prompt_fn(question_text, question_type)
        
        # Track metrics
        start_time = time.time()
        
        # Generate answer
        try:
            if model_type == "local":
                model = model_info["model"]
                tokenizer = model_info["tokenizer"]
                response, metadata = generate_text_with_model(
                    prompt, 
                    model_type="local",
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name
                )
            elif model_type == "gemini":
                model = model_info["model"]
                response, metadata = generate_text_with_model(
                    prompt, 
                    model_type="gemini",
                    model=model,
                    model_name=model_name
                )
            else:
                response = f"Unsupported model type: {model_type}"
                metadata = {"error": "Unsupported model type"}
        except Exception as e:
            response = f"Error: {str(e)}"
            metadata = {"error": str(e)}
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Evaluate answer
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
    
    def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        model_info: Dict[str, Any],
        prompt_type: str,
        batch_size: int = 10,
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of questions.
        
        Args:
            questions: List of question dictionaries
            model_info: Model information dictionary
            prompt_type: Type of prompt to use
            batch_size: Number of questions per batch
            max_workers: Maximum number of parallel workers
            
        Returns:
            list: List of evaluation results
        """
        batch_results = []
        
        model_name = model_info["name"]
        print_status("info", f"Evaluating {model_name} with {prompt_type} prompt on {len(questions)} questions")
        
        # Process questions in batches
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            
            # Use ThreadPoolExecutor for parallel processing within batch
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.evaluate_single_question, 
                        question, 
                        model_info, 
                        prompt_type
                    )
                    for question in batch
                ]
                
                # Collect results as they complete
                for future in tqdm(futures, desc=f"Batch {i//batch_size + 1}/{len(questions)//batch_size + 1}"):
                    result = future.result()
                    batch_results.append(result)
                    
                    # Save partial results
                    self.results.append(result)
                    if len(self.results) % 10 == 0:
                        self.save_partial_results()
        
        return batch_results
    
    def evaluate_models(
        self,
        questions: List[Dict[str, Any]],
        models: List[Dict[str, Any]],
        prompt_types: List[str],
        batch_size: int = 10,
        max_questions: Optional[int] = None
    ):
        """
        Evaluate multiple models on multiple prompt types.
        
        Args:
            questions: List of question dictionaries
            models: List of model information dictionaries
            prompt_types: List of prompt types to evaluate
            batch_size: Number of questions per batch
            max_questions: Maximum number of questions to evaluate (None for all)
        """
        # Update configuration
        self.config["models"] = [m["name"] for m in models]
        self.config["prompt_types"] = prompt_types
        self.config["batch_size"] = batch_size
        
        # Limit number of questions if specified
        if max_questions and max_questions < len(questions):
            eval_questions = questions[:max_questions]
            self.config["num_questions"] = max_questions
        else:
            eval_questions = questions
            self.config["num_questions"] = len(questions)
        
        print_status("info", f"Starting evaluation with {len(eval_questions)} questions")
        print_status("info", f"Models: {', '.join(self.config['models'])}")
        print_status("info", f"Prompt types: {', '.join(prompt_types)}")
        
        # Save configuration
        save_config(self.config, self.result_dir)
        
        # Evaluate each model with each prompt type
        total_evaluations = len(models) * len(prompt_types)
        current = 1
        
        start_time = time.time()
        
        for model_info in models:
            for prompt_type in prompt_types:
                print_status(
                    "progress", 
                    f"Evaluation {current}/{total_evaluations}: {model_info['name']} with {prompt_type} prompt"
                )
                
                # Evaluate this model and prompt type combination
                batch_results = self.evaluate_batch(
                    eval_questions,
                    model_info,
                    prompt_type,
                    batch_size
                )
                
                # Save partial results
                self.save_partial_results()
                
                current += 1
        
        # Save final results
        elapsed_time = time.time() - start_time
        print_status("success", f"Evaluation completed in {elapsed_time:.2f} seconds")
        
        self.save_results()
        
        return self.results
    
    def save_partial_results(self):
        """
        Save partial results during evaluation.
        """
        if not self.results:
            return
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save as CSV
        csv_path = Path(self.result_dir) / "partial_results.csv"
        save_dataframe(results_df, csv_path, format="csv")
        
        # Save as JSON
        json_path = Path(self.result_dir) / "partial_results.json"
        save_json(self.results, json_path)
        
        logger.info(f"Saved partial results with {len(self.results)} entries")
    
    def save_results(self):
        """
        Save final evaluation results.
        """
        if not self.results:
            return
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save as CSV
        csv_path = Path(self.result_dir) / "results.csv"
        save_dataframe(results_df, csv_path, format="csv")
        
        # Save as JSON
        json_path = Path(self.result_dir) / "results.json"
        save_json(self.results, json_path)
        
        print_status("success", f"Saved final results with {len(self.results)} entries")
        logger.info(f"Saved final results to {self.result_dir}")
        
        return {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "result_dir": str(self.result_dir)
        } 