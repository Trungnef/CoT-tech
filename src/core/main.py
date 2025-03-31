"""
Main script for evaluating different LLMs on classical problems.
"""

import os
import argparse
import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from dotenv import load_dotenv

from src.models import (
    get_or_load_model,
    load_gemini_model,
    clear_memory,
    check_gpu_memory,
    generate_text_with_model
)
from src.evaluators import (
    ModelEvaluator,
    ParallelEvaluator,
    parse_gpu_allocation
)
from src.visualization import ReportGenerator
from src.utils import (
    setup_logging,
    print_status,
    print_section_header,
    format_time
)

# Load environment variables
load_dotenv()

def setup_argparse():
    """
    Set up command line arguments.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="LLM Evaluation Framework")
    
    # Input options
    parser.add_argument("--questions_file", type=str, default="db/questions/problems.json",
                        help="Path to questions JSON file (default: db/questions/problems.json)")
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (None for all)"
    )
    
    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["llama", "qwen", "gemini"],
        default=["llama", "gemini"],
        help="Models to evaluate"
    )
    
    # Prompt types
    parser.add_argument(
        "--prompt_types",
        nargs="+",
        choices=["standard", "cot", "hybrid_cot", "zero_shot_cot"],
        default=["standard", "cot"],
        help="Prompt types to evaluate"
    )
    
    # Execution mode
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run evaluation in parallel"
    )
    
    # GPU configuration
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use"
    )
    parser.add_argument(
        "--gpu_allocation",
        nargs="+",
        default=None,
        help="Custom GPU allocation in format 'model:gpu_id' (e.g., llama:0 qwen:1)"
    )
    
    # Optimization options
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for model loading"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for processing questions"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser

def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """
    Load questions from a JSON file.
    
    Args:
        questions_file: Path to the JSON file
        
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
        
        print_status("success", f"Loaded {len(questions)} questions from {questions_file}")
        return questions
    except Exception as e:
        print_status("error", f"Error loading questions: {str(e)}")
        return []

def load_models(model_names: List[str]) -> List[Dict[str, Any]]:
    """
    Load specified models.
    
    Args:
        model_names: List of model names to load
        
    Returns:
        list: List of model info dictionaries
    """
    models = []
    
    print_status("info", "Loading models...")
    
    for model_name in model_names:
        if model_name.lower() == "gemini":
            try:
                model = load_gemini_model()
                models.append({
                    "name": model_name,
                    "type": "gemini",
                    "model": model
                })
                print_status("success", f"Loaded {model_name} model")
            except Exception as e:
                print_status("error", f"Failed to load {model_name} model: {str(e)}")
        else:
            # For models loaded in evaluate_single_model_process, just create a placeholder
            models.append({
                "name": model_name,
                "type": "local"
            })
    
    return models

def evaluate_sequential(args: argparse.Namespace) -> str:
    """
    Run sequential model evaluation.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Output directory
    """
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_info = check_gpu_memory()
        print(gpu_info)
    
    # Print header
    print("\n" + "╭" + "─" * 40 + "╮")
    print("│ LLM Evaluation Framework │")
    print("╰" + "─" * 40 + "╯\n")
    
    # Load questions
    questions = load_questions(args.questions_file)
    if not questions:
        return None
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    model_list = []
    
    for model_name in args.models:
        if model_name.lower() == "gemini":
            try:
                model = load_gemini_model()
                model_info = {
                    "name": model_name,
                    "type": "gemini",
                    "model": model
                }
                model_list.append(model_info)
                print_status("success", f"Loaded {model_name} model")
            except Exception as e:
                print_status("error", f"Failed to load {model_name} model: {str(e)}")
        else:
            try:
                # Use primary GPU (0) for sequential evaluation
                gpu_id = 0 if torch.cuda.is_available() else -1
                tokenizer, model = get_or_load_model(model_name, gpu_id, args.use_4bit)
                model_info = {
                    "name": model_name,
                    "type": "local",
                    "model": model,
                    "tokenizer": tokenizer
                }
                model_list.append(model_info)
                print_status("success", f"Loaded {model_name} model on GPU {gpu_id}")
            except Exception as e:
                print_status("error", f"Failed to load {model_name} model: {str(e)}")
    
    if not model_list:
        print_status("error", "No models loaded, cannot proceed with evaluation")
        return None
    
    # Print header
    print("\n" + "╭" + "─" * 40 + "╮")
    print("│ Starting Sequential Evaluation │")
    print("╰" + "─" * 40 + "╯\n")
    
    # Start evaluation
    start_time = time.time()
    
    results = evaluator.evaluate_models(
        questions,
        model_list,
        args.prompt_types,
        args.batch_size,
        args.max_questions
    )
    
    elapsed_time = time.time() - start_time
    print_status("success", f"Evaluation completed in {elapsed_time:.2f}s")
    
    # Generate report
    if results:
        print_status("info", "Generating evaluation report...")
        report_generator = ReportGenerator(evaluator.result_dir)
        report_path = report_generator.generate_report()
        print_status("success", f"Report generated: {report_path}")
    
    print_status("success", f"Evaluation complete. Results saved to: {evaluator.result_dir}")
    
    return evaluator.result_dir

def evaluate_parallel(args: argparse.Namespace) -> str:
    """
    Run parallel model evaluation using multiple processes.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Output directory
    """
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_info = check_gpu_memory()
        print(gpu_info)
    
    # Print header
    print("\n" + "╭" + "─" * 40 + "╮")
    print("│ LLM Evaluation Framework │")
    print("╰" + "─" * 40 + "╯\n")
    
    # Load questions
    questions = load_questions(args.questions_file)
    if not questions:
        return None
    
    # Parse GPU IDs
    gpu_ids = [int(id) for id in args.gpu_ids.split(",")] if args.gpu_ids else [0]
    
    # Initialize evaluator
    evaluator = ParallelEvaluator()
    
    # Start evaluation
    result_info = evaluator.evaluate_models_in_parallel(
        questions,
        args.models,
        args.prompt_types,
        gpu_ids,
        args.gpu_allocation,
        args.batch_size,
        args.use_4bit,
        args.max_questions
    )
    
    # Generate report
    print_status("info", "Generating evaluation report...")
    report_generator = ReportGenerator(result_info["result_dir"])
    report_path = report_generator.generate_report()
    print_status("success", f"Report generated: {report_path}")
    print_status("success", f"Evaluation complete. Results saved to: {result_info['result_dir']}")
    
    return result_info["result_dir"]

def main():
    """Main entry point."""
    # Parse arguments
    args = setup_argparse().parse_args()
    
    # Configure output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run in parallel or sequential mode
    if args.parallel:
        result_dir = evaluate_parallel(args)
    else:
        result_dir = evaluate_sequential(args)
    
    # Return success/failure
    if result_dir:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 