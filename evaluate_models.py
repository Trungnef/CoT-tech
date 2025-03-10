"""
Main script to evaluate different LLMs on classical problems from PDF document.
"""

import os
import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import time
from datetime import datetime

from model_manager import (
    load_model_optimized,
    load_gemini_model,
    generate_text_with_model,
    parallel_generate,
    clear_memory,
    check_gpu_memory,
    qwen_model_path,
    qwen_tokenizer_path,
    llama_model_path,
    llama_tokenizer_path
)
from prompts import (
    standard_prompt,
    chain_of_thought_prompt,
    hybrid_cot_prompt,
    zero_shot_cot_prompt,
    tree_of_thought_prompt
)
from model_evaluator import ModelEvaluator

def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on classical problems from PDF document"
    )
    
    parser.add_argument(
        "--questions_json",
        type=str,
        default="db/questions/problems.json",
        help="Path to JSON file containing questions"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["llama", "qwen", "gemini"],
        choices=["llama", "qwen", "gemini"],
        help="Models to evaluate"
    )
    
    parser.add_argument(
        "--prompt_types",
        type=str,
        nargs="+",
        default=["standard", "cot", "hybrid_cot", "zero_shot_cot", "tree_of_thought"],
        choices=["standard", "cot", "hybrid_cot", "zero_shot_cot", "tree_of_thought"],
        help="Prompt types to evaluate"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of questions to process in parallel"
    )
    
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (None for all)"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for loading models"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers"
    )
    
    return parser

def process_batch(questions, model_type, prompt_type, prompt_fn):
    """Process a batch of questions with the specified model and prompt type."""
    prompts = [prompt_fn(q, "classical_problem") for q in questions]
    return parallel_generate(prompts, model_type, max_workers=3)

def main():
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    print("🚀 Starting LLM evaluation on classical problems")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load questions
    try:
        with open(args.questions_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract questions from the nested structure
        if "questions" in data:
            all_questions = [q["question"] for q in data["questions"]]
            print(f"✅ Loaded {len(all_questions)} questions from {args.questions_json}")
        else:
            all_questions = data
            print(f"✅ Loaded {len(all_questions)} questions")
        
        if args.max_questions:
            all_questions = all_questions[:args.max_questions]
            print(f"👉 Using first {args.max_questions} questions")
    except Exception as e:
        print(f"❌ Error loading questions: {e}")
        return
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ Found {torch.cuda.device_count()} GPUs")
        check_gpu_memory()
    else:
        print("⚠️ No GPU found")
    
    # Prepare prompt functions
    prompt_functions = {
        "standard": standard_prompt,
        "cot": chain_of_thought_prompt,
        "hybrid_cot": hybrid_cot_prompt,
        "zero_shot_cot": zero_shot_cot_prompt,
        #"tree_of_thought": tree_of_thought_prompt
    }
    
    # Process questions in batches for each model and prompt type
    results = []
    
    for model_type in args.models:
        print(f"\n🔄 Processing with {model_type.upper()} model")
        
        for prompt_type in args.prompt_types:
            print(f"\n📝 Using {prompt_type.upper()} prompting")
            prompt_fn = prompt_functions[prompt_type]
            
            # Process questions in batches
            for i in range(0, len(all_questions), args.batch_size):
                batch = all_questions[i:i + args.batch_size]
                print(f"\n⚡ Processing batch {i//args.batch_size + 1}/{len(all_questions)//args.batch_size + 1}")
                
                try:
                    # Record start time
                    start_time = time.time()
                    
                    # Process batch
                    responses = process_batch(batch, model_type, prompt_type, prompt_fn)
                    
                    # Record metrics
                    end_time = time.time()
                    batch_time = end_time - start_time
                    
                    # Save results
                    for q, r in zip(batch, responses):
                        results.append({
                            "model": model_type,
                            "prompt_type": prompt_type,
                            "question": q,
                            "response": r,
                            "time": batch_time / len(batch)
                        })
                    
                    # Clear memory after each batch
                    clear_memory()
                    
                except Exception as e:
                    print(f"❌ Error processing batch: {e}")
                    continue
    
    # Save all results
    results_file = os.path.join(results_dir, "evaluation_results.json")
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results saved to {results_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    # Create evaluator and generate report
    evaluator = ModelEvaluator(results_dir=results_dir)
    evaluator.analyze_results_from_json(results_file)
    report_path = evaluator.generate_report()
    
    print(f"\n✅ Evaluation completed. Report available at: {report_path}")

if __name__ == "__main__":
    main() 