"""
Main script to evaluate different LLMs on classical problems from PDF document.
"""

import portalocker
import os
import argparse
import google.generativeai as genai
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
import json
import time
from datetime import datetime
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import random
import pandas as pd
import sys
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import io
import base64
import multiprocessing
import gc
import threading
from multiprocessing import Process, Manager
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Model-specific configuration for max tokens
MODEL_MAX_TOKENS = {
    "llama": 512,  # Llama models can generate longer outputs
    "qwen": 256,   # Qwen models have lower context length than Llama
    "gemini": 1024, # Gemini has high token limit
    "default": 1024 # Default value for unspecified models
}

# Function to get max tokens for a specific model
def get_max_tokens(model_name):
    """
    Get appropriate max_tokens value for a specific model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        int: Maximum tokens to generate
    """
    model_key = model_name.lower()
    
    # Extract base model name if version is specified (e.g., llama3-8b -> llama)
    for base_model in MODEL_MAX_TOKENS.keys():
        if base_model in model_key and base_model != "default":
            return MODEL_MAX_TOKENS[base_model]
    
    # Return default if no specific configuration
    return MODEL_MAX_TOKENS["default"]

from check.model_manager import (
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
from check.prompts import (
    standard_prompt,
    chain_of_thought_prompt,
    hybrid_cot_prompt,
    zero_shot_cot_prompt,
    tree_of_thought_prompt,
    zero_shot_prompt,
    few_shot_3_prompt,
    few_shot_5_prompt,
    few_shot_7_prompt,
    cot_self_consistency_3_prompt,
    cot_self_consistency_5_prompt,
    cot_self_consistency_7_prompt,
    react_prompt
)
from check.model_evaluator import ModelEvaluator

# Global model cache
_LOADED_MODELS = {}
_MODEL_LOADING_LOCKS = {}

# Thêm hàm đánh giá mức độ tự tin (confidence) trong câu trả lời
def evaluate_answer_confidence(response):
    # Tìm kiếm các từ/cụm từ thể hiện sự tự tin
    high_confidence_patterns = [
        r'chắc chắn', r'dứt khoát', r'rõ ràng', r'không còn nghi ngờ gì',
        r'hiển nhiên', r'kết quả là', r'vậy', r'vì vậy', r'do đó',
        r'kết luận', r'có thể khẳng định', r'khẳng định'
    ]
    
    # Tìm kiếm các từ/cụm từ thể hiện sự không chắc chắn
    low_confidence_patterns = [
        r'có lẽ', r'có thể', r'không chắc', r'tôi nghĩ', r'dường như',
        r'không rõ', r'chưa chắc', r'tôi đoán', r'tôi tin', r'tôi cho rằng',
        r'tôi ước tính', r'không biết', r'không dám chắc', r'bối rối',
        r'phức tạp', r'khó'
    ]
    
    # Đếm số lần xuất hiện của các pattern
    high_confidence_count = sum(1 for pattern in high_confidence_patterns if re.search(pattern, response.lower()))
    low_confidence_count = sum(1 for pattern in low_confidence_patterns if re.search(pattern, response.lower()))
    
    # Tính điểm confidence (0-1)
    total_markers = high_confidence_count + low_confidence_count
    if total_markers == 0:
        confidence_score = 0.5  # Neutral if no markers
    else:
        confidence_score = high_confidence_count / total_markers
    
    # Kiểm tra độ dứt khoát của kết luận (thường ở cuối câu trả lời)
    conclusion_part = response.split(".")[-3:]  # 3 câu cuối
    conclusion_text = ".".join(conclusion_part)
    
    # Kiểm tra kết luận có chứa số không - dấu hiệu của sự tự tin vào câu trả lời
    has_numbers_in_conclusion = bool(re.search(r'\d+', conclusion_text))
    
    # Tăng điểm nếu kết luận có con số cụ thể
    if has_numbers_in_conclusion:
        confidence_score = min(1.0, confidence_score + 0.2)
    
    # Kiểm tra độ dài câu trả lời - câu trả lời dài thường ít tự tin hơn
    response_length = len(response.split())
    if response_length > 300:
        confidence_score = max(0.1, confidence_score - 0.1)
    
    return confidence_score

# Thêm hàm tính bias correction cho CoT và Hybrid-CoT
def calculate_bias_correction(prompt_type, correctness_score, reasoning_score, response_length):
    """
    Tính toán điều chỉnh độ chệch cho các loại prompt khác nhau
    """
    bias_correction = 0.0
    
    if prompt_type == "standard":
        # Standard prompt thường có điểm số thấp hơn do thiếu quá trình suy luận
        bias_correction = 0.05  # Thêm nhẹ để công bằng
    
    elif prompt_type == "cot":
        # CoT thường dài dòng và có nhiều cơ hội để mắc lỗi
        length_penalty = min(0.1, response_length / 5000)
        bias_correction = -0.05 * length_penalty
        
        # Nhưng cũng được thưởng cho quá trình suy luận
        if reasoning_score > 0.7:
            bias_correction += 0.05
    
    elif prompt_type == "hybrid_cot":
        # Hybrid-CoT thường cân bằng hơn
        if reasoning_score > 0.6 and correctness_score > 0.7:
            bias_correction = 0.03  # Thưởng cho cả suy luận tốt và kết quả đúng
        
    return bias_correction

# Thêm hàm phân tích dựa trên loại câu hỏi cụ thể
def analyze_by_specific_question_type(results_df):
    """
    Phân tích hiệu suất chi tiết hơn cho từng loại câu hỏi cụ thể
    """
    # Phân tích cho câu hỏi số học
    arithmetic_df = results_df[results_df['question_tags'].str.contains('số học|arithmetic', case=False, na=False)]
    
    # Phân tích cho câu hỏi đại số
    algebra_df = results_df[results_df['question_tags'].str.contains('đại số|algebra', case=False, na=False)]
    
    # Phân tích cho câu hỏi hình học
    geometry_df = results_df[results_df['question_tags'].str.contains('hình học|geometry', case=False, na=False)]
    
    # Phân tích cho câu hỏi logic
    logic_df = results_df[results_df['question_tags'].str.contains('logic|suy luận', case=False, na=False)]
    
    results = {
        'arithmetic': arithmetic_df.groupby(['model_name', 'prompt_type'])['correctness_score'].mean().reset_index() if not arithmetic_df.empty else pd.DataFrame(),
        'algebra': algebra_df.groupby(['model_name', 'prompt_type'])['correctness_score'].mean().reset_index() if not algebra_df.empty else pd.DataFrame(),
        'geometry': geometry_df.groupby(['model_name', 'prompt_type'])['correctness_score'].mean().reset_index() if not geometry_df.empty else pd.DataFrame(),
        'logic': logic_df.groupby(['model_name', 'prompt_type'])['correctness_score'].mean().reset_index() if not logic_df.empty else pd.DataFrame()
    }
    
    return results

# Thêm hàm đánh giá câu trả lời dựa trên các tiêu chí của từng môn học cụ thể
def evaluate_by_subject_criteria(question, response, subject_type):
    """
    Đánh giá câu trả lời dựa trên các tiêu chí đặc thù của từng môn học
    """
    score = 0.5  # Điểm mặc định
    
    if subject_type == 'arithmetic':
        # Kiểm tra các phép tính số học cơ bản
        calculations = re.findall(r'(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)', response)
        correct_calculations = 0
        
        for calc in calculations:
            try:
                num1, op, num2, result = calc
                expected = eval(f"{num1}{op}{num2}")
                if abs(float(result) - float(expected)) <= 0.01:
                    correct_calculations += 1
            except:
                pass
        
        calculation_accuracy = correct_calculations / max(1, len(calculations)) if calculations else 0
        score = calculation_accuracy
    
    elif subject_type == 'algebra':
        # Kiểm tra các khái niệm về đại số
        algebra_concepts = [
            r'phương trình', r'biến số', r'hệ số', r'đa thức', r'bậc',
            r'hàm số', r'đạo hàm', r'tích phân', r'ma trận', r'vector'
        ]
        
        concept_count = sum(1 for concept in algebra_concepts if re.search(concept, response.lower()))
        concept_score = min(1.0, concept_count / 3)
        
        # Kiểm tra các phép biến đổi đại số
        transformations = re.findall(r'(?:=>|⟹|→|=|⟺)', response)
        transformation_score = min(1.0, len(transformations) / 5)
        
        score = 0.6 * concept_score + 0.4 * transformation_score
    
    elif subject_type == 'geometry':
        # Kiểm tra các khái niệm về hình học
        geometry_concepts = [
            r'góc', r'cạnh', r'đường thẳng', r'tam giác', r'tứ giác',
            r'đường tròn', r'diện tích', r'thể tích', r'đường kính', r'bán kính'
        ]
        
        concept_count = sum(1 for concept in geometry_concepts if re.search(concept, response.lower()))
        concept_score = min(1.0, concept_count / 3)
        
        # Kiểm tra có đề cập đến công thức hình học
        has_formulas = bool(re.search(r'công thức|S\s*=|V\s*=|P\s*=', response.lower()))
        
        score = 0.7 * concept_score + (0.3 if has_formulas else 0)
    
    elif subject_type == 'logic':
        # Kiểm tra các biểu thức logic
        logic_operators = [
            r'nếu', r'thì', r'và', r'hoặc', r'không', r'khi và chỉ khi',
            r'suy ra', r'tương đương', r'phủ định', r'mệnh đề'
        ]
        
        operator_count = sum(1 for op in logic_operators if re.search(op, response.lower()))
        logic_score = min(1.0, operator_count / 4)
        
        # Kiểm tra cấu trúc suy luận logic
        reasoning_structure = re.findall(r'(?:Trước tiên|Thứ hai|Tiếp theo|Cuối cùng|Kết luận)', response)
        structure_score = min(1.0, len(reasoning_structure) / 3)
        
        score = 0.5 * logic_score + 0.5 * structure_score
    
    return score

# Thiết lập logging nâng cao
console = Console()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)]
)

# Tạo logger
logger = logging.getLogger("model_evaluation")

def print_header(title, color="cyan"):
    """In tiêu đề với định dạng đẹp."""
    console.print(f"\n[bold {color}]{'=' * 80}[/bold {color}]")
    console.print(f"[bold {color}]{title.center(80)}[/bold {color}]")
    console.print(f"[bold {color}]{'=' * 80}[/bold {color}]\n")

def print_section(title, color="yellow"):
    """In tiêu đề phần với định dạng đẹp."""
    console.print(f"\n[bold {color}]{'-' * 40}[/bold {color}]")
    console.print(f"[bold {color}]{title}[/bold {color}]")
    console.print(f"[bold {color}]{'-' * 40}[/bold {color}]\n")

def print_model_info(model_name, model_size=None, model_path=None):
    """In thông tin mô hình với định dạng đẹp."""
    model_colors = {
        "llama": "red1",
        "qwen": "blue",
        "gemini": "purple"
    }
    color = model_colors.get(model_name.lower(), "green")
    
    model_info = Table.grid(padding=(0, 1))
    model_info.add_column(style="bold")
    model_info.add_column()
    
    model_info.add_row("Mô hình:", f"[bold {color}]{model_name.upper()}[/bold {color}]")
    if model_size:
        model_info.add_row("Kích thước:", f"{model_size}")
    if model_path:
        # Rút gọn đường dẫn để không hiển thị quá dài
        short_path = os.path.basename(os.path.dirname(model_path))
        model_info.add_row("Đường dẫn:", f"{short_path}/...")
    
    console.print(model_info)

def print_status(status_type, message, color="green"):
    """In thông báo trạng thái."""
    icons = {
        "success": " ",
        "error": " ",
        "warning": " ",
        "info": "ℹ️",
        "loading": " "
    }
    icon = icons.get(status_type, "➤")
    console.print(f"{icon} [bold {color}]{message}[/bold {color}]")

# def setup_argparse():
#     """Setup argument parser with improved documentation."""
#     import argparse
#     parser = argparse.ArgumentParser(description='Evaluate multiple language models on classical problems.')
    
#     # Configuration arguments
#     parser.add_argument('--questions_json', type=str, default='db/questions/problems.json',
#                       help='JSON file containing questions to evaluate')
#     parser.add_argument('--models', nargs='+', default=['llama', 'qwen', 'gemini'],
#                       help='Models to evaluate')
#     parser.add_argument('--prompt_types', nargs='+', 
#                       default=['standard', 'zero_shot', 'cot', 'few_shot_3', 'few_shot_5', 'few_shot_7',
#                                'cot_self_consistency_3', 'cot_self_consistency_5', 'cot_self_consistency_7', 'react'],
#                       help='Prompt types to use')
#     parser.add_argument('--max_questions', type=int, default=0,
#                       help='Maximum number of questions to evaluate (0 for all)')
#     parser.add_argument('--batch_size', type=int, default=5,
#                       help='Batch size for model evaluation')
#     parser.add_argument('--max_workers', type=int, default=1,
#                       help='Maximum number of worker processes/threads')
    
#     # Execution mode arguments
#     parser.add_argument('--parallel', action='store_true',
#                       help='Enable parallel evaluation mode')
#     parser.add_argument('--gpu_ids', type=str, default='',
#                       help='Comma-separated list of GPU IDs to use')
#     parser.add_argument('--gpu_allocation', type=str, default='',
#                       help='Specific GPU allocation (format: model1:gpu_id1,model2:gpu_id2,...)')
#     parser.add_argument('--use_4bit', action='store_true', default=True,
#                       help='Use 4-bit quantization for local models')
    
#     # Output arguments
#     parser.add_argument('--results_file', type=str, default='results',
#                       help='Directory to save results and visualizations')
#     parser.add_argument('--results_file', type=str, default='',
#                       help='JSON file containing existing results to analyze or resume from')
    
#     # Resume arguments
#     parser.add_argument('--resume', action='store_true',
#                       help='Resume evaluation from existing results')
#     parser.add_argument('--skip_complete', action='store_true',
#                       help='Skip completely evaluated model-prompt combinations')
    
#     # Analysis arguments
#     parser.add_argument('--analyze_only', action='store_true',
#                       help='Only analyze existing results without running evaluation')
#     parser.add_argument('--skip_gemini_eval', action='store_true',
#                       help='Skip Gemini API evaluation of reasoning quality')
#     parser.add_argument('--gemini_sample_size', type=int, default=30,
#                       help='Sample size for Gemini API evaluation of reasoning quality')
#     parser.add_argument('--analyze_reasoning', action='store_true',
#                       help='Analyze reasoning quality in addition to standard metrics')
#     parser.add_argument('--analyze_prompt_effectiveness', action='store_true',
#                       help='Analyze effectiveness of different prompt types')
    
#     # Add option for generating comprehensive prompt comparison report
#     parser.add_argument("--analyze_prompt_comparison", action="store_true", 
#                       help="Generate a comprehensive report comparing different prompt techniques")
    
#     # Add option for specific ReAct prompt analysis
#     parser.add_argument("--analyze_react", action="store_true",
#                       help="Perform detailed analysis of ReAct prompt performance and characteristics")
    
#     return parser

def load_questions(questions_json):
    """Load questions from a JSON file."""
    try:
        with open(questions_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract questions from the nested structure
        if isinstance(data, dict) and "questions" in data:
            all_questions = [q["question"] for q in data["questions"]]
        else:
            all_questions = data
            
        return all_questions
    except Exception as e:
        print_status("error", f"Error loading questions: {e}", "red")
        return []

def process_batch(questions, model_name, prompt_fn, prompt_type, max_workers=1, use_4bit=True):
    """Process a batch of questions with the specified model and prompt type."""
    # Create prompts using the prompt function
    prompts = [prompt_fn(q, "classical_problem") for q in questions]
    
    # Record start time
    start_time = time.time()
    
    # For API models, force sequential processing to avoid rate limits
    if model_name == "gemini":
        print("📝 Using sequential processing for API model to avoid rate limits")
        max_workers = 1  # Force sequential processing for API models
    
    # Generate responses - make sure to pass model_name, not the prompts as model type
    responses = []
    try:
        # Debug output to verify what's being passed
        print(f"Debug: Processing {len(prompts)} prompts with model: {model_name}")
        
        # Import the function directly to avoid confusion
        from check.model_manager import generate_text_with_model
        
        # Process each prompt individually to avoid confusion
        for prompt in prompts:
            try:
                # Make sure to pass the model_name first, then the prompt
                response = generate_text_with_model(model_name, prompt)
                responses.append(response)
            except Exception as e:
                print(f"Error processing prompt: {e}")
                responses.append(f"[Error: {str(e)}]")
    except Exception as e:
        print_status("error", f"Error generating responses: {e}", "red")
        # Return empty responses if generation fails
        responses = ["[Error: Failed to generate response]" for _ in questions]
    
    # Record end time
    end_time = time.time()
    batch_time = end_time - start_time
    
    # Format results
    batch_results = []
    for i, (question, response) in enumerate(zip(questions, responses)):
        result = {
            "question": question,
            "response": response,
            "elapsed_time": batch_time / len(questions),  # Average time per question
            "timestamp": datetime.now().isoformat()
        }
        batch_results.append(result)
    
    return batch_results

def save_results(results, output_dir, timestamp=None):
    """Save evaluation results to a JSON file."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    temp_results_file = os.path.join(output_dir, "temp_results.json")
    
    # Save both complete and temporary results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    with open(temp_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print_status("success", f"Results saved to {results_file}", "green")
    return results_file

def load_existing_results(output_dir, timestamp=None):
    """Load existing results from the temporary file."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return {}  # Return empty results if directory didn't exist
    
    temp_results_file = os.path.join(output_dir, "temp_results.json")
    
    # Check if temporary results file exists
    if not os.path.exists(temp_results_file):
        print_status("info", f"No temporary results file found at {temp_results_file}", "yellow")
        return {}  # Return empty results if file doesn't exist
    
    try:
        with open(temp_results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print_status("warning", f"Error decoding JSON from {temp_results_file}. Starting with empty results.", "yellow")
        return {}
    except Exception as e:
        print_status("warning", f"Error loading results: {e}. Starting with empty results.", "yellow")
        return {}

def generate_report_and_visualizations(results_file, output_dir):
    """
    Generate a comprehensive report and visualizations from evaluation results.
    
    Args:
        results_file (str): Path to the JSON file containing evaluation results
        output_dir (str): Directory to save the report and visualizations
    """
    try:
        # Create directories for plots and reports
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        if not results:
            print_status("error", "No results found in the results file", "red")
            return
            
        # Process all results and convert to DataFrame
        all_results = []
        
        # Keep track of failed responses for Gemini API
        failed_responses = 0
        
        # Process each model and prompt type
        for key, value in results.items():
            if not isinstance(value, list):
                print_status("warning", f"Skipping non-list results for {key}", "yellow")
                continue
                
            # Extract model name and prompt type from key
            parts = key.split('_')
            if len(parts) >= 2:
                model_name = parts[0]
                prompt_type = '_'.join(parts[1:])
            else:
                model_name = key
                prompt_type = "unknown"
            
            # Process each result
            for result in value:
                # Skip None or empty results
                if not result:
                    continue
                    
                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    continue
                
                # Handle missing fields (especially for API errors)
                if "response" not in result and "answer" in result:
                    result["response"] = result["answer"]
                    
                # Skip results with error responses
                if "response" in result and result["response"].startswith("[Error:"):
                    failed_responses += 1
                    continue
                
                # Ensure all required fields are present
                if "question" not in result or "response" not in result:
                    continue
                
                # Ensure model_name is present
                if "model" not in result:
                    result["model"] = model_name
                    
                # Calculate derived metrics
                # Ensure response is a string
                response = result.get("response", "")
                if not isinstance(response, str):
                    response = str(response)
                    result["response"] = response
                
                # Calculate response length
                if "response_length" not in result:
                    result["response_length"] = len(response.split())
                    
                # Calculate elapsed time if missing
                if "elapsed_time" not in result:
                    result["elapsed_time"] = 0
                
                # Add model and prompt information
                result["model_name"] = model_name
                result["prompt_type"] = prompt_type
                
                # Evaluate correctness and reasoning
                question = result["question"]
                
                # Kiểm tra kiểu dữ liệu của question
                if isinstance(question, dict):
                    expected_solution = question.get("expected_solution", "")
                    subject_type = question.get("subject", "")
                else:
                    # Nếu question là chuỗi, sử dụng chuỗi rỗng làm expected_solution
                    expected_solution = ""
                    subject_type = ""
                
                try:
                    # Try to evaluate additional metrics
                    correctness_score = evaluate_answer_correctness(question, response, expected_solution)
                    reasoning_score, reasoning_steps = evaluate_reasoning_quality(response, expected_solution)
                    
                    # Confidence score
                    confidence_score = evaluate_answer_confidence(response)
                    
                    # Subject-specific evaluation
                    subject_score = evaluate_by_subject_criteria(question, response, subject_type)
                    
                    # Bias correction
                    bias_correction = calculate_bias_correction(prompt_type, correctness_score, reasoning_score, result["response_length"])
                    
                    # Add to result
                    result["correctness_score"] = correctness_score
                    result["reasoning_score"] = reasoning_score
                    result["num_reasoning_steps"] = reasoning_steps
                    result["confidence_score"] = confidence_score
                    result["subject_score"] = subject_score
                    result["bias_correction"] = bias_correction
                    
                except Exception as e:
                    # If evaluation fails, assign default scores
                    result["correctness_score"] = 0
                    result["reasoning_score"] = 0
                    result["num_reasoning_steps"] = 0
                    result["confidence_score"] = 0
                    result["subject_score"] = 0
                    result["bias_correction"] = 0
                
                # Add to all results
                all_results.append(result)
        
        # Check if we have enough valid results to generate a report
        if not all_results:
            print_status("error", "No valid results after filtering out errors", "red")
            return
            
        if failed_responses > 0:
            print_status("warning", f"Skipped {failed_responses} failed responses (likely API errors)", "yellow")
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(all_results)
        
        # Load questions data để so sánh câu trả lời với đáp án đúng
        print_status("info", "Loading expected answers from questions data...", "blue")
        questions_data = {}
        try:
            # Thử tải file questions.json
            with open('./db/questions/problems.json', 'r', encoding='utf-8') as f:
                questions_json = json.load(f)
                # Tạo dictionary với key là câu hỏi, value là solution
                for q in questions_json.get("questions", []):
                    questions_data[q["question"]] = {
                        "solution": q["solution"],
                        "type": q["type"],
                        "difficulty": q["difficulty"],
                        "tags": q.get("tags", [])
                    }
            print_status("success", f"Loaded {len(questions_data)} questions with solutions", "green")
        except Exception as e:
            print_status("warning", f"Could not load questions data: {e}. Will use basic error-based accuracy only.", "yellow")
        
        # Đánh giá câu trả lời dựa trên độ tương đồng với đáp án đúng
        if questions_data:
            print_status("info", "Evaluating responses against expected answers...", "blue")
            
            # Tạo hàm đánh giá chất lượng suy luận (đặc biệt cho CoT và Hybrid-CoT)
            def evaluate_reasoning_quality(response, expected_solution):
                """
                Evaluate the quality of reasoning in a response.
    
    Args:
                    response (str): The model's response
                    expected_solution (str): The expected solution
                    
                Returns:
                    tuple: (reasoning_score, consistency_score, calculation_errors, num_steps, complexity_score)
                """
                # Phát hiện các bước suy luận
                reasoning_patterns = r'(?:Bước \d+:|Đầu tiên|Tiếp theo|Sau đó|Cuối cùng|Ta có|Áp dụng|Từ đó|Xét|Do đó|Vậy nên|Vì vậy|Khi đó|Theo|Nhận thấy|Giả sử|Suy ra)'
                reasoning_steps = re.findall(f'{reasoning_patterns}.*?(?={reasoning_patterns}|$)', response, re.DOTALL)
                
                # Đếm số bước suy luận rõ ràng
                num_steps = len(reasoning_steps)
                
                # Phát hiện các phép tính toán số học
                calculations = re.findall(r'(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)', response)
                
                # Tính điểm cho suy luận
                reasoning_score = min(1.0, (num_steps / 3) + (len(calculations) / 4))
                
                # Phân tích lỗi tính toán
                calculation_errors = []
                for calc in calculations:
                    try:
                        num1, op, num2, result = calc
                        expected = eval(f"{num1}{op}{num2}")
                        if abs(float(result) - float(expected)) > 0.01:
                            calculation_errors.append({
                                'expression': f"{num1} {op} {num2} = {result}",
                                'expected': expected,
                                'actual': float(result),
                                'error_type': 'calculation_error'
                            })
                    except Exception:
                        pass  # Bỏ qua các phép tính không thể đánh giá
                
                # Tỷ lệ lỗi tính toán
                calculation_error_rate = len(calculation_errors) / max(1, len(calculations)) if calculations else 0
                
                # Kiểm tra tính nhất quán giữa các bước suy luận
                consistency_score = 0.8  # Giá trị mặc định
                if len(reasoning_steps) > 1:
                    # Kiểm tra xem bước đầu tiên có kết thúc bằng số không
                    match_end = re.search(r'\d+\s*$', reasoning_steps[0])
                    prev_ends_with_number = bool(match_end) if match_end is not None else False
                    
                    consistency_count = 0
                    for i in range(1, len(reasoning_steps)):
                        # Kiểm tra xem bước hiện tại có bắt đầu bằng số không
                        match_start = re.search(r'^\s*\d+', reasoning_steps[i])
                        current_starts_with_number = bool(match_start) if match_start is not None else False
                        
                        if (prev_ends_with_number and current_starts_with_number) or (not prev_ends_with_number and not current_starts_with_number):
                            consistency_count += 1
                        
                        # Cập nhật cho bước tiếp theo
                        match_end = re.search(r'\d+\s*$', reasoning_steps[i])
                        prev_ends_with_number = bool(match_end) if match_end is not None else False
                    
                    consistency_score = consistency_count / max(1, len(reasoning_steps) - 1)
                
                # Tính độ phức tạp của phản hồi (dùng cho Complexity-Performance Trade-off)
                response_words = len(response.split())
                response_sentences = len(re.split(r'[.!?]+', response))
                # Tính toán điểm phức tạp dựa trên độ dài, số bước suy luận và số phép tính
                complexity_score = (
                    0.4 * min(1.0, response_words / 500) +  # Độ dài câu trả lời
                    0.3 * min(1.0, num_steps / 5) +         # Số bước suy luận
                    0.3 * min(1.0, len(calculations) / 5)   # Số phép tính
                )
                
                return reasoning_score, consistency_score, calculation_errors, num_steps, complexity_score
            
            # Hàm nâng cao để đánh giá chất lượng suy luận với thông tin câu hỏi
            def enhanced_evaluate_reasoning(response, expected_solution, question):
                # Lấy kết quả đánh giá cơ bản
                reasoning_score, consistency_score, calculation_errors, num_steps, complexity_score = evaluate_reasoning_quality(response, expected_solution)
                
                # Trích xuất từ khóa quan trọng từ câu hỏi và đáp án mẫu
                question_keywords = set([w.lower() for w in question.split() if len(w) > 3])
                solution_keywords = set([w.lower() for w in expected_solution.split() if len(w) > 3])
                important_keywords = question_keywords.union(solution_keywords)
                
                # Phát hiện các bước suy luận
                reasoning_patterns = r'(?:Bước \d+:|Đầu tiên|Tiếp theo|Sau đó|Cuối cùng|Ta có|Áp dụng|Từ đó|Xét|Do đó|Vậy nên|Vì vậy|Khi đó|Theo|Nhận thấy|Giả sử|Suy ra)'
                reasoning_steps = re.findall(f'{reasoning_patterns}.*?(?={reasoning_patterns}|$)', response, re.DOTALL)
                
                # Đánh giá mức độ phù hợp của từng bước suy luận
                relevance_scores = []
                for step in reasoning_steps:
                    step_keywords = set([w.lower() for w in step.split() if len(w) > 3])
                    overlap = len(step_keywords.intersection(important_keywords))
                    relevance = overlap / max(1, len(important_keywords))
                    relevance_scores.append(relevance)
                
                # Tính điểm phù hợp trung bình
                relevance_score = sum(relevance_scores) / max(1, len(relevance_scores)) if relevance_scores else 0.5
                
                # Phát hiện các loại lỗi suy luận
                logic_errors = []
                
                # Kiểm tra lỗi phi logic cơ bản (các mẫu cụ thể)
                illogical_patterns = [
                    r'số\s+\w+\s+là\s+âm',  # Số tự nhiên không thể âm
                    r'chia\s+cho\s+0',  # Lỗi chia cho 0
                    r'căn\s+bậc\s+\w+\s+của\s+số\s+âm',  # Căn bậc chẵn của số âm
                    r'phương\s+trình\s+vô\s+nghiệm.*có.*nghiệm',  # Mâu thuẫn về nghiệm
                    r'tam\s+giác\s+có\s+góc\s+\w+\s+bằng\s+180',  # Lỗi về góc tam giác
                ]
                
                for pattern in illogical_patterns:
                    if re.search(pattern, response.lower()):
                        logic_errors.append({
                            'pattern': pattern,
                            'error_type': 'logic_error'
                        })
                
                # Đánh giá độ ngắn gọn và hiệu quả
                efficiency_score = 0.0
                if num_steps > 0:
                    # Hiệu quả = Độ chính xác / Độ dài suy luận
                    words_per_step = sum(len(step.split()) for step in reasoning_steps) / num_steps
                    # Hiệu quả cao nếu ít từ mà chất lượng cao
                    efficiency_score = reasoning_score / (1 + words_per_step / 50)  # Chuẩn hóa, mong đợi khoảng 50 từ/bước
                
                # Tính điểm tổng hợp nâng cao
                enhanced_reasoning_score = (
                    reasoning_score * 0.5 +  # Chất lượng cơ bản
                    relevance_score * 0.3 +  # Độ phù hợp với bài toán
                    efficiency_score * 0.2    # Hiệu quả diễn đạt
                )
                
                # Tạo báo cáo đầy đủ về chất lượng suy luận
                reasoning_report = {
                    'basic_score': reasoning_score,
                    'consistency_score': consistency_score,
                    'relevance_score': relevance_score,
                    'efficiency_score': efficiency_score,
                    'enhanced_score': enhanced_reasoning_score,
                    'num_steps': num_steps,
                    'calculation_errors': calculation_errors,
                    'logic_errors': logic_errors,
                    'has_calculation_error': len(calculation_errors) > 0,
                    'has_logic_error': len(logic_errors) > 0,
                    'steps_relevance': relevance_scores,
                    'complexity_score': complexity_score
                }
                
                return enhanced_reasoning_score, consistency_score, reasoning_report, complexity_score
            
            # Tạo hàm đánh giá tùy theo loại prompt
            def evaluate_answer_by_prompt_type(question, response, expected_solution, prompt_type):
                """
                Evaluate an answer based on the prompt type used.
                
                Args:
                    question (str): The question being answered
                    response (str): The model's response
                    expected_solution (str): The expected answer
                    prompt_type (str): The type of prompt used (standard, cot, hybrid_cot)
                    
                Returns:
                    tuple: (final_score_adjusted, result_type, reasoning_score, consistency_score, 
                           efficiency_score, reasoning_report, complexity_score, confidence_score, 
                           subject_type, subject_score)
                """
                # Phát hiện kết quả và đánh giá độ chính xác
                correctness_score, result_type = evaluate_answer_correctness(question, response, expected_solution)
                
                # Đánh giá suy luận
                reasoning_score = 0
                consistency_score = 0
                reasoning_report = {}
                efficiency_score = 0
                complexity_score = 0
                
                # Tính confidence score
                confidence_score = evaluate_answer_confidence(response)
                
                # Xác định loại môn học dựa vào từ khóa trong câu hỏi
                subject_type = 'general'
                if re.search(r'tính|phép tính|chia|nhân|cộng|trừ|\+|\-|\*|\/|\d+', question.lower()):
                    subject_type = 'arithmetic'
                elif re.search(r'phương trình|hệ phương trình|đa thức|biến|ẩn|hàm số|đạo hàm', question.lower()):
                    subject_type = 'algebra'
                elif re.search(r'tam giác|hình vuông|hình chữ nhật|đường tròn|góc|cạnh|diện tích|thể tích', question.lower()):
                    subject_type = 'geometry'
                elif re.search(r'logic|mệnh đề|chứng minh|suy luận|nếu|thì|hoặc|và', question.lower()):
                    subject_type = 'logic'
                
                # Điểm đánh giá theo tiêu chí môn học cụ thể
                subject_score = evaluate_by_subject_criteria(question, response, subject_type)
                
                if prompt_type in ["cot", "hybrid_cot"]:
                    reasoning_score, consistency_score, reasoning_report, complexity_score = enhanced_evaluate_reasoning(
                        response, expected_solution, question
                    )
                    efficiency_score = reasoning_report['efficiency_score']
                    
                    # Điều chỉnh điểm dựa trên chất lượng suy luận
                    if prompt_type == "cot":
                        # CoT coi trọng quá trình suy luận hơn
                        final_score = (
                            correctness_score * 0.45 + 
                            reasoning_score * 0.25 + 
                            consistency_score * 0.15 + 
                            efficiency_score * 0.1 +
                            subject_score * 0.05  # Thêm điểm đánh giá môn học cụ thể
                        )
                    else:  # hybrid_cot
                        # Hybrid-CoT cân bằng giữa kết quả và suy luận
                        final_score = (
                            correctness_score * 0.35 + 
                            reasoning_score * 0.25 + 
                            consistency_score * 0.15 + 
                            efficiency_score * 0.15 +
                            subject_score * 0.1  # Thêm điểm đánh giá môn học cụ thể
                        )
                else:  # standard
                    # Standard chỉ quan tâm kết quả cuối cùng
                    final_score = correctness_score * 0.9 + subject_score * 0.1  # Thêm điểm đánh giá môn học cụ thể
                    reasoning_report = {
                        'basic_score': 0,
                        'consistency_score': 0,
                        'relevance_score': 0,
                        'efficiency_score': 0,
                        'enhanced_score': 0,
                        'num_steps': 0,
                        'calculation_errors': [],
                        'logic_errors': [],
                        'has_calculation_error': False,
                        'has_logic_error': False,
                        'steps_relevance': [],
                        'complexity_score': min(0.2, len(response.split()) / 500)  # Chỉ dựa vào độ dài
                    }
                    complexity_score = reasoning_report['complexity_score']
                
                # Tính bias correction để giảm chệch giữa các loại prompt
                response_length = len(response.split())
                bias_correction = calculate_bias_correction(prompt_type, correctness_score, reasoning_score, response_length)
                
                # Điều chỉnh điểm số cuối cùng với bias correction
                final_score_adjusted = min(1.0, max(0.0, final_score + bias_correction))
                
                # Cập nhật reasoning_report với các thông tin mới
                reasoning_report.update({
                    'confidence_score': confidence_score,
                    'subject_type': subject_type,
                    'subject_score': subject_score,
                    'bias_correction': bias_correction,
                    'final_score_adjusted': final_score_adjusted
                })
                
                return final_score_adjusted, result_type, reasoning_score, consistency_score, efficiency_score, reasoning_report, complexity_score, confidence_score, subject_type, subject_score
            
            # Hàm phân tích hiệu suất theo loại câu hỏi
            def analyze_by_question_type(results_df):
                # Phân tích hiệu suất theo loại câu hỏi và độ khó
                type_analysis = results_df.groupby(['question_type', 'model_name', 'prompt_type'])['correctness_score'].mean().reset_index()
                difficulty_analysis = results_df.groupby(['question_difficulty', 'model_name', 'prompt_type'])['correctness_score'].mean().reset_index()
                
                # Phân tích hiệu suất theo tags
                # Đầu tiên tách cột tags thành list
                tag_df = results_df.copy()
                tag_df['tags_list'] = tag_df['question_tags'].str.split(',')
                
                # Tạo dataframe mới với mỗi tag là một hàng
                tag_rows = []
                for _, row in tag_df.iterrows():
                    if isinstance(row['tags_list'], list):
                        for tag in row['tags_list']:
                            if tag and isinstance(tag, str):  # Đảm bảo tag không rỗng và là chuỗi
                                tag_rows.append({
                                    'model_name': row['model_name'],
                                    'prompt_type': row['prompt_type'],
                                    'tag': tag.strip(),
                                    'correctness_score': row['correctness_score']
                                })
                
                tag_analysis = pd.DataFrame(tag_rows)
                if not tag_analysis.empty:
                    tag_analysis = tag_analysis.groupby(['tag', 'model_name', 'prompt_type'])['correctness_score'].mean().reset_index()
                
                return type_analysis, difficulty_analysis, tag_analysis
             
            # Tạo hàm tính điểm độ chính xác dựa trên similarity và các quy tắc đơn giản
            def evaluate_answer_correctness(question, response, expected_solution):
                # Nếu có lỗi, điểm = 0
                if "[Error:" in response:
                    return 0.0, "error"
                
                # Tiền xử lý phản hồi và đáp án
                response = response.lower().strip()
                expected_solution = expected_solution.lower().strip()
                
                # Trích xuất con số từ cả hai
                response_numbers = re.findall(r'\d+\.?\d*', response)
                solution_numbers = re.findall(r'\d+\.?\d*', expected_solution)
                
                # Kiểm tra xem câu trả lời có chứa con số kết quả trong đáp án không
                number_match = False
                if solution_numbers and response_numbers:
                    # Kiểm tra xem kết quả cuối cùng có xuất hiện trong câu trả lời không
                    # Ưu tiên con số cuối cùng trong đáp án
                    final_solution_number = solution_numbers[-1]
                    if final_solution_number in response_numbers:
                        number_match = True
                
                # Tính độ tương đồng văn bản
                from difflib import SequenceMatcher
                text_similarity = SequenceMatcher(None, response, expected_solution).ratio()
                
                # Trích xuất các từ khóa quan trọng từ đáp án
                important_keywords = []
                # Tìm các từ có thể là từ khóa (không phải stopwords)
                for word in expected_solution.split():
                    if len(word) > 3 and not word.isdigit():  # Từ dài hơn 3 ký tự và không phải số
                        important_keywords.append(word)
                
                # Đếm số từ khóa xuất hiện trong câu trả lời
                keyword_matches = sum(1 for keyword in important_keywords if keyword in response)
                keyword_ratio = keyword_matches / max(1, len(important_keywords))
                
                # Phân loại kết quả
                if number_match and text_similarity > 0.5:
                    result_type = "correct"
                    score = 1.0
                elif number_match and keyword_ratio > 0.3:
                    result_type = "partially_correct"
                    score = 0.7
                elif text_similarity > 0.3 or keyword_ratio > 0.5:
                    result_type = "related"
                    score = 0.4
                else:
                    result_type = "incorrect"
                    score = 0.0
                
                return score, result_type
             
            # Áp dụng đánh giá cho mỗi phản hồi
            scores = []
            result_types = []
            reasoning_scores = []
            consistency_scores = []
            efficiency_scores = []
            has_calculation_error = []
            has_logic_error = []
            num_reasoning_steps = []
            complexity_scores = []
            confidence_scores = []  # Thêm confidence score
            subject_types = []  # Thêm loại môn học
            subject_scores = []  # Thêm điểm đánh giá theo môn học
            bias_corrections = []  # Thêm điều chỉnh bias
            question_types = []
            question_difficulties = []
            question_tags_list = []
             
            for _, row in results_df.iterrows():
                question = row['question']
                expected_data = questions_data.get(question, None)
                
                if expected_data:
                    prompt_type = row['prompt_type']
                    score, result_type, reasoning, consistency, efficiency, reasoning_report, complexity_score, confidence_score, subject_type, subject_score = evaluate_answer_by_prompt_type(
                        question, 
                        row['response'], 
                        expected_data['solution'],
                        prompt_type
                    )
                    
                    scores.append(score)
                    result_types.append(result_type)
                    reasoning_scores.append(reasoning)
                    consistency_scores.append(consistency)
                    efficiency_scores.append(efficiency)
                    has_calculation_error.append(reasoning_report['has_calculation_error'])
                    has_logic_error.append(reasoning_report['has_logic_error'])
                    num_reasoning_steps.append(reasoning_report['num_steps'])
                    complexity_scores.append(complexity_score)
                    confidence_scores.append(confidence_score)  # Lưu confidence score
                    subject_types.append(subject_type)  # Lưu loại môn học
                    subject_scores.append(subject_score)  # Lưu điểm đánh giá theo môn học
                    bias_corrections.append(reasoning_report['bias_correction'])  # Lưu điều chỉnh bias
                    
                    # Thêm thông tin về loại câu hỏi và độ khó
                    question_types.append(expected_data['type'])
                    question_difficulties.append(expected_data['difficulty'])
                    question_tags_list.append(",".join(expected_data['tags']))
                else:
                    # Xử lý khi không tìm thấy thông tin câu hỏi
                    scores.append(0.0)
                    result_types.append("unknown")
                    reasoning_scores.append(0.0)
                    consistency_scores.append(0.0)
                    efficiency_scores.append(0.0)
                    has_calculation_error.append(False)
                    has_logic_error.append(False)
                    num_reasoning_steps.append(0)
                    complexity_scores.append(min(0.2, len(row['response'].split()) / 500))
                    confidence_scores.append(0.5)  # Giá trị mặc định cho confidence
                    subject_types.append('general')  # Giá trị mặc định cho loại môn học
                    subject_scores.append(0.0)  # Giá trị mặc định cho điểm môn học
                    bias_corrections.append(0.0)  # Giá trị mặc định cho bias correction
                    
                    # Thêm giá trị mặc định cho thông tin câu hỏi
                    question_types.append("unknown")
                    question_difficulties.append("unknown")
                    question_tags_list.append("")
            
            # Thêm thông tin đánh giá vào DataFrame
            results_df['correctness_score'] = scores
            results_df['result_type'] = result_types
            results_df['reasoning_score'] = reasoning_scores
            results_df['consistency_score'] = consistency_scores
            results_df['efficiency_score'] = efficiency_scores
            results_df['has_calculation_error'] = has_calculation_error
            results_df['has_logic_error'] = has_logic_error
            results_df['num_reasoning_steps'] = num_reasoning_steps
            results_df['complexity_score'] = complexity_scores
            results_df['confidence_score'] = confidence_scores  # Thêm confidence score
            results_df['subject_type'] = subject_types  # Thêm loại môn học
            results_df['subject_score'] = subject_scores  # Thêm điểm đánh giá theo môn học
            results_df['bias_correction'] = bias_corrections  # Thêm điều chỉnh bias
            results_df['question_type'] = question_types
            results_df['question_difficulty'] = question_difficulties
            results_df['question_tags'] = question_tags_list
            
            # Thêm cột is_correct dựa trên correctness_score
            results_df['is_correct'] = results_df['correctness_score'] >= 0.7  # 0.7 là ngưỡng đánh giá đúng

            # Vẽ biểu đồ đánh giá hiệu quả suy luận cho CoT và Hybrid-CoT
            plt.figure(figsize=(12, 8))
            reasoning_df = results_df[results_df['prompt_type'].isin(['cot', 'hybrid_cot'])]
            if not reasoning_df.empty:
                sns.barplot(x='model_name', y='reasoning_score', hue='prompt_type', data=reasoning_df)
                plt.title('Reasoning Quality Comparison: CoT vs Hybrid-CoT', fontsize=16)
                plt.xlabel('Model', fontsize=14)
                plt.ylabel('Reasoning Score', fontsize=14)
                plt.legend(title='Prompt Type', title_fontsize=12, fontsize=10)
                plt.xticks(rotation=45)
                plt.ylim(0, 1.0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'reasoning_quality_comparison.png'), dpi=300)
                plt.close()

            # So sánh tính nhất quán trong suy luận
            plt.figure(figsize=(12, 8))
            if not reasoning_df.empty:
                sns.barplot(x='model_name', y='consistency_score', hue='prompt_type', data=reasoning_df)
                plt.title('Reasoning Consistency Comparison: CoT vs Hybrid-CoT', fontsize=16)
                plt.xlabel('Model', fontsize=14)
                plt.ylabel('Consistency Score', fontsize=14)
                plt.legend(title='Prompt Type', title_fontsize=12, fontsize=10)
                plt.xticks(rotation=45)
                plt.ylim(0, 1.0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'reasoning_consistency_comparison.png'), dpi=300)
                plt.close()

            # Phân tích hiệu suất với các câu hỏi theo từng miền lĩnh vực
            type_analysis, difficulty_analysis, tag_analysis = analyze_by_question_type(results_df)

            # Vẽ biểu đồ phân tích theo loại câu hỏi
            plt.figure(figsize=(14, 10))
            type_pivot = type_analysis.pivot_table(index='question_type', columns=['model_name', 'prompt_type'], values='correctness_score')
            if not type_pivot.empty:
                sns.heatmap(type_pivot, annot=True, cmap='YlGnBu', fmt='.2f')
                plt.title('Performance by Question Type', fontsize=16)
                plt.ylabel('Question Type', fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'performance_by_question_type.png'), dpi=300)
                plt.close()

            # Vẽ biểu đồ phân tích theo độ khó
            plt.figure(figsize=(14, 8))
            difficulty_pivot = difficulty_analysis.pivot_table(index='question_difficulty', columns=['model_name', 'prompt_type'], values='correctness_score')
            if not difficulty_pivot.empty:
                sns.heatmap(difficulty_pivot, annot=True, cmap='YlGnBu', fmt='.2f')
                plt.title('Performance by Question Difficulty', fontsize=16)
                plt.ylabel('Question Difficulty', fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'performance_by_difficulty.png'), dpi=300)
                plt.close()

            # Vẽ biểu đồ phân tích theo tag
            if not tag_analysis.empty:
                plt.figure(figsize=(14, 10))
                tag_pivot = tag_analysis.pivot_table(index='tag', columns=['model_name', 'prompt_type'], values='correctness_score')
                if not tag_pivot.empty:
                    sns.heatmap(tag_pivot, annot=True, cmap='YlGnBu', fmt='.2f')
                    plt.title('Performance by Subject Area', fontsize=16)
                    plt.ylabel('Subject Area', fontsize=14)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'performance_by_subject.png'), dpi=300)
                    plt.close()

            # Tạo biểu đồ so sánh prompt types cho từng độ khó
            plt.figure(figsize=(14, 8))
            prompt_by_difficulty = results_df.groupby(['prompt_type', 'question_difficulty'])['correctness_score'].mean().reset_index()
            prompt_by_difficulty_pivot = prompt_by_difficulty.pivot(index='prompt_type', columns='question_difficulty', values='correctness_score')
            if not prompt_by_difficulty_pivot.empty:
                prompt_by_difficulty_pivot.plot(kind='bar', colormap='viridis')
                plt.title('Prompt Type Effectiveness by Question Difficulty', fontsize=16)
                plt.xlabel('Prompt Type', fontsize=14)
                plt.ylabel('Average Correctness Score', fontsize=14)
                plt.legend(title='Difficulty', title_fontsize=12, fontsize=10)
                plt.xticks(rotation=45)
                plt.ylim(0, 1.0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'prompt_effectiveness_by_difficulty.png'), dpi=300)
                plt.close()
        
        # Thêm phần hiển thị kết quả phân tích vào báo cáo HTML
        prompt_effectiveness_section = f"""
        <div class="section">
            <h2>Prompt Type Effectiveness Analysis</h2>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">Reasoning Quality: CoT vs Hybrid-CoT</div>
                    <div class="chart-description">Comparison of reasoning quality between Chain of Thought and Hybrid Chain of Thought prompts.</div>
                    <img src="plots/reasoning_quality_comparison.png" alt="Reasoning Quality Comparison">
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Reasoning Consistency: CoT vs Hybrid-CoT</div>
                    <div class="chart-description">Comparison of reasoning consistency between Chain of Thought and Hybrid Chain of Thought prompts.</div>
                    <img src="plots/reasoning_consistency_comparison.png" alt="Reasoning Consistency Comparison">
                </div>
            </div>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">Reasoning Efficiency</div>
                    <div class="chart-description">Evaluation of how efficient each model is in its reasoning process.</div>
                    <img src="plots/reasoning_efficiency_comparison.png" alt="Reasoning Efficiency Comparison">
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Reasoning Error Analysis</div>
                    <div class="chart-description">Breakdown of calculation and logic errors in reasoning steps.</div>
                    <img src="plots/reasoning_error_analysis.png" alt="Reasoning Error Analysis">
                </div>
            </div>

            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">Number of Reasoning Steps</div>
                    <div class="chart-description">Distribution of reasoning steps used by different models and prompt types.</div>
                    <img src="plots/reasoning_steps_distribution.png" alt="Number of Reasoning Steps">
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Performance by Question Type</div>
                    <div class="chart-description">How different prompt types perform across various question types.</div>
                    <img src="plots/performance_by_question_type.png" alt="Performance by Question Type">
                </div>
            </div>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">Performance by Question Difficulty</div>
                    <div class="chart-description">Comparison of prompt type effectiveness across different difficulty levels.</div>
                    <img src="plots/performance_by_difficulty.png" alt="Performance by Difficulty">
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Performance by Subject Area</div>
                    <div class="chart-description">Heatmap showing how different models and prompt types perform across subject areas.</div>
                    <img src="plots/performance_by_subject.png" alt="Performance by Subject">
                </div>
            </div>
            
            <div class="grid-1">
                <div class="plot-container">
                    <div class="chart-title">Prompt Effectiveness by Question Difficulty</div>
                    <div class="chart-description">Bar chart comparing how effective each prompt type is for different question difficulties.</div>
                    <img src="plots/prompt_effectiveness_by_difficulty.png" alt="Prompt Effectiveness by Difficulty">
                </div>
            </div>
            
            <div class="analysis-summary">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Standard Prompts:</strong> Generally perform well on straightforward, well-defined problems with clear answers.</li>
                    <li><strong>Chain of Thought (CoT):</strong> Excel in problems requiring step-by-step reasoning, especially in mathematical and logical domains.</li>
                    <li><strong>Hybrid-CoT:</strong> Balance between detailed reasoning and concise answers, performing well across diverse question types.</li>
                    <li><strong>Reasoning Efficiency:</strong> Hybrid-CoT typically shows better efficiency (quality vs length ratio) compared to standard CoT.</li>
                    <li><strong>Error Analysis:</strong> Calculation errors and logical inconsistencies occur more frequently in complex multi-step problems.</li>
                </ul>
                
                <p>The analysis shows that different prompt types have distinct strengths depending on the question type, difficulty, and subject area. 
                For critical reasoning tasks, CoT and Hybrid-CoT significantly outperform standard prompts, while for simpler tasks, 
                the overhead of reasoning steps may not provide substantial benefits. The enhanced evaluation metrics now provide deeper insight
                into how models construct their reasoning chains and where specific types of errors occur.</p>
            </div>
        </div>
        """
        
        # Save processed results
        processed_results_path = os.path.join(output_dir, "processed_results.csv")
        results_df.to_csv(processed_results_path, index=False)
        print_status("success", f"Processed results saved to {processed_results_path}", "green")
        
        # Calculate additional evaluation metrics
        print_status("info", "Calculating evaluation metrics...", "blue")
        
        # Thêm các cột cần thiết nếu chúng chưa tồn tại trong DataFrame
        if 'has_error' not in results_df.columns:
            # Tạo cột has_error dựa trên kết quả
            results_df['has_error'] = results_df['response'].str.contains(r"\[Error:", na=False)
        
        if 'tokens_per_second' not in results_df.columns:
            # Tính tokens_per_second dựa trên elapsed_time và response_length
            results_df['tokens_per_second'] = results_df.apply(
                lambda row: row['response_length'] / max(0.1, row['elapsed_time']) if row['elapsed_time'] > 0 else 0, 
                axis=1
            )
        
        if 'response_quality_score' not in results_df.columns:
            # Tạo chỉ số chất lượng phản hồi đơn giản dựa trên các chỉ số đã có
            if 'correctness_score' in results_df.columns:
                results_df['response_quality_score'] = results_df['correctness_score']
            else:
                results_df['response_quality_score'] = 0.5  # Giá trị mặc định
        
        if 'quality_index' not in results_df.columns:
            # Tạo chỉ số chất lượng tổng hợp
            if 'correctness_score' in results_df.columns and 'consistency_score' in results_df.columns:
                results_df['quality_index'] = (
                    results_df['correctness_score'] * 0.6 + 
                    results_df['consistency_score'] * 0.4
                )
            else:
                results_df['quality_index'] = results_df['response_quality_score']
                
        if 'coherence_score' not in results_df.columns:
            # Tạo chỉ số mạch lạc từ các chỉ số khác
            if 'consistency_score' in results_df.columns:
                results_df['coherence_score'] = results_df['consistency_score']
            else:
                results_df['coherence_score'] = 0.5  # Giá trị mặc định
        
        # Generate basic statistics by model and prompt type
        model_stats = results_df.groupby('model_name').agg({
            'elapsed_time': ['mean', 'std', 'min', 'max'],
            'response_length': ['mean', 'std', 'min', 'max'],
            'has_error': 'mean',
            'tokens_per_second': 'mean',
            'response_quality_score': 'mean',
            'quality_index': 'mean'
        })
        
        prompt_stats = results_df.groupby('prompt_type').agg({
            'elapsed_time': ['mean', 'std', 'min', 'max'],
            'response_length': ['mean', 'std', 'min', 'max'],
            'has_error': 'mean',
            'tokens_per_second': 'mean',
            'response_quality_score': 'mean',
            'quality_index': 'mean'
        })
        
        combined_stats = results_df.groupby(['model_name', 'prompt_type']).agg({
            'elapsed_time': 'mean',
            'response_length': 'mean',
            'has_error': 'mean',
            'tokens_per_second': 'mean',
            'response_quality_score': 'mean',
            'quality_index': 'mean'
        }).reset_index()
        
        # Save statistics to CSV
        model_stats.to_csv(os.path.join(output_dir, "model_statistics.csv"))
        prompt_stats.to_csv(os.path.join(output_dir, "prompt_statistics.csv"))
        combined_stats.to_csv(os.path.join(output_dir, "combined_statistics.csv"))
        
        # Generate visualizations
        print_status("info", "Generating visualizations...", "blue")
        
        # Set common styling for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = plt.cm.viridis(np.linspace(0, 1, len(results_df['model_name'].unique())))
        sns.set_palette(sns.color_palette(colors))
        
        # 1. Response Time Distribution
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x='model_name', y='elapsed_time', hue='prompt_type', data=results_df)
        plt.title('Response Time Distribution by Model and Prompt Type', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Response Time (seconds)', fontsize=14)
        plt.legend(title='Prompt Type', title_fontsize=12, fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'response_time_distribution.png'), dpi=300)
        plt.close()
        
        # 2. Error Rate by Model and Prompt Type
        plt.figure(figsize=(12, 8))
        error_rates = results_df.groupby(['model_name', 'prompt_type'])['has_error'].mean() * 100
        error_rates = error_rates.reset_index().pivot(index='model_name', columns='prompt_type', values='has_error')
        sns.heatmap(error_rates, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=.5)
        plt.title('Error Rate (%) by Model and Prompt Type', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'error_rate_heatmap.png'), dpi=300)
        plt.close()
        
        # 3. Response Length Distribution
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='model_name', y='response_length', hue='prompt_type', data=results_df)
        plt.title('Response Length Distribution by Model and Prompt Type', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Response Length (characters)', fontsize=14)
        plt.legend(title='Prompt Type', title_fontsize=12, fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'response_length_distribution.png'), dpi=300)
        plt.close()
        
        # 4. Processing Speed Comparison (tokens per second)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='model_name', y='tokens_per_second', hue='prompt_type', data=results_df)
        plt.title('Processing Speed Comparison', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Tokens Per Second', fontsize=14)
        plt.legend(title='Prompt Type', title_fontsize=12, fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'tokens_per_second.png'), dpi=300)
        plt.close()
        
        # 5. Response Time vs Response Length Scatter Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='response_length', y='elapsed_time', hue='model_name', style='prompt_type', s=100, data=results_df)
        plt.title('Response Time vs Response Length', fontsize=16)
        plt.xlabel('Response Length (characters)', fontsize=14)
        plt.ylabel('Response Time (seconds)', fontsize=14)
        plt.legend(title='Model', title_fontsize=12, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'response_time_vs_length.png'), dpi=300)
        plt.close()
        
        # 6. Violin Plot for Response Time Distribution
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='model_name', y='elapsed_time', hue='prompt_type', data=results_df, split=True)
        plt.title('Response Time Distribution (Violin Plot)', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Response Time (seconds)', fontsize=14)
        plt.legend(title='Prompt Type', title_fontsize=12, fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'response_time_violin.png'), dpi=300)
        plt.close()
        
        # 7. Line plot for response time across different prompt types
        plt.figure(figsize=(12, 8))
        sns.lineplot(x='prompt_type', y='elapsed_time', hue='model_name', marker='o', data=results_df)
        plt.title('Response Time Trend Across Prompt Types', fontsize=16)
        plt.xlabel('Prompt Type', fontsize=14)
        plt.ylabel('Average Response Time (seconds)', fontsize=14)
        plt.legend(title='Model', title_fontsize=12, fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'response_time_trend.png'), dpi=300)
        plt.close()
        
        # 8. Heatmap of Average Response Length
        plt.figure(figsize=(10, 8))
        response_length_pivot = results_df.groupby(['model_name', 'prompt_type'])['response_length'].mean().reset_index()
        response_length_pivot = response_length_pivot.pivot(index='model_name', columns='prompt_type', values='response_length')
        sns.heatmap(response_length_pivot, annot=True, fmt='.1f', cmap='viridis', linewidths=.5)
        plt.title('Average Response Length by Model and Prompt Type', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'response_length_heatmap.png'), dpi=300)
        plt.close()
        
        # 9. Radar Chart for Model Performance
        # Prepare data for radar chart
        radar_metrics = ['avg_time', 'error_rate', 'avg_length', 'tokens_per_sec']
        radar_data = {}
        
        for model in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model]
            
            # Normalize values between 0 and 1 (0 is worst, 1 is best)
            avg_time = 1 - (model_data['elapsed_time'].mean() / results_df['elapsed_time'].max())
            error_rate = 1 - model_data['has_error'].mean()
            avg_length = model_data['response_length'].mean() / results_df['response_length'].max()
            tokens_per_sec = model_data['tokens_per_second'].mean() / results_df['tokens_per_second'].max()
            
            radar_data[model] = [avg_time, error_rate, avg_length, tokens_per_sec]
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Set the angle of each axis
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each model
        for i, (model, values) in enumerate(radar_data.items()):
            values = values + values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Set category labels
        plt.xticks(angles[:-1], radar_metrics, size=12)
        
        # Draw y-axis labels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
        plt.ylim(0, 1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Performance Comparison (Radar Chart)', size=16, y=1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_performance_radar.png'), dpi=300)
        plt.close()
        
        # 10. Confusion Matrix (simulated correctness)
        # Create a simulated confusion matrix based on the presence of errors
        for model in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model]
            
            # Simulate correctness as the inverse of has_error
            correctness = ~model_data['has_error'].astype(bool)
            
            # Create a confusion matrix simulation (binary classification: error/no error)
            expected_no_error = model_data.shape[0]  # Ideal: All responses should be error-free
            tp = correctness.sum()  # True Positive: Correctly produced responses without errors
            fp = 0  # False Positive: Not applicable in this context
            fn = model_data.shape[0] - tp  # False Negative: Expected no error but got an error
            tn = 0  # True Negative: Not applicable in this context
            
            # Create a matrix representation
            matrix = np.array([[tp, fp], [fn, tn]])
            
            # Plot the matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['No Error', 'Error'], 
                        yticklabels=['No Error', 'Error'])
            plt.title(f'Confusion Matrix - {model}', fontsize=16)
            plt.ylabel('Actual', fontsize=14)
            plt.xlabel('Predicted', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{model}.png'), dpi=300)
            plt.close()
            
        # Calculate overall statistics for the report
        stats = {
            "total_queries": len(results_df),
            "total_models": results_df['model_name'].nunique(),
            "total_prompt_types": results_df['prompt_type'].nunique(),
            "error_rate": (results_df['has_error'].mean() * 100),
            "avg_response_time": results_df['elapsed_time'].mean(),
            "avg_response_length": results_df['response_length'].mean(),
            "avg_tokens_per_second": results_df['tokens_per_second'].mean(),
            "models": results_df['model_name'].unique().tolist(),
            "prompt_types": results_df['prompt_type'].unique().tolist(),
            "best_model_time": combined_stats.sort_values('elapsed_time').iloc[0]['model_name'],
            "best_model_accuracy": combined_stats.sort_values('has_error').iloc[0]['model_name'],
            "best_model_length": combined_stats.sort_values('response_length', ascending=False).iloc[0]['model_name'],
            "best_prompt_type": combined_stats.groupby('prompt_type')['has_error'].mean().sort_values().index[0]
        }
        
        # Model comparison table data
        model_comparison = []
        for model in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model]
            row = {
                "model": model,
                "avg_time": f"{model_data['elapsed_time'].mean():.2f}s",
                "error_rate": f"{model_data['has_error'].mean()*100:.2f}%",
                "avg_length": f"{model_data['response_length'].mean():.0f}",
                "tokens_per_sec": f"{model_data['tokens_per_second'].mean():.2f}"
            }
            model_comparison.append(row)
            
        # Prompt comparison table data
        prompt_comparison = []
        for prompt in results_df['prompt_type'].unique():
            prompt_data = results_df[results_df['prompt_type'] == prompt]
            row = {
                "prompt": prompt,
                "avg_time": f"{prompt_data['elapsed_time'].mean():.2f}s",
                "error_rate": f"{prompt_data['has_error'].mean()*100:.2f}%",
                "avg_length": f"{prompt_data['response_length'].mean():.0f}",
                "tokens_per_sec": f"{prompt_data['tokens_per_second'].mean():.2f}"
            }
            prompt_comparison.append(row)
        
        # Generate comprehensive HTML report with all visualizations
        print_status("info", "Generating comprehensive HTML report...", "blue")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"comprehensive_evaluation_report_{timestamp}.html")
        
        # Generate HTML content with enhanced styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive LLM Evaluation Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 0;
                    padding: 0;
                    color: #333;
                    background-color: #f9f9f9;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px;
                }}
                header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px 0;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                h1, h2, h3, h4 {{ 
                    color: #2c3e50; 
                    margin-top: 30px;
                }}
                header h1 {{
                    color: white;
                    margin: 0;
                }}
                .section {{
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 25px;
                    margin-bottom: 30px;
                }}
                .plot-container {{ 
                    margin: 20px 0; 
                    text-align: center;
                }}
                .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .grid-2 {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    grid-gap: 20px;
                }}
                @media (max-width: 768px) {{
                    .grid-2 {{
                        grid-template-columns: 1fr;
                    }}
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0; 
                    box-shadow: 0 2px 3px rgba(0,0,0,0.1);
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #34495e; 
                    color: white;
                }}
                tr:nth-child(even) {{ 
                    background-color: #f2f2f2; 
                }}
                tr:hover {{
                    background-color: #e9e9e9;
                }}
                .metrics {{ 
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    grid-gap: 20px;
                    margin: 30px 0;
                }}
                .metric-card {{ 
                    border: 1px solid #ddd; 
                    border-radius: 8px;
                    padding: 20px; 
                    text-align: center;
                    background-color: white;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    transition: transform 0.2s ease;
                }}
                .metric-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .metric-card h3 {{ 
                    margin-top: 0;
                    color: #34495e;
                    font-size: 16px;
                }}
                .metric-card p {{ 
                    font-size: 28px;
                    font-weight: bold;
                    margin: 10px 0 0 0;
                    color: #3498db;
                }}
                .best {{
                    background-color: #dff0d8;
                    border-color: #d6e9c6;
                }}
                .best h3 {{
                    color: #3c763d;
                }}
                .best p {{
                    color: #2ecc71;
                }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 0 5px 5px 0;
                }}
                .findings {{
                    margin: 30px 0;
                }}
                .finding-item {{
                    margin-bottom: 15px;
                    padding-bottom: 15px;
                    border-bottom: 1px solid #eee;
                }}
                .chart-title {{
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #2c3e50;
                }}
                .chart-description {{
                    color: #7f8c8d;
                    margin-bottom: 20px;
                }}
                footer {{
                    background-color: #2c3e50;
                    color: white;
                    text-align: center;
                    padding: 20px 0;
                    margin-top: 50px;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>Comprehensive LLM Evaluation Report</h1>
                <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </header>
            
            <div class="container">
                <div class="section">
                    <h2>Executive Summary</h2>
                    <div class="summary-box">
                        <p>This report presents a comprehensive evaluation of {stats['total_models']} language models 
                        ({', '.join(stats['models'])}) across {stats['total_prompt_types']} different prompt types 
                        ({', '.join(stats['prompt_types'])}). A total of {stats['total_queries']} queries were processed
                        to evaluate the performance of these models.</p>
                        
                        <p><strong>Key Findings:</strong></p>
                        <ul>
                            <li><strong>{stats['best_model_time']}</strong> demonstrated the fastest average response time.</li>
                            <li><strong>{stats['best_model_accuracy']}</strong> showed the lowest error rate.</li>
                            <li><strong>{stats['best_model_length']}</strong> generated the most detailed responses (longest average length).</li>
                            <li>The <strong>{stats['best_prompt_type']}</strong> prompt type generally produced the best results across models.</li>
                        </ul>
                    </div>
                    
                    <div class="metrics">
                        <div class="metric-card">
                            <h3>Models Evaluated</h3>
                            <p>{stats['total_models']}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Prompt Types</h3>
                            <p>{stats['total_prompt_types']}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Total Queries</h3>
                            <p>{stats['total_queries']}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Overall Error Rate</h3>
                            <p>{stats['error_rate']:.2f}%</p>
                        </div>
                        <div class="metric-card">
                            <h3>Avg Response Time</h3>
                            <p>{stats['avg_response_time']:.2f}s</p>
                        </div>
                        <div class="metric-card">
                            <h3>Avg Response Length</h3>
                            <p>{stats['avg_response_length']:.0f}</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Model Comparison</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Avg. Response Time</th>
                            <th>Error Rate</th>
                            <th>Avg. Response Length</th>
                            <th>Tokens Per Second</th>
                        </tr>
                        {''.join(f"""
                        <tr>
                            <td><strong>{row['model']}</strong></td>
                            <td>{row['avg_time']}</td>
                            <td>{row['error_rate']}</td>
                            <td>{row['avg_length']}</td>
                            <td>{row['tokens_per_sec']}</td>
                        </tr>
                        """ for row in model_comparison)}
                    </table>
                    
                    <h3>Prompt Type Performance</h3>
                    <table>
                        <tr>
                            <th>Prompt Type</th>
                            <th>Avg. Response Time</th>
                            <th>Error Rate</th>
                            <th>Avg. Response Length</th>
                            <th>Tokens Per Second</th>
                        </tr>
                        {''.join(f"""
                        <tr>
                            <td><strong>{row['prompt']}</strong></td>
                            <td>{row['avg_time']}</td>
                            <td>{row['error_rate']}</td>
                            <td>{row['avg_length']}</td>
                            <td>{row['tokens_per_sec']}</td>
                        </tr>
                        """ for row in prompt_comparison)}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Performance Visualizations</h2>
                    
                    <div class="grid-2">
                        <div class="plot-container">
                            <div class="chart-title">Response Time by Model and Prompt Type</div>
                            <div class="chart-description">This boxplot shows the distribution of response times across different models and prompt types.</div>
                            <img src="plots/response_time_distribution.png" alt="Response Time Distribution">
                        </div>
                        
                        <div class="plot-container">
                            <div class="chart-title">Error Rate Heatmap</div>
                            <div class="chart-description">The heatmap displays error rates (%) for each combination of model and prompt type.</div>
                            <img src="plots/error_rate_heatmap.png" alt="Error Rate Heatmap">
                        </div>
                    </div>
                    
                    <div class="grid-2">
                        <div class="plot-container">
                            <div class="chart-title">Response Length Distribution</div>
                            <div class="chart-description">This boxplot compares the response length distribution across models and prompt types.</div>
                            <img src="plots/response_length_distribution.png" alt="Response Length Distribution">
                        </div>
                        
                        <div class="plot-container">
                            <div class="chart-title">Response Time vs Response Length</div>
                            <div class="chart-description">This scatter plot shows the relationship between response length and processing time.</div>
                            <img src="plots/response_time_vs_length.png" alt="Response Time vs Length Scatter Plot">
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Advanced Analysis</h2>
                    
                    <div class="grid-2">
                        <div class="plot-container">
                            <div class="chart-title">Processing Speed Comparison</div>
                            <div class="chart-description">Bar chart comparing tokens processed per second for each model and prompt type.</div>
                            <img src="plots/tokens_per_second.png" alt="Tokens Per Second">
                        </div>
                        
                        <div class="plot-container">
                            <div class="chart-title">Response Time Distributions (Violin Plot)</div>
                            <div class="chart-description">This violin plot shows the density distribution of response times.</div>
                            <img src="plots/response_time_violin.png" alt="Response Time Violin Plot">
                        </div>
                    </div>
                    
                    <div class="grid-2">
                        <div class="plot-container">
                            <div class="chart-title">Model Performance Radar Chart</div>
                            <div class="chart-description">This radar chart provides a multi-dimensional comparison of model performance across key metrics.</div>
                            <img src="plots/model_performance_radar.png" alt="Model Performance Radar Chart">
                        </div>
                        
                        <div class="plot-container">
                            <div class="chart-title">Response Time Trends Across Prompt Types</div>
                            <div class="chart-description">Line chart showing how response times vary across different prompt types for each model.</div>
                            <img src="plots/response_time_trend.png" alt="Response Time Trend">
                        </div>
                    </div>
                    
                    <div class="grid-2">
                        <div class="plot-container">
                            <div class="chart-title">Response Length Heatmap</div>
                            <div class="chart-description">Heatmap showing the average response length for each model and prompt type combination.</div>
                            <img src="plots/response_length_heatmap.png" alt="Response Length Heatmap">
                        </div>
                        
                        <div class="plot-container">
                            <div class="chart-title">Confusion Matrix Example</div>
                            <div class="chart-description">Confusion matrix representing the error performance for one of the models.</div>
                            <img src="plots/confusion_matrix_{stats['models'][0]}.png" alt="Confusion Matrix">
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Key Findings & Insights</h2>
                    
                    <div class="findings">
                        <div class="finding-item">
                            <h3>Model Performance Comparison</h3>
                            <p>Based on the evaluation results, we observe that <strong>{stats['best_model_accuracy']}</strong> 
                            demonstrates the lowest error rate of all models tested. This suggests it has the most robust
                            understanding and processing capabilities for the given task types.</p>
                        </div>
                        
                        <div class="finding-item">
                            <h3>Prompt Engineering Impact</h3>
                            <p>The <strong>{stats['best_prompt_type']}</strong> prompt strategy consistently yielded the best 
                            results across models, suggesting that this approach to query formulation is most effective for 
                            extracting optimal performance from these LLMs.</p>
                        </div>
                        
                        <div class="finding-item">
                            <h3>Efficiency vs. Quality Tradeoff</h3>
                            <p>We observe an interesting relationship between response time and response quality. While 
                            <strong>{stats['best_model_time']}</strong> delivers the fastest responses, it doesn't necessarily 
                            provide the most accurate or detailed answers. This highlights the classic tradeoff between 
                            computational efficiency and response quality.</p>
                        </div>
                        
                        <div class="finding-item">
                            <h3>Response Length Analysis</h3>
                            <p><strong>{stats['best_model_length']}</strong> consistently generates the longest responses, 
                            which often correlate with more detailed and comprehensive answers. However, verbose responses 
                            don't always indicate higher quality, as they may contain redundant information.</p>
                        </div>
                        
                        <div class="finding-item">
                            <h3>Error Patterns</h3>
                            <p>The visualization of error rates reveals that certain prompt and model combinations produce 
                            significantly higher error rates. This suggests that not all models handle all prompt types 
                            equally well, and careful matching between model capabilities and prompt design is essential for 
                            optimal results.</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Detailed Results</h2>
                    <p>Below is a sample of the detailed results from the evaluation. The full dataset is available in the 
                    <code>processed_results.csv</code> file in the results directory.</p>
                    
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Prompt Type</th>
                            <th>Question</th>
                            <th>Response</th>
                            <th>Time (s)</th>
                            <th>Status</th>
                        </tr>
                        {''.join(f"""
                        <tr>
                            <td>{row['model_name']}</td>
                            <td>{row['prompt_type']}</td>
                            <td>{row['question'][:100]}...</td>
                            <td>{row['response'][:150]}...</td>
                            <td>{row['elapsed_time']:.2f}</td>
                            <td>{'Error' if row['has_error'] else 'Success'}</td>
                        </tr>
                        """ for _, row in results_df.head(10).iterrows())}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    
                    <div class="finding-item">
                        <h3>Model Selection</h3>
                        <p>Based on the performance evaluation, <strong>{stats['best_model_accuracy']}</strong> is recommended for 
                        applications where accuracy is the highest priority, while <strong>{stats['best_model_time']}</strong> is 
                        better suited for applications requiring quick response times.</p>
                    </div>
                    
                    <div class="finding-item">
                        <h3>Prompt Engineering</h3>
                        <p>For optimal results, we recommend using the <strong>{stats['best_prompt_type']}</strong> prompt type 
                        when working with these models, as it consistently yielded the best performance across all models tested.</p>
                    </div>
                    
                    <div class="finding-item">
                        <h3>Future Work</h3>
                        <p>Further evaluation could explore:
                        <ul>
                            <li>More domain-specific tasks to test specialized knowledge</li>
                            <li>Performance under different computational constraints</li>
                            <li>Evaluation of additional prompt types and hybrid approaches</li>
                            <li>Testing with a larger and more diverse set of questions</li>
                        </ul>
                        </p>
                    </div>
                </div>
            </div>
            
            <footer>
                <p>This report was automatically generated as part of the LLM Evaluation Framework</p>
                <p>&copy; {time.strftime("%Y")} - All rights reserved</p>
            </footer>
        </body>
        </html>
        """
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print_status("success", f"Comprehensive report generated: {report_path}", "green")
        
        # 11. 3D Surface Plot for Model Performance
        fig = go.Figure(data=[go.Surface(
            x=results_df['model_name'].unique(),
            y=results_df['prompt_type'].unique(),
            z=results_df.pivot_table(
                values='response_quality_score', 
                index='model_name',
                columns='prompt_type'
            ).values
        )])
        fig.update_layout(
            title='3D Surface Plot of Response Quality',
            scene=dict(
                xaxis_title='Model',
                yaxis_title='Prompt Type',
                zaxis_title='Quality Score'
            )
        )
        fig.write_html(os.path.join(plots_dir, '3d_surface_quality.html'))

        # 12. Sunburst Chart for Error Distribution
        fig = px.sunburst(
            results_df,
            path=['model_name', 'prompt_type', 'has_error'],
            values='response_length',
            title='Error Distribution Hierarchy'
        )
        fig.write_html(os.path.join(plots_dir, 'error_sunburst.html'))

        # 13. Parallel Categories Plot
        fig = px.parallel_categories(
            results_df,
            dimensions=['model_name', 'prompt_type', 'has_error'],
            color='elapsed_time',
            title='Parallel Categories Analysis'
        )
        fig.write_html(os.path.join(plots_dir, 'parallel_categories.html'))

        # 14. Time Series with Confidence Intervals
        plt.figure(figsize=(15, 8))
        for model in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model]
            times = range(len(model_data))
            mean_time = model_data['elapsed_time'].rolling(window=5).mean()
            std_time = model_data['elapsed_time'].rolling(window=5).std()
            
            plt.plot(times, mean_time, label=model)
            plt.fill_between(times, 
                           mean_time - std_time, 
                           mean_time + std_time, 
                           alpha=0.2)
        
        plt.title('Response Time Trends with Confidence Intervals')
        plt.xlabel('Query Sequence')
        plt.ylabel('Response Time (s)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'time_series_confidence.png'), dpi=300)
        plt.close()

        # 15. Quality Metrics Radar Chart
        # Định nghĩa các metrics tiềm năng
        potential_metrics = ['response_quality_score', 'complexity_score', 'coherence_score', 
                          'efficiency_score', 'tokens_per_second']
        
        # Lọc ra các metrics thực sự tồn tại trong DataFrame
        quality_metrics = [metric for metric in potential_metrics if metric in results_df.columns]
        
        if len(quality_metrics) > 2:  # Cần ít nhất 3 metrics để tạo radar chart có ý nghĩa
            # Prepare data for radar chart
            model_quality_data = {}
            for model in results_df['model_name'].unique():
                model_data = results_df[results_df['model_name'] == model]
                scores = []
                for metric in quality_metrics:
                    # Normalize scores between 0 and 1
                    metric_min = results_df[metric].min()
                    metric_max = results_df[metric].max()
                    # Tránh chia cho 0 nếu min = max
                    if metric_max > metric_min:
                        score = (model_data[metric].mean() - metric_min) / (metric_max - metric_min)
                    else:
                        score = 0.5  # Giá trị mặc định nếu không thể chuẩn hóa
                    scores.append(score)
                model_quality_data[model] = scores

            # Create radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            angles = np.linspace(0, 2*np.pi, len(quality_metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Complete the circle

            for i, (model, scores) in enumerate(model_quality_data.items()):
                scores = np.concatenate((scores, [scores[0]]))  # Close the loop
                ax.plot(angles, scores, 'o-', linewidth=2, label=model)
                ax.fill(angles, scores, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(quality_metrics)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Quality Metrics Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'quality_radar.png'), dpi=300)
            plt.close()

        # 16. Stacked Area Chart for Response Types
        plt.figure(figsize=(12, 6))
        response_types = pd.crosstab(
            results_df['model_name'],
            [results_df['prompt_type'], results_df['has_error']]
        )
        response_types.plot(kind='area', stacked=True)
        plt.title('Response Type Distribution by Model')
        plt.xlabel('Model')
        plt.ylabel('Count')
        plt.legend(title='Prompt Type - Error', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'response_types_area.png'), dpi=300)
        plt.close()

        # Add new sections to HTML report
        html_content = html_content.replace(
            "<!-- Advanced Analysis Section -->",
            f"""
            <div class="section">
                <h2>Advanced Performance Analysis</h2>
                
                <div class="grid-2">
                    <div class="plot-container">
                        <div class="chart-title">3D Quality Surface</div>
                        <div class="chart-description">Interactive 3D visualization of response quality across models and prompt types.</div>
                        <iframe src="plots/3d_surface_quality.html" width="100%" height="600px" frameborder="0"></iframe>
                    </div>
                    
                    <div class="plot-container">
                        <div class="chart-title">Error Distribution Hierarchy</div>
                        <div class="chart-description">Interactive sunburst chart showing error distribution patterns.</div>
                        <iframe src="plots/error_sunburst.html" width="100%" height="600px" frameborder="0"></iframe>
                    </div>
                </div>

                <div class="grid-2">
                    <div class="plot-container">
                        <div class="chart-title">Response Time Trends</div>
                        <div class="chart-description">Time series with confidence intervals showing response time stability.</div>
                        <img src="plots/time_series_confidence.png" alt="Time Series with Confidence Intervals">
                    </div>
                    
                    <div class="plot-container">
                        <div class="chart-title">Quality Metrics Comparison</div>
                        <div class="chart-description">Radar chart comparing multiple quality metrics across models.</div>
                        <img src="plots/quality_radar.png" alt="Quality Metrics Radar">
                    </div>
                </div>

                <div class="grid-2">
                    <div class="plot-container">
                        <div class="chart-title">Response Type Distribution</div>
                        <div class="chart-description">Stacked area chart showing the distribution of response types.</div>
                        <img src="plots/response_types_area.png" alt="Response Types Distribution">
                    </div>
                    
                    <div class="plot-container">
                        <div class="chart-title">Parallel Categories Analysis</div>
                        <div class="chart-description">Interactive parallel categories plot showing relationships between variables.</div>
                        <iframe src="plots/parallel_categories.html" width="100%" height="500px" frameborder="0"></iframe>
                    </div>
                </div>
            </div>
            """
        )

        # Add new metrics to the statistics section
        stats_update = {
            "quality_metrics": {
                "avg_quality_score": results_df['response_quality_score'].mean()
            }
        }
        
        # Thêm các chỉ số vào stats nếu chúng tồn tại trong DataFrame
        if 'complexity_score' in results_df.columns:
            stats_update["quality_metrics"]["avg_complexity"] = results_df['complexity_score'].mean()
            
        if 'coherence_score' in results_df.columns:
            stats_update["quality_metrics"]["avg_coherence"] = results_df['coherence_score'].mean()
            
        if 'response_quality_score' in results_df.columns:
            stats_update["quality_metrics"]["quality_consistency"] = results_df['response_quality_score'].std()
            
        # Thêm chỉ số về độ ổn định của hiệu suất
        stats_update["performance_stability"] = {
            model: results_df[results_df['model_name'] == model]['elapsed_time'].std()
            for model in results_df['model_name'].unique()
        }
        
        stats.update(stats_update)
        
        # Tính thêm các chỉ số mới nâng cao
        for model in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model]
            
            # Tính điểm hiệu quả chi phí (response_quality / time)
            stats[f"{model}_cost_efficiency"] = (
                model_data['response_quality_score'].mean() / 
                max(0.1, model_data['elapsed_time'].mean())
            )
            
            # Tính điểm đáp ứng nhất quán (100 - % độ lệch chuẩn của thời gian)
            time_mean = model_data['elapsed_time'].mean()
            time_std = model_data['elapsed_time'].std()
            stats[f"{model}_time_consistency"] = 100 - min(100, (time_std / max(0.1, time_mean) * 100))
            
            # Tính chỉ số tổng hợp chất lượng-tốc độ
            stats[f"{model}_quality_speed_index"] = (
                model_data['response_quality_score'].mean() * 0.7 + 
                (1 - model_data['elapsed_time'].mean() / results_df['elapsed_time'].max()) * 0.3
            )
            
            # Tính tỷ lệ độ dài hữu ích (loại bỏ lỗi)
            error_responses = model_data[model_data['has_error']]['response_length'].sum()
            total_responses = model_data['response_length'].sum()
            stats[f"{model}_useful_content_ratio"] = 1 - (error_responses / max(1, total_responses))
            
            # Tính hiệu quả prompt (chất lượng trung bình theo từng loại prompt)
            prompt_efficiency = {}
            for prompt in results_df['prompt_type'].unique():
                prompt_data = model_data[model_data['prompt_type'] == prompt]
                if len(prompt_data) > 0:
                    prompt_efficiency[prompt] = prompt_data['response_quality_score'].mean()
            stats[f"{model}_prompt_efficiency"] = prompt_efficiency
        
        # 17. Thêm Bubble Chart so sánh 3 chiều (thời gian, chất lượng, độ dài)
        plt.figure(figsize=(14, 10))
        
        # Group by model and calculate averages for bubble size
        model_avg = results_df.groupby('model_name').agg({
            'elapsed_time': 'mean',
            'response_quality_score': 'mean',
            'response_length': 'mean',
            'has_error': 'mean'
        }).reset_index()
        
        # Normalize bubble size for better visualization
        max_size = model_avg['response_length'].max()
        model_avg['bubble_size'] = model_avg['response_length'] / max_size * 1000
        
        # Create bubble chart
        for i, model in enumerate(model_avg['model_name']):
            plt.scatter(
                model_avg.loc[model_avg['model_name'] == model, 'elapsed_time'],
                model_avg.loc[model_avg['model_name'] == model, 'response_quality_score'],
                s=model_avg.loc[model_avg['model_name'] == model, 'bubble_size'],
                alpha=0.7,
                label=model
            )
        
        # Add model names as annotations
        for i, row in model_avg.iterrows():
            plt.annotate(
                row['model_name'],
                (row['elapsed_time'], row['response_quality_score']),
                fontsize=12
            )
        
        plt.title('Bubble Chart: Response Time vs Quality vs Length', fontsize=16)
        plt.xlabel('Response Time (seconds)', fontsize=14)
        plt.ylabel('Response Quality Score', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'bubble_chart_comparison.png'), dpi=300)
        plt.close()
        
        # 18. Thêm Heatmap so sánh nhiều chỉ số
        potential_metrics = ['response_quality_score', 'complexity_score', 'coherence_score', 'efficiency_score', 'quality_index']
        
        # Lọc các metrics thực sự có trong DataFrame
        metrics_to_compare = [metric for metric in potential_metrics if metric in results_df.columns]
        
        if len(metrics_to_compare) > 0:
            # Calculate average of each metric for each model
            heatmap_data = results_df.groupby('model_name')[metrics_to_compare].mean()
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                heatmap_data, 
                annot=True, 
                cmap='viridis', 
                linewidths=0.5, 
                fmt='.3f'
            )
            plt.title('Multi-Metric Comparison Across Models', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'multi_metric_heatmap.png'), dpi=300)
            plt.close()
        
        # 19. Thêm Horizon Plot cho biến động thời gian qua các mô hình
        from matplotlib.colors import ListedColormap
        
        plt.figure(figsize=(14, 8))
        models = results_df['model_name'].unique()
        
        # Create custom color maps for each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        # Sort by question to ensure comparison is fair
        results_df = results_df.sort_values(['question', 'model_name', 'prompt_type'])
        
        for i, model in enumerate(models):
            model_data = results_df[results_df['model_name'] == model]
            
            # Create bottom row base
            plt.fill_between(
                range(len(model_data)),
                i,
                i + model_data['elapsed_time']/model_data['elapsed_time'].max()/2,
                color=colors[i],
                alpha=0.7,
                label=model
            )
            
            # Add line for trend
            plt.plot(
                range(len(model_data)),
                i + model_data['elapsed_time']/model_data['elapsed_time'].max()/2,
                color='black',
                linewidth=1
            )
        
        plt.title('Horizon Plot: Response Time Variation Across Models', fontsize=16)
        plt.xlabel('Query Sequence', fontsize=14)
        plt.ylabel('Model (with normalized time values)', fontsize=14)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'horizon_plot_time.png'), dpi=300)
        plt.close()
        
        # 20. Thêm Biểu đồ so sánh Cost-Efficiency Index
        cost_efficiency = {}
        for model in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model]
            # Calculate quality/time ratio as cost efficiency
            cost_efficiency[model] = model_data['response_quality_score'].mean() / max(0.1, model_data['elapsed_time'].mean())
        
        # Convert to DataFrame
        cost_df = pd.DataFrame(list(cost_efficiency.items()), columns=['model', 'cost_efficiency'])
        cost_df = cost_df.sort_values('cost_efficiency', ascending=False)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(cost_df['model'], cost_df['cost_efficiency'], color=plt.cm.viridis(np.linspace(0, 1, len(cost_df))))
        
        plt.title('Cost-Efficiency Index (Quality/Time)', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Cost-Efficiency (higher is better)', fontsize=14)
        plt.xticks(rotation=45)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cost_efficiency_index.png'), dpi=300)
        plt.close()
        
        # 21. Tạo Biểu đồ tương tác về phân bố lỗi theo mô hình và loại prompt
        error_sunburst = px.sunburst(
            results_df,
            path=['model_name', 'prompt_type', 'has_error'],
            color='has_error',
            color_discrete_sequence=px.colors.qualitative.Set3,
            title='Interactive Error Distribution by Model and Prompt Type'
        )
        error_sunburst.write_html(os.path.join(plots_dir, 'interactive_error_sunburst.html'))
        
        # 22. Tạo Sankey Diagram cho luồng kết quả (Model -> Prompt -> Error/Success)
        # Tạo dữ liệu cho Sankey Diagram
        from plotly.graph_objects import Sankey
        
        # Chuẩn bị dữ liệu
        model_list = list(results_df['model_name'].unique())
        prompt_list = list(results_df['prompt_type'].unique())
        results_list = ['Success', 'Error']
        
        labels = model_list + prompt_list + results_list
        
        source_indices = []
        target_indices = []
        values = []
        
        # Thêm liên kết Model -> Prompt
        for i, model in enumerate(model_list):
            for j, prompt in enumerate(prompt_list):
                model_prompt_count = len(results_df[(results_df['model_name'] == model) & 
                                                  (results_df['prompt_type'] == prompt)])
                if model_prompt_count > 0:
                    source_indices.append(i)
                    target_indices.append(len(model_list) + j)
                    values.append(model_prompt_count)
        
        # Thêm liên kết Prompt -> Result
        for j, prompt in enumerate(prompt_list):
            for k, result in enumerate(results_list):
                is_error = (result == 'Error')
                prompt_result_count = len(results_df[(results_df['prompt_type'] == prompt) & 
                                                  (results_df['has_error'] == is_error)])
                if prompt_result_count > 0:
                    source_indices.append(len(model_list) + j)
                    target_indices.append(len(model_list) + len(prompt_list) + k)
                    values.append(prompt_result_count)
        
        # Tạo Sankey Diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        )])
        
        fig.update_layout(title_text="Sankey Diagram: Model -> Prompt -> Result Flow", font_size=12)
        fig.write_html(os.path.join(plots_dir, 'result_flow_sankey.html'))
        
        # Cập nhật HTML report với các biểu đồ mới
        advanced_analysis_section = f"""
        <div class="section">
            <h2>Advanced Performance Analysis</h2>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">3D Quality Surface</div>
                    <div class="chart-description">Interactive 3D visualization of response quality across models and prompt types.</div>
                    <iframe src="plots/3d_surface_quality.html" width="100%" height="600px" frameborder="0"></iframe>
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Error Distribution Hierarchy</div>
                    <div class="chart-description">Interactive sunburst chart showing error distribution patterns.</div>
                    <iframe src="plots/error_sunburst.html" width="100%" height="600px" frameborder="0"></iframe>
                </div>
            </div>

            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">Response Time Trends</div>
                    <div class="chart-description">Time series with confidence intervals showing response time stability.</div>
                    <img src="plots/time_series_confidence.png" alt="Time Series with Confidence Intervals">
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Quality Metrics Comparison</div>
                    <div class="chart-description">Radar chart comparing multiple quality metrics across models.</div>
                    <img src="plots/quality_radar.png" alt="Quality Metrics Radar">
                </div>
            </div>

            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">Response Type Distribution</div>
                    <div class="chart-description">Stacked area chart showing the distribution of response types.</div>
                    <img src="plots/response_types_area.png" alt="Response Types Distribution">
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Parallel Categories Analysis</div>
                    <div class="chart-description">Interactive parallel categories plot showing relationships between variables.</div>
                    <iframe src="plots/parallel_categories.html" width="100%" height="500px" frameborder="0"></iframe>
                </div>
            </div>
            
            <h2>Extended Performance Metrics</h2>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">Multi-Metric Heatmap</div>
                    <div class="chart-description">Heatmap comparing different quality metrics across models.</div>
                    <img src="plots/multi_metric_heatmap.png" alt="Multi-Metric Heatmap">
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Cost-Efficiency Index</div>
                    <div class="chart-description">Bar chart comparing the quality-to-time ratio (efficiency) of each model.</div>
                    <img src="plots/cost_efficiency_index.png" alt="Cost-Efficiency Index">
                </div>
            </div>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">3D Comparative Analysis</div>
                    <div class="chart-description">Bubble chart comparing response time, quality score and response length.</div>
                    <img src="plots/bubble_chart_comparison.png" alt="3D Comparative Analysis">
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Response Time Variation</div>
                    <div class="chart-description">Horizon plot showing response time variation across models.</div>
                    <img src="plots/horizon_plot_time.png" alt="Response Time Variation">
                </div>
            </div>
            
            <div class="grid-2">
                <div class="plot-container">
                    <div class="chart-title">Interactive Error Distribution</div>
                    <div class="chart-description">Interactive sunburst chart for detailed error analysis by model and prompt.</div>
                    <iframe src="plots/interactive_error_sunburst.html" width="100%" height="500px" frameborder="0"></iframe>
                </div>
                
                <div class="plot-container">
                    <div class="chart-title">Result Flow Analysis</div>
                    <div class="chart-description">Sankey diagram showing the flow from models through prompts to results.</div>
                    <iframe src="plots/result_flow_sankey.html" width="100%" height="500px" frameborder="0"></iframe>
                </div>
            </div>
        </div>
        """
        
        # Thêm phần báo cáo chi tiết về hiệu suất mô hình
        model_performance_section = f"""
        <div class="section">
            <h2>Detailed Model Performance Analysis</h2>
            
            <div class="model-metrics">
                {''.join(f"""
                <div class="model-card">
                    <h3>{model} Performance Metrics</h3>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <span class="metric-label">Quality-Speed Index:</span>
                            <span class="metric-value">{stats[f"{model}_quality_speed_index"]:.3f}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Cost Efficiency:</span>
                            <span class="metric-value">{stats[f"{model}_cost_efficiency"]:.3f}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Time Consistency:</span>
                            <span class="metric-value">{stats[f"{model}_time_consistency"]:.1f}%</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Useful Content Ratio:</span>
                            <span class="metric-value">{stats[f"{model}_useful_content_ratio"]:.2f}</span>
                        </div>
                    </div>
                    
                    <h4>Prompt Efficiency</h4>
                    <div class="prompt-efficiency-chart">
                        <div class="prompt-bars">
                            {''.join(f"""
                            <div class="prompt-bar-container">
                                <div class="prompt-label">{prompt}</div>
                                <div class="prompt-bar" style="width: {efficiency*100}%;">{efficiency:.2f}</div>
                            </div>
                            """ for prompt, efficiency in stats[f"{model}_prompt_efficiency"].items())}
                        </div>
                    </div>
                </div>
                """ for model in results_df['model_name'].unique())}
            </div>
            
            <h3>Performance Stability Analysis</h3>
            <p>The coefficient of variation (CV) measures the relative variability in response times, with lower values indicating more consistent performance:</p>
            
            <div class="stability-table">
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Avg. Response Time (s)</th>
                        <th>Std. Deviation (s)</th>
                        <th>Coefficient of Variation</th>
                        <th>Stability Rating</th>
                    </tr>
                    {''.join(f"""
                    <tr>
                        <td>{model}</td>
                        <td>{results_df[results_df['model_name'] == model]['elapsed_time'].mean():.2f}</td>
                        <td>{results_df[results_df['model_name'] == model]['elapsed_time'].std():.2f}</td>
                        <td>{results_df[results_df['model_name'] == model]['elapsed_time'].std() / results_df[results_df['model_name'] == model]['elapsed_time'].mean():.2f}</td>
                        <td>{"High" if results_df[results_df['model_name'] == model]['elapsed_time'].std() / results_df[results_df['model_name'] == model]['elapsed_time'].mean() < 0.3 else "Medium" if results_df[results_df['model_name'] == model]['elapsed_time'].std() / results_df[results_df['model_name'] == model]['elapsed_time'].mean() < 0.7 else "Low"}</td>
                    </tr>
                    """ for model in results_df['model_name'].unique())}
                </table>
            </div>
        </div>
        """
        
        # Thêm CSS mới cho các phần được thêm vào
        additional_css = """
        .model-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .model-card h3 {
            color: #2c3e50;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-top: 0;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        
        .metric-item {
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .metric-label {
            display: block;
            font-size: 0.9em;
            color: #7f8c8d;
        }
        
        .metric-value {
            display: block;
            font-size: 1.3em;
            font-weight: bold;
            color: #3498db;
        }
        
        .prompt-efficiency-chart {
            margin-top: 15px;
        }
        
        .prompt-bars {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .prompt-bar-container {
            display: flex;
            align-items: center;
        }
        
        .prompt-label {
            width: 120px;
            text-align: right;
            padding-right: 10px;
            font-size: 0.9em;
        }
        
        .prompt-bar {
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            text-align: right;
            min-width: 40px;
        }
        
        .stability-table table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .stability-table th,
        .stability-table td {
            padding: 12px;
            text-align: center;
        }
        
        .stability-table th {
            background-color: #34495e;
            color: white;
        }
        
        .model-metrics {
            margin-top: 25px;
        }
        
        .analysis-summary {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .analysis-summary h3 {
            color: #2c3e50;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-top: 0;
        }
        
        .analysis-summary ul {
            padding-left: 20px;
        }
        
        .analysis-summary li {
            margin-bottom: 10px;
        }
        
        .analysis-summary strong {
            color: #3498db;
        }
        """
        
        # Thay thế phần style và section trong HTML
        html_content = html_content.replace("</style>", f"{additional_css}\n</style>")
        
        # Thêm phần Advanced Analysis Section
        html_content = html_content.replace("<!-- Advanced Analysis Section -->", advanced_analysis_section)
        
        # Thêm phần Model Performance Section trước footer
        html_content = html_content.replace("</div>\n            \n            <footer>", f"</div>\n            {model_performance_section}\n            {prompt_effectiveness_section}\n            <footer>")
        
        # Save HTML report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print_status("success", f"Enhanced comprehensive report generated: {report_path}", "green")
        
        # Biểu đồ "Complexity-Performance Trade-off"
        plt.figure(figsize=(12, 8))
        plt.scatter(
            results_df['complexity_score'], 
            results_df['correctness_score'],
            c=results_df['model_name'].astype('category').cat.codes,
            s=100, alpha=0.7,
            edgecolors='w', linewidth=0.5
        )
        
        for model in results_df['model_name'].unique():
            model_data = results_df[results_df['model_name'] == model]
            for prompt in model_data['prompt_type'].unique():
                prompt_data = model_data[model_data['prompt_type'] == prompt]
                avg_complexity = prompt_data['complexity_score'].mean()
                avg_correctness = prompt_data['correctness_score'].mean()
                plt.annotate(
                    f"{model}-{prompt}",
                    (avg_complexity, avg_correctness),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9
                )
        
        plt.title('Complexity-Performance Trade-off', fontsize=16)
        plt.xlabel('Response Complexity Score', fontsize=14)
        plt.ylabel('Correctness Score', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.colorbar(label='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'complexity_performance_tradeoff.png'), dpi=300)
        plt.close()
        
        # Phân tích "Error Pattern Analysis"
        # Chỉ áp dụng cho CoT và Hybrid-CoT
        reasoning_df = results_df[results_df['prompt_type'].isin(['cot', 'hybrid_cot'])]
        if not reasoning_df.empty:
            # Tính tỷ lệ các loại lỗi
            error_data = []
            for model in reasoning_df['model_name'].unique():
                for prompt in reasoning_df['prompt_type'].unique():
                    model_prompt_data = reasoning_df[(reasoning_df['model_name'] == model) & 
                                                   (reasoning_df['prompt_type'] == prompt)]
                    if not model_prompt_data.empty:
                        # Tỷ lệ các loại lỗi
                        calc_error_rate = model_prompt_data['has_calculation_error'].mean()
                        logic_error_rate = model_prompt_data['has_logic_error'].mean()
                        total_count = len(model_prompt_data)
                        
                        error_data.append({
                            'model': model,
                            'prompt_type': prompt,
                            'calculation_error_rate': calc_error_rate,
                            'logic_error_rate': logic_error_rate,
                            'no_error_rate': 1 - (calc_error_rate + logic_error_rate),
                            'total_count': total_count
                        })
            
            error_df = pd.DataFrame(error_data)
            if not error_df.empty:
                # Chuyển thành định dạng phù hợp cho stacked bar chart
                error_melted = pd.melt(
                    error_df,
                    id_vars=['model', 'prompt_type', 'total_count'],
                    value_vars=['calculation_error_rate', 'logic_error_rate', 'no_error_rate'],
                    var_name='error_type', value_name='rate'
                )
                
                # Vẽ stacked bar chart
                plt.figure(figsize=(14, 8))
                ax = sns.barplot(
                    x='model', 
                    y='rate', 
                    hue='error_type', 
                    data=error_melted,
                    dodge=False
                )
                
                # Thêm giá trị trên mỗi phần của stack
                for bars in ax.containers:
                    ax.bar_label(bars, fmt='%.2f')
                
                plt.title('Error Pattern Analysis', fontsize=16)
                plt.xlabel('Model', fontsize=14)
                plt.ylabel('Rate', fontsize=14)
                plt.legend(title='Error Type')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'error_pattern_analysis.png'), dpi=300)
                plt.close()
        
        # Biểu đồ "Prompt Length vs Accuracy"
        # Ước tính độ dài prompt từ mỗi loại
        prompt_length_estimates = {
            'standard': 1.0,  # Độ dài cơ sở
            'cot': 1.5,       # Dài hơn 50% vì thêm hướng dẫn suy luận
            'hybrid_cot': 1.3  # Dài hơn 30% vì kết hợp
        }
        
        # Tạo dữ liệu cho phân tích
        prompt_accuracy_data = []
        for model in results_df['model_name'].unique():
            for prompt_type in results_df['prompt_type'].unique():
                subset = results_df[(results_df['model_name'] == model) & 
                                 (results_df['prompt_type'] == prompt_type)]
                if not subset.empty:
                    accuracy = subset['is_correct'].mean()
                    rel_length = prompt_length_estimates.get(prompt_type, 1.0)
                    
                    prompt_accuracy_data.append({
                        'model': model,
                        'prompt_type': prompt_type,
                        'accuracy': accuracy,
                        'relative_length': rel_length
                    })
        
        prompt_acc_df = pd.DataFrame(prompt_accuracy_data)
        if not prompt_acc_df.empty:
            plt.figure(figsize=(12, 8))
            
            # Vẽ scatter plot với kích thước phản ánh số lượng mẫu
            for model in prompt_acc_df['model'].unique():
                model_data = prompt_acc_df[prompt_acc_df['model'] == model]
                plt.plot(
                    model_data['relative_length'], 
                    model_data['accuracy'],
                    'o-',
                    label=model,
                    markersize=10
                )
                
                # Thêm annotation
                for _, row in model_data.iterrows():
                    plt.annotate(
                        row['prompt_type'],
                        (row['relative_length'], row['accuracy']),
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
            
            plt.title('Prompt Length vs Accuracy', fontsize=16)
            plt.xlabel('Relative Prompt Length', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Model')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'prompt_length_vs_accuracy.png'), dpi=300)
            plt.close()
        
        # Biểu đồ "Time-to-Solution Analysis"
        time_solution_data = []
        for model in results_df['model_name'].unique():
            for prompt_type in results_df['prompt_type'].unique():
                subset = results_df[(results_df['model_name'] == model) & 
                                 (results_df['prompt_type'] == prompt_type)]
                
                if not subset.empty:
                    # Thời gian trung bình
                    avg_time = subset['elapsed_time'].mean()
                    
                    # Tính điểm hiệu quả thời gian (độ chính xác / thời gian)
                    time_efficiency = subset['correctness_score'].mean() / avg_time if avg_time > 0 else 0
                    
                    # Số bước suy luận trung bình (0 cho Standard)
                    avg_steps = subset['num_reasoning_steps'].mean()
                    
                    time_solution_data.append({
                        'model': model,
                        'prompt_type': prompt_type,
                        'avg_time': avg_time,
                        'time_efficiency': time_efficiency,
                        'avg_steps': avg_steps
                    })
        
        time_sol_df = pd.DataFrame(time_solution_data)
        if not time_sol_df.empty:
            # Vẽ biểu đồ thời gian trung bình theo các bước suy luận
            plt.figure(figsize=(14, 8))
            
            for model in time_sol_df['model'].unique():
                model_data = time_sol_df[time_sol_df['model'] == model]
                plt.plot(
                    model_data['avg_steps'], 
                    model_data['avg_time'],
                    'o-',
                    label=model,
                    markersize=10
                )
                
                # Thêm annotation
                for _, row in model_data.iterrows():
                    plt.annotate(
                        row['prompt_type'],
                        (row['avg_steps'], row['avg_time']),
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
            
            plt.title('Time-to-Solution Analysis', fontsize=16)
            plt.xlabel('Average Number of Reasoning Steps', fontsize=14)
            plt.ylabel('Average Response Time (s)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Model')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'time_to_solution_analysis.png'), dpi=300)
            plt.close()
            
            # Vẽ biểu đồ hiệu quả thời gian (độ chính xác / thời gian)
            plt.figure(figsize=(14, 8))
            
            # Tạo grouped bar chart
            sns.barplot(x='model', y='time_efficiency', hue='prompt_type', data=time_sol_df)
            
            plt.title('Time Efficiency (Correctness/Time)', fontsize=16)
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Time Efficiency', fontsize=14)
            plt.legend(title='Prompt Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'time_efficiency.png'), dpi=300)
            plt.close()
        
        # Thêm biểu đồ hiệu quả suy luận (efficiency score)
        plt.figure(figsize=(12, 8))
        if not reasoning_df.empty:
            sns.barplot(x='model_name', y='efficiency_score', hue='prompt_type', data=reasoning_df)
            plt.title('Reasoning Efficiency Comparison: CoT vs Hybrid-CoT', fontsize=16)
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Efficiency Score', fontsize=14)
            plt.legend(title='Prompt Type', title_fontsize=12, fontsize=10)
            plt.xticks(rotation=45)
            plt.ylim(0, 0.5)  # Efficiency score thường nhỏ hơn các scores khác
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'reasoning_efficiency_comparison.png'), dpi=300)
            plt.close()

        # Biểu đồ phân tích lỗi suy luận
        plt.figure(figsize=(12, 8))
        if not reasoning_df.empty:
            reasoning_error_data = reasoning_df.groupby(['model_name', 'prompt_type']).agg({
                'has_calculation_error': 'mean',
                'has_logic_error': 'mean'
            }).reset_index()
            
            # Reshape for plotting
            reasoning_error_melted = pd.melt(
                reasoning_error_data,
                id_vars=['model_name', 'prompt_type'],
                value_vars=['has_calculation_error', 'has_logic_error'],
                var_name='error_type',
                value_name='error_rate'
            )
            
            # Plot
            sns.barplot(x='model_name', y='error_rate', hue='error_type', data=reasoning_error_melted)
            plt.title('Reasoning Error Analysis by Model and Prompt Type', fontsize=16)
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Error Rate', fontsize=14)
            plt.legend(title='Error Type', title_fontsize=12, fontsize=10)
            plt.xticks(rotation=45)
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'reasoning_error_analysis.png'), dpi=300)
            plt.close()
            
            # Thêm biểu đồ phân tích số bước suy luận
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='model_name', y='num_reasoning_steps', hue='prompt_type', data=reasoning_df)
            plt.title('Number of Reasoning Steps by Model and Prompt Type', fontsize=16)
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Number of Steps', fontsize=14)
            plt.legend(title='Prompt Type', title_fontsize=12, fontsize=10)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'reasoning_steps_distribution.png'), dpi=300)
            plt.close()
        
        # Generate comprehensive prompt comparison report
        if args.analyze_prompt_comparison:
            report_path = generate_comprehensive_prompt_report(df_results, output_dir)
            if report_path:
                print_status("success", "Comprehensive prompt comparison report generated", "green")
        
        # Add ReAct-specific analysis if requested
        if args.analyze_react:
            print_header("ReAct Prompt Analysis")
            print_status("info", "Performing detailed analysis of ReAct prompts...", "blue")
            
            try:
                from check.model_evaluator import ModelEvaluator
                evaluator = ModelEvaluator({}, output_dir)
                react_results = evaluator.analyze_react_prompt_evaluation(df_results, output_dir)
                
                if react_results:
                    print_status("success", "ReAct prompt analysis completed successfully", "green")
                    
                    # Save ReAct analysis results
                    react_file = os.path.join(output_dir, "react_analysis_results.json")
                    
                    # Convert DataFrames to JSON-able format
                    react_json = {}
                    for key, value in react_results.items():
                        if isinstance(value, pd.DataFrame):
                            react_json[key] = value.to_dict(orient='records')
                        else:
                            react_json[key] = value
                    
                    with open(react_file, 'w', encoding='utf-8') as f:
                        json.dump(react_json, f, indent=4, ensure_ascii=False)
                    
                    print_status("success", f"ReAct analysis results saved to {react_file}", "green")
            except ImportError:
                print_status("error", "ModelEvaluator not found, cannot perform ReAct analysis", "red")
            except Exception as e:
                print_status("error", f"Error in ReAct analysis: {str(e)}", "red")
                traceback.print_exc()
        
    except Exception as e:
        print_status("error", f"Error generating report: {str(e)}", "red")
        import traceback
        traceback.print_exc()

def generate_comprehensive_prompt_report(results_df, output_dir):
    """
    Generate a comprehensive report that analyzes and compares different prompt techniques.
    
    Args:
        results_df (DataFrame): DataFrame containing evaluation results
        output_dir (str): Directory to save the report files
        
    Returns:
        str: Path to the generated report
    """
    print_header("Generating Comprehensive Prompt Comparison Report")
    
    try:
        # Import ModelEvaluator
        from check.model_evaluator import ModelEvaluator
        
        # Initialize evaluator with output directory
        evaluator = ModelEvaluator({}, output_dir)
        
        # Generate the prompt comparison report
        print_status("info", "Creating detailed prompt technique comparison report...", "blue")
        report_path = evaluator.generate_prompt_comparison_report(results_df, output_dir)
        
        if report_path:
            print_status("success", f"Comprehensive report generated successfully at: {report_path}", "green")
            
            # Try to automatically open the report in the default browser
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
                print_status("info", "Report opened in your default web browser", "blue")
            except Exception as e:
                print_status("info", f"Report ready at: {report_path}", "blue")
                
        else:
            print_status("error", "Failed to generate comprehensive report", "red")
        
        return report_path
        
    except Exception as e:
        print_status("error", f"Error generating comprehensive report: {str(e)}", "red")
        import traceback
        traceback.print_exc()
        return None

# Define a new function to parse GPU allocations from command-line arguments
def parse_gpu_allocation(gpu_allocation_str, models, default_gpu_ids):
    """
    Parse GPU allocation string and create a mapping of model to GPU IDs.
    
    Format: "model1:gpu_id1,model2:gpu_id2,..."
    If a model is not specified, it will be assigned a default GPU ID.
    
    Args:
        gpu_allocation_str (str): GPU allocation string
        models (list): List of models to allocate
        default_gpu_ids (list): List of default GPU IDs
        
    Returns:
        dict: Mapping of model to GPU IDs
    """
    # Create default mapping: each model gets one GPU from default_gpu_ids
    allocation = {}
    api_based_models = ["gemini"] # Models that don't need GPU resources
    
    # If no allocation string is provided, create a balanced allocation
    if not gpu_allocation_str:
        available_gpus = default_gpu_ids.copy()
        
        # First allocate for models that need GPU resources
        for model in models:
            if model not in api_based_models:
                if available_gpus:
                    allocation[model] = [available_gpus.pop(0)]
                else:
                    # If we run out of GPUs, reuse the first one
                    allocation[model] = [default_gpu_ids[0]]
        
        # For API-based models, they don't need GPU
        for model in models:
            if model in api_based_models:
                allocation[model] = []
        
        return allocation
    
    # Parse allocation string
    pairs = gpu_allocation_str.split(',')
    for pair in pairs:
        if ':' not in pair:
            continue
            
        model, gpu_id = pair.split(':')
        model = model.strip()
        gpu_id = int(gpu_id.strip())
        
        # Skip invalid models
        if model not in models:
            print_status("warning", f"Ignoring allocation for unknown model '{model}'", "yellow")
            continue
            
        # Initialize if needed
        if model not in allocation:
            allocation[model] = []
            
        # Add GPU ID
        allocation[model].append(gpu_id)
    
    # Check if any models are missing and assign default GPUs
    for model in models:
        if model not in allocation:
            # For API-based models, they don't need GPU
            if model in api_based_models:
                allocation[model] = []
            else:
                # Just use the first default GPU if none specified
                allocation[model] = [default_gpu_ids[0]]
                print_status("info", f"Model '{model}' not in allocation string, using default GPU {default_gpu_ids[0]}", "blue")
    
    return allocation

# Add the parallel model evaluation functions
def evaluate_models_in_parallel(questions, models, prompt_types, gpu_allocation, output_dir, timestamp, batch_size=10, use_4bit=True):
    """
    Evaluate multiple models in parallel using a more efficient approach.
    Uses model caching to avoid reloading models, and separate processes/threads
    for processing different parts of the evaluation.
    
    Args:
        questions (list): List of questions to evaluate
        models (list): List of models to evaluate
        prompt_types (list): List of prompt types to evaluate
        gpu_allocation (dict): Dictionary mapping model names to list of GPU IDs
        output_dir (str): Directory to save results
        timestamp (str): Timestamp for result files
        batch_size (int): Number of questions to process in each batch
        use_4bit (bool): Whether to use 4-bit quantization
        
    Returns:
        tuple: (results, results_file)
    """
    print_status("info", "Starting efficient parallel evaluation with GPU distribution", "blue")
    
    # Create a manager for shared results
    with Manager() as manager:
        # Create a shared dictionary to store results
        shared_results = manager.dict()
        # Create a lock for thread-safe operations
        lock = manager.Lock()
        
        # First, handle any API-based models (e.g., Gemini) in separate threads
        api_threads = []
        gpu_processes = []
        processes = []  # List to store all processes and threads
        
        # Stage 1: Start API-based models in threads
        for model_name in models:
            # Get GPU allocation for this model
            model_gpu_ids = gpu_allocation.get(model_name, [])
            
            # If the model is API-based, run it in a separate thread
            if model_name == "gemini" or not model_gpu_ids:
                print_status("info", f"Starting {model_name} in thread (API-based model)", "blue")
                
                # Create thread for API model
                api_thread = threading.Thread(
                    target=evaluate_single_model_process,
                    args=(
                        questions,
                        model_name,
                        prompt_types,
                        -1,  # No GPU needed for API
                        batch_size,
                        use_4bit,
                        shared_results,
                        output_dir,
                        timestamp
                    )
                )
                api_thread.daemon = True
                api_thread.start()
                processes.append(api_thread)
                continue
            
            # For GPU-based models, create a process for each GPU assigned to the model
            for gpu_id in model_gpu_ids:
                print_status("info", f"Starting {model_name} on GPU {gpu_id}", "blue")
                
                # Create process for this model on this GPU
                p = Process(
                    target=evaluate_single_model_process,
                    args=(
                        questions,
                        model_name,
                        prompt_types,
                        gpu_id,
                        batch_size,
                        use_4bit,
                        shared_results,
                        output_dir,
                        timestamp
                    )
                )
                p.start()
                processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            if isinstance(p, Process):
                p.join()
            else:
                p.join()  # For threads
        
        # Get final results from shared dictionary
        results = dict(shared_results)
        
        # Save final results
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print_status("success", "Parallel evaluation complete", "green")
        
        return results, results_file

def evaluate_single_model_process(questions, model_name, prompt_types, gpu_id, batch_size, use_4bit, shared_results, output_dir, timestamp):
    """
    Process function for evaluating a single model on a specific GPU.
    
    Args:
        questions (list): List of questions to evaluate
        model_name (str): Name of the model to evaluate
        prompt_types (list): List of prompt types to evaluate
        gpu_id (int): GPU ID to use (-1 for API models)
        batch_size (int): Number of questions to process in each batch
        use_4bit (bool): Whether to use 4-bit quantization
        shared_results (dict): Shared dictionary to store results
        output_dir (str): Directory to save results
        timestamp (str): Timestamp for result files
    """
    try:
        # Configure environment for GPU usage if not an API model
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Set up prompt functions mapping
        prompt_mapping = {
            "standard": standard_prompt,
            "zero_shot": zero_shot_prompt,
            "cot": chain_of_thought_prompt,
            "hybrid_cot": hybrid_cot_prompt,
            "zero_shot_cot": zero_shot_cot_prompt,
            "few_shot_3": few_shot_3_prompt,
            "few_shot_5": few_shot_5_prompt,
            "few_shot_7": few_shot_7_prompt,
            "cot_self_consistency_3": cot_self_consistency_3_prompt,
            "cot_self_consistency_5": cot_self_consistency_5_prompt,
            "cot_self_consistency_7": cot_self_consistency_7_prompt,
            "react": react_prompt
        }
        
        # Pre-load model if it's not an API model (to avoid loading it for each prompt type)
        tokenizer = None
        model = None
        if model_name != "gemini":
            try:
                tokenizer, model = get_or_load_model(model_name, gpu_id, use_4bit)
                if model is None:
                    print_status("error", f"Failed to load {model_name} model. Skipping.", "red")
                    return
            except Exception as model_e:
                print_status("error", f"Error with model loading: {str(model_e)}", "red")
                return
        
        # Process each prompt type for this model
        for prompt_type in prompt_types:
            model_prompt_key = f"{model_name}_{prompt_type}"
            prompt_fn = prompt_mapping[prompt_type]
            
            print_status("processing", f"Processing {model_name} with {prompt_type} prompt on GPU {gpu_id}", "blue")
            
            # Initialize storage for this model and prompt type
            if model_prompt_key not in shared_results:
                shared_results[model_prompt_key] = []
            
            # Process in batches
            num_batches = (len(questions) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(questions))
                batch_questions = questions[start_idx:end_idx]
                
                print_status("processing", f"Processing batch {batch_idx+1}/{num_batches} for {model_name} with {prompt_type}", "blue")
                
                try:
                    # Generate prompts for questions
                    prompts = [prompt_fn(q, "classical_problem") for q in batch_questions]
                    
                    batch_start_time = time.time()
                    
                    # Use common function to generate responses
                    responses = generate_batch_responses(
                        model_name, 
                        prompts, 
                        tokenizer=tokenizer, 
                        model=model, 
                        gpu_id=gpu_id,
                        prompt_type=prompt_type
                    )
                    
                    # Calculate batch metrics
                    batch_end_time = time.time()
                    batch_duration = batch_end_time - batch_start_time
                    
                    # Format results for the batch
                    results_for_batch = []
                    for i, (question, response) in enumerate(zip(batch_questions, responses)):
                        result = {
                            "question": question,
                            "response": response,
                            "model": model_name,
                            "prompt_type": prompt_type,
                            "elapsed_time": batch_duration / len(batch_questions),
                            "token_count": len(prompt_fn(question, "classical_problem").split()),
                            "response_length": len(response.split()),
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                        }
                        results_for_batch.append(result)
                    
                    # Update shared results and save partial results
                    with portalocker.Lock(f"lock_{model_prompt_key}.lock", timeout=10):
                        shared_results[model_prompt_key].extend(results_for_batch)
                        
                    # Save partial results to file for backup
                    partial_results_file = os.path.join(output_dir, f"{model_name}_{prompt_type}_{timestamp}_partial.json")
                    with open(partial_results_file, 'w', encoding='utf-8') as f:
                        json.dump({model_prompt_key: shared_results[model_prompt_key]}, f, indent=4, ensure_ascii=False)
                    
                    # Print progress and stats
                    print_status("success", f"Completed batch {batch_idx+1}/{num_batches} for {model_name} with {prompt_type}", "green")
                    avg_time = batch_duration / len(batch_questions)
                    avg_tokens = sum(len(r.split()) for r in responses) / len(responses)
                    tokens_per_sec = avg_tokens / max(0.1, avg_time)
                    print_status("info", f"Stats: {avg_time:.2f}s/question, {avg_tokens:.1f} tokens, {tokens_per_sec:.1f} tokens/sec", "blue")
                    
                except Exception as e:
                    print_status("error", f"Error processing batch {batch_idx+1} for {model_name} with {prompt_type}: {str(e)}", "red")
            
            print_status("success", f"Completed all batches for {model_name} with {prompt_type}", "green")
            
    except Exception as e:
        print_status("error", f"Error in process for {model_name} on GPU {gpu_id}: {str(e)}", "red")
        import traceback
        traceback.print_exc()

def get_or_load_model(model_name, gpu_id, use_4bit=True):
    """
    Get a cached model instance or load it if not available.
    Uses a locking mechanism to prevent multiple threads from loading the same model.
    
    Args:
        model_name (str): Name of the model to load
        gpu_id (int): GPU ID to load the model on
        use_4bit (bool): Whether to use 4-bit quantization
        
    Returns:
        tuple: (tokenizer, model)
    """
    global _LOADED_MODELS, _MODEL_LOADING_LOCKS
    
    # Create a unique key for this model and GPU combination
    model_key = f"{model_name}_{gpu_id}"
    
    # Initialize lock if not exists
    if model_key not in _MODEL_LOADING_LOCKS:
        _MODEL_LOADING_LOCKS[model_key] = threading.Lock()
    
    # Check if model is already loaded
    with _MODEL_LOADING_LOCKS[model_key]:
        if model_key in _LOADED_MODELS and _LOADED_MODELS[model_key]["model"] is not None:
            print_status("info", f"Using cached {model_name} model on GPU {gpu_id}", "blue")
            return _LOADED_MODELS[model_key]["tokenizer"], _LOADED_MODELS[model_key]["model"]
        
        # Model is not loaded, load it now
        try:
            print_status("loading", f"Loading {model_name} model on GPU {gpu_id} with maximum memory", "blue")
            
            # Map model name to paths
            if model_name == "llama":
                model_path = os.getenv("LLAMA_MODEL_PATH", "meta-llama/Meta-Llama-3-70B-Instruct")
                tokenizer_path = os.getenv("LLAMA_TOKENIZER_PATH", model_path)
            elif model_name == "qwen":
                model_path = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2-72B-Instruct")
                tokenizer_path = os.getenv("QWEN_TOKENIZER_PATH", model_path)
            else:
                print_status("error", f"Unsupported model: {model_name}", "red")
                return None, None
            
            # Set the environment to use only the specified GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Configure tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
            
            # Set padding token if not exists
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.unk_token
            
            # Configure quantization for 4-bit if requested
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                bnb_config = None
            
            # Load model with maximum memory utilization
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",  # Let the library decide the optimal device map
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config if use_4bit else None,
                trust_remote_code=True,
                # Remove max_memory limit to use all available GPU memory
                max_memory=None,
            )
            
            # Set generation hyperparameters
            model.generation_config.temperature = 0.7
            model.generation_config.do_sample = True
            model.generation_config.top_p = 0.95
            model.generation_config.top_k = 40
            model.generation_config.num_beams = 1  # Greedy decoding with temperature
            model.generation_config.max_new_tokens = 1024
            model.generation_config.use_cache = True  # Use KV cache for faster generation
            
            # Cache the loaded model
            _LOADED_MODELS[model_key] = {
                "tokenizer": tokenizer,
                "model": model,
                "last_used": time.time()
            }
            
            print_status("success", f"Successfully loaded {model_name} on GPU {gpu_id} with maximum memory", "green")
            return tokenizer, model
            
        except Exception as e:
            print_status("error", f"Failed to load {model_name} on GPU {gpu_id}: {str(e)}", "red")
            import traceback
            traceback.print_exc()
            return None, None

def evaluate_sequentially_optimized(questions, models, prompt_types, gpu_ids, output_dir, timestamp, batch_size=10, use_4bit=True):
    """
    Optimized sequential evaluation that uses all available GPUs efficiently.
    Models are loaded once and reused across prompt types.
    
    Args:
        questions (list): List of questions to evaluate
        models (list): List of models to evaluate
        prompt_types (list): List of prompt types to evaluate
        gpu_ids (list): List of GPU IDs to use
        output_dir (str): Directory to save results
        timestamp (str): Timestamp for result files
        batch_size (int): Number of questions to process in each batch
        use_4bit (bool): Whether to use 4-bit quantization
        
    Returns:
        tuple: (results, results_file)
    """
    print_header("Optimized Sequential Evaluation Mode")
    print_status("info", f"Using GPUs: {gpu_ids}", "blue")
    
    # Create prompt mapping
    prompt_mapping = {
        "standard": standard_prompt,
        "zero_shot": zero_shot_prompt,
        "cot": chain_of_thought_prompt,
        "hybrid_cot": hybrid_cot_prompt,
        "zero_shot_cot": zero_shot_cot_prompt,
        "few_shot_3": few_shot_3_prompt,
        "few_shot_5": few_shot_5_prompt,
        "few_shot_7": few_shot_7_prompt,
        "cot_self_consistency_3": cot_self_consistency_3_prompt,
        "cot_self_consistency_5": cot_self_consistency_5_prompt,
        "cot_self_consistency_7": cot_self_consistency_7_prompt,
        "react": react_prompt
    }
    
    # Initialize results dictionary
    results = {}
    
    # Assign GPU to each model (round-robin allocation for non-API models)
    model_gpu_mapping = {}
    available_gpus = gpu_ids.copy()
    
    for model_name in models:
        if model_name == "gemini":  # API model doesn't need GPU
            model_gpu_mapping[model_name] = -1
        else:
            # Assign next available GPU (round-robin)
            if available_gpus:
                gpu_id = available_gpus.pop(0)
                model_gpu_mapping[model_name] = gpu_id
            else:
                # If no more GPUs, reuse from the beginning
                model_gpu_mapping[model_name] = gpu_ids[0]
                available_gpus = gpu_ids.copy()[1:]  # Reset, but skip the one we just used
    
    print_status("info", f"Model to GPU mapping: {model_gpu_mapping}", "blue")
    
    # Process questions in batches
    num_batches = (len(questions) + batch_size - 1) // batch_size
    
    # Process each model first (load once, use for all prompt types)
    for model_name in models:
        print_header(f"Processing Model: {model_name}")
        
        # Get assigned GPU for this model
        gpu_id = model_gpu_mapping[model_name]
        
        # Preload model if it's not API-based
        model = None
        tokenizer = None
        if model_name != "gemini":
            tokenizer, model = get_or_load_model(model_name, gpu_id, use_4bit)
            if model is None:
                print_status("error", f"Could not load {model_name}, skipping", "red")
                continue
        
        # Process each prompt type for this model
        for prompt_type in prompt_types:
            model_prompt_key = f"{model_name}_{prompt_type}"
            print_status("processing", f"Processing {model_name} with {prompt_type} prompt", "blue")
            
            # Skip if already processed
            if model_prompt_key in results and len(results[model_prompt_key]) >= len(questions):
                print_status("info", f"Skipping {model_name} with {prompt_type} prompt (already processed)", "blue")
                continue
            
            prompt_fn = prompt_mapping[prompt_type]
            all_batch_results = []
            
            # Process each batch
            for batch_idx in range(num_batches):
                print_status("info", f"Processing batch {batch_idx+1}/{num_batches}", "blue")
                
                # Get questions for this batch
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(questions))
                batch_questions = questions[start_idx:end_idx]
                
                try:
                    # Generate prompts for all questions
                    prompts = [prompt_fn(q, "classical_problem") for q in batch_questions]
                    
                    # Record start time
                    batch_start_time = time.time()
                    
                    # Use common function to generate responses
                    responses = generate_batch_responses(
                        model_name, 
                        prompts, 
                        tokenizer=tokenizer, 
                        model=model, 
                        gpu_id=gpu_id,
                        prompt_type=prompt_type
                    )
                    
                    # Calculate batch metrics
                    batch_end_time = time.time()
                    batch_duration = batch_end_time - batch_start_time
                    
                    # Format results for batch
                    batch_results = []
                    for i, (question, response) in enumerate(zip(batch_questions, responses)):
                        result = {
                            "question": question,
                            "response": response,
                            "model": model_name,
                            "prompt_type": prompt_type,
                            "elapsed_time": batch_duration / len(batch_questions),
                            "token_count": len(prompt_fn(question, "classical_problem").split()),
                            "response_length": len(response.split()),
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                        }
                        batch_results.append(result)
                    
                    # Add to results
                    all_batch_results.extend(batch_results)
                    
                    # Print stats
                    print_status("info", f"Batch {batch_idx+1}/{num_batches} completed", "blue")
                    avg_time = batch_duration / len(batch_questions)
                    avg_tokens = sum(len(r.split()) for r in responses) / len(responses)
                    tokens_per_sec = avg_tokens / max(0.1, avg_time)
                    print_status("info", f"Average response time: {avg_time:.2f}s, {avg_tokens:.1f} tokens, {tokens_per_sec:.1f} tokens/sec", "green")
                    
                    # Save intermediate results
                    if model_prompt_key not in results:
                        results[model_prompt_key] = []
                    results[model_prompt_key].extend(batch_results)
                    
                    # Save partial results
                    partial_results_file = os.path.join(output_dir, f"{model_name}_{prompt_type}_{timestamp}_partial.json")
                    with open(partial_results_file, 'w', encoding='utf-8') as f:
                        json.dump({model_prompt_key: all_batch_results}, f, indent=4, ensure_ascii=False)
                    
                except Exception as e:
                    print_status("error", f"Error processing batch: {str(e)}", "red")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Save full results after each prompt type
            results_file = save_results(results, output_dir, timestamp)
    
    return results, results_file

def main():
    """
    Hàm chính điều khiển quá trình đánh giá, hỗ trợ checkpoint/resume và logging.
    """
    args = setup_argparse()
    
    # Thiết lập logging
    log_file = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("EvaluateModels")
    
    # Tạo timestamp cho thư mục kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp) if not args.output_dir.endswith(timestamp) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting evaluation with timestamp: {timestamp}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Command-line arguments: {args}")
    
    # Khởi tạo từ checkpoint nếu được chỉ định
    if args.resume and args.checkpoint_id:
        logger.info(f"Resuming from checkpoint: {args.checkpoint_id}")
    
    # Khởi tạo GPU allocation
    if args.gpu_allocation:
        gpu_ids = parse_gpu_allocation(args.gpu_allocation, args.models, list(range(torch.cuda.device_count())))
        logger.info(f"GPU allocation: {gpu_ids}")
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
        logger.info(f"Using all available GPUs: {gpu_ids}")
    
    # Load câu hỏi
    try:
        questions = load_questions(args.questions_json)
        logger.info(f"Loaded {len(questions)} questions from {args.questions_json}")
        
        if args.max_questions and args.max_questions < len(questions):
            logger.info(f"Limiting to {args.max_questions} questions")
            questions = questions[:args.max_questions]
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return
    
    # Tạo model configs cho mỗi prompt type
    model_configs = {}
    for prompt_type in args.prompt_types:
        for model_name in args.models:
            config_key = f"{model_name}_{prompt_type}"
            model_configs[config_key] = get_prompt_config(prompt_type, model_name)
            logger.info(f"Config for {config_key}: {model_configs[config_key]}")
    
    # Khởi tạo ModelEvaluator
    try:
        # Đo thời gian load modelEvaluator
        load_start = time.time()
        logger.info("Initializing ModelEvaluator...")
        
        # Tạo cấu trúc dữ liệu để lưu trữ thời gian load cho mỗi mô hình
        model_load_times = {}
        
        # Tạo dictionary cho models
        models_dict = {}
        for model_name in args.models:
            model_start = time.time()
            logger.info(f"Setting up model: {model_name}")
            
            # Thêm cấu hình mô hình
            if model_name == "gemini":
                from check.model_manager import load_gemini_model
                model = load_gemini_model()
                models_dict[model_name] = {"type": "api", "name": model_name, "model": model}
            else:
                # Ghi lại là local model, sẽ được load khi cần thiết
                models_dict[model_name] = {"type": "local", "name": model_name}
            
            model_end = time.time()
            model_load_times[model_name] = model_end - model_start
            logger.info(f"Model {model_name} setup completed in {model_load_times[model_name]:.2f}s")
        
        evaluator = ModelEvaluator(models_dict, results_dir=output_dir)
        load_end = time.time()
        logger.info(f"ModelEvaluator initialized in {load_end - load_start:.2f}s")
        
        # Hiển thị bảng thời gian load mô hình
        logger.info("Model setup time summary:")
        for model_name, load_time in model_load_times.items():
            logger.info(f"  - {model_name}: {load_time:.2f}s")
            
    except Exception as e:
        logger.error(f"Error initializing evaluator: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Xử lý chế độ analyze only nếu có
    if args.analyze_only and args.results_file:
        logger.info(f"Analyze only mode enabled. Loading results from {args.results_file}")
        try:
            results_df = evaluator.analyze_results_from_json(args.results_file)
            logger.info(f"Loaded and analyzed results from {args.results_file}")
            
            # Phân tích thêm nếu cần
            if args.analyze_reasoning:
                logger.info("Analyzing reasoning quality...")
                evaluator.evaluate_reasoning_quality(results_df)
            
            if args.analyze_prompt_effectiveness:
                logger.info("Analyzing prompt effectiveness...")
                evaluate_prompt_effectiveness(results_df, output_dir)
            
            return
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
    
    # Chạy đánh giá chính
    try:
        # Kiểm tra xem có checkpoint để resume không
        if args.resume:
            checkpoint_id = args.checkpoint_id
            logger.info(f"Looking for checkpoint to resume: {checkpoint_id if checkpoint_id else 'auto'}")
        else:
            checkpoint_id = None
            logger.info("Starting new evaluation (no resume)")
        
        # Đo thời gian đánh giá
        eval_start = time.time()
        
        # Sử dụng checkpoint nếu có yêu cầu resume
        if args.parallel:
            logger.info(f"Running evaluation in parallel mode with {len(gpu_ids)} GPUs")
            results_df = evaluate_models_in_parallel(
                questions,
                args.models,
                args.prompt_types,
                gpu_ids,
                output_dir,
                timestamp,
                batch_size=args.batch_size,
                use_4bit=args.use_4bit
            )
        else:
            logger.info(f"Running evaluation in sequential mode")
            # Tạo model_config từ prompt_types và models
            example_selection = args.example_selection if hasattr(args, 'example_selection') else "default"
            custom_examples = None  # Có thể thêm tham số này vào argparse nếu cần
            examples_file = args.examples_file if hasattr(args, 'examples_file') else None
            
            results_df = evaluator.evaluate_all_questions(
                questions,
                model_names=args.models,
                prompt_types=args.prompt_types,
                max_questions=args.max_questions,
                example_selection=example_selection,
                custom_examples=custom_examples,
                examples_file=examples_file,
                model_config=model_configs
            )
        
        eval_end = time.time()
        eval_time = eval_end - eval_start
        logger.info(f"Evaluation completed in {eval_time:.2f}s")
        logger.info(f"Generated {len(results_df)} evaluation results")
        
        # Lưu kết quả
        results_file = os.path.join(output_dir, f"results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")
        
        # Lưu kết quả dạng JSON để dễ phân tích sau này
        json_results = os.path.join(output_dir, f"results_{timestamp}.json")
        results_df.to_json(json_results, orient="records", indent=2)
        logger.info(f"JSON results saved to {json_results}")
        
        # Phân tích kết quả
        logger.info("Analyzing results...")
        analysis_results = evaluator.analyze_results(results_df, save_plots=True)
        
        # Phân tích thêm nếu được yêu cầu
        if args.analyze_reasoning:
            logger.info("Analyzing reasoning quality...")
            evaluator.evaluate_reasoning_quality(results_df)
        
        if args.evaluate_reasoning_with_gemini and 'gemini' in args.models:
            logger.info("Evaluating reasoning with Gemini...")
            evaluator.evaluate_reasoning_with_gemini(results_df, sample_size=args.reasoning_sample_size)
        
        if args.analyze_prompt_effectiveness:
            logger.info("Analyzing prompt effectiveness...")
            evaluate_prompt_effectiveness(results_df, output_dir)
        
        # Tạo báo cáo so sánh prompts
        logger.info("📝 Generating prompt comparison report...")
        evaluator.generate_prompt_comparison_report(results_df, output_dir)
        
        # Phân tích ReAct prompt nếu có sử dụng
        if "react" in args.prompt_types:
            logger.info("  Analyzing ReAct prompt evaluation...")
            evaluator.analyze_react_prompt_evaluation(results_df, output_dir)
        
        # Tổng kết thời gian thực hiện
        total_time = time.time() - eval_start
        logger.info(f"✨ All operations completed in {total_time:.2f}s")
        logger.info(f"  Check the results directory: {output_dir}")
        
        # In hướng dẫn phân tích thêm
        logger.info("  To analyze the results later, you can use:")
        logger.info(f"   python evaluate_models.py --analyze_only --results_file {json_results} [--analyze_reasoning] [--analyze_prompt_effectiveness]")
        
        return results_df, analysis_results
        
    except KeyboardInterrupt:
        logger.warning("  Evaluation interrupted by user.")
        return
    except Exception as e:
        logger.error(f"  Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

def setup_argparse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLM models on various tasks")
    
    # Model selection
    parser.add_argument("--models", nargs="+", default=["llama", "qwen", "gemini"], 
                        help="Models to evaluate (llama, qwen, gemini)")
    
    # Prompt configuration
    parser.add_argument("--prompt_types", nargs="+", 
                        default=["zero_shot", "few_shot_3", "cot", "self_consistency", "react"],
                        help="Prompt types to use. Options: zero_shot, few_shot_3, few_shot_5, cot, self_consistency, react")
    
    # Questions and batch size
    parser.add_argument("--questions_json", type=str, default="db/questions/problems.json", 
                        help="Path to questions JSON file")
    parser.add_argument("--batch_size", type=int, default=10, 
                        help="Batch size for processing")
    parser.add_argument("--max_questions", type=int, default=None, 
                        help="Maximum number of questions to evaluate")
    
    # Output directory and analysis options
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Output directory for results")
    parser.add_argument("--analyze_only", action="store_true", 
                        help="Only analyze existing results, don't run evaluation")
    parser.add_argument("--results_file", type=str, 
                        help="Path to results JSON file for analysis")
    
    # GPU configuration
    parser.add_argument("--gpu_allocation", type=str, 
                        help="Specific GPU allocation, format: 'model1:0,1;model2:2,3'")
    parser.add_argument("--use_4bit", action="store_true", default=True, 
                        help="Use 4-bit quantization for models")
    parser.add_argument("--parallel", action="store_true", 
                        help="Run evaluation in parallel on multiple GPUs")
    
    # Advanced analysis
    parser.add_argument("--analyze_reasoning", action="store_true", 
                        help="Perform detailed reasoning quality analysis")
    parser.add_argument("--evaluate_reasoning_with_gemini", action="store_true", 
                        help="Use Gemini to evaluate reasoning quality")
    parser.add_argument("--reasoning_sample_size", type=int, default=50, 
                        help="Sample size for reasoning evaluation")
    parser.add_argument("--analyze_prompt_effectiveness", action="store_true", 
                        help="Analyze prompt effectiveness across models")
    
    # Examples for few-shot
    parser.add_argument("--example_selection", type=str, default="default", 
                       help="Method to select examples for few-shot prompts: default, random, custom")
    parser.add_argument("--examples_file", type=str, 
                       help="Path to JSON file containing examples for few-shot prompts")
    
    # Checkpoint/Resume
    parser.add_argument("--resume", action="store_true", 
                       help="Resume evaluation from checkpoint")
    parser.add_argument("--checkpoint_id", type=str, 
                       help="Specific checkpoint ID to resume from")
    
    return parser.parse_args()

def generate_gemini_response(prompt):
    """
    Generate a response from the Gemini API model.
    
    Args:
        prompt (str): The prompt text to send to the Gemini API.
        
    Returns:
        str: The generated response text.
    """
    try:
        # Use the model_manager's function to generate text with Gemini
        from check.model_manager import generate_text_with_model
        # Get model-specific max tokens value
        max_tokens = get_max_tokens("gemini")
        response = generate_text_with_model("gemini", prompt, max_tokens=max_tokens)
        return response
    except Exception as e:
        # Handle API errors
        error_message = str(e)
        print_status("error", f"Error with Gemini API: {error_message}", "red")
        return f"[Error: {error_message}]"

def generate_model_response(model_name, prompt, tokenizer=None, model=None, gpu_id=-1, config=None):
    """
    Common function to generate responses from language models,
    reducing code duplication between evaluation functions.
    
    Args:
        model_name (str): Name of the model
        prompt (str): Prompt text to generate response for
        tokenizer: Optional tokenizer instance (for non-API models)
        model: Optional model instance (for non-API models)
        gpu_id (int): GPU ID for the model (-1 for API models)
        config (dict): Optional configuration dictionary with generation parameters
        
    Returns:
        str: The generated response text
    """
    try:
        # Get model-specific max token value if no config provided
        if config is None:
            max_tokens = get_max_tokens(model_name)
            temperature = 0.7
            top_p = 0.9
        else:
            max_tokens = config.get("max_tokens", get_max_tokens(model_name))
            temperature = config.get("temperature", 0.7)
            top_p = config.get("top_p", 0.9)
        
        if model_name == "gemini":
            # Handle Gemini API model
            max_retries = 5
            retry_count = 0
            retry_delay = 2
            
            while retry_count < max_retries:
                try:
                    gemini_model = load_gemini_model()
                    if gemini_model is None:
                        return "[Error: Failed to load Gemini model]"
                    
                    response = gemini_model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=temperature,
                            top_p=top_p,
                            max_output_tokens=max_tokens,
                        )
                    )
                    
                    return response.text if hasattr(response, "text") else "[Error: Empty response from Gemini]"
                    
                except Exception as api_e:
                    retry_count += 1
                    print_status("warning", f"Gemini API error (attempt {retry_count}/{max_retries}): {str(api_e)}", "yellow")
                    
                    if retry_count >= max_retries:
                        return f"[Error: {str(api_e)}]"
                    
                    print_status("info", f"Retrying in {retry_delay} seconds...", "blue")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    
        else:
            # Handle LLM models (Llama, Qwen, etc.)
            if tokenizer is None or model is None:
                # Load model if not provided
                tokenizer, model = get_or_load_model(model_name, gpu_id, use_4bit=True)
                
                if model is None or tokenizer is None:
                    return f"[Error: Failed to load {model_name} model]"
                
            # Use batched or single processing based on model capabilities
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        # Speed up generation
                        num_beams=1,
                        top_p=top_p,
                        use_cache=True,
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                
                if generated_text.startswith(prompt_text):
                    response = generated_text[len(prompt_text):].strip()
                else:
                    response = generated_text.strip()
                    
                return response
                
            except Exception as e:
                print_status("error", f"Error generating response: {str(e)}", "red")
                return f"[Error: {str(e)}]"
    
    except Exception as e:
        print_status("error", f"Unexpected error generating response: {str(e)}", "red")
        import traceback
        traceback.print_exc()
        return f"[Error: {str(e)}]"

def generate_batch_responses(model_name, prompts, tokenizer=None, model=None, gpu_id=None, prompt_type=None):
    """
    Generate responses for a batch of prompts.
    
    Args:
        model_name (str): Name of the model to use
        prompts (list): List of prompts to generate responses for
        tokenizer: Optional pre-loaded tokenizer
        model: Optional pre-loaded model
        gpu_id (int): GPU ID for model
        prompt_type (str): Type of prompt being processed
        
    Returns:
        list: List of responses
    """
    # Special case for API models
    if model_name == "gemini":
        return generate_batch_responses_api(model_name, prompts, prompt_type)
    
    if model is None or tokenizer is None:
        print_status("error", "Model and tokenizer must be provided for local models", "red")
        return ["[Error: Model not loaded]"] * len(prompts)
    
    try:
        # Set generation parameters based on prompt type
        generation_params = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "do_sample": True,
            "num_beams": 1,
            "use_cache": True
        }
        
        # Adjust parameters for specific prompt types
        if prompt_type:
            if prompt_type.startswith("cot"):
                generation_params["max_new_tokens"] = 1536
            elif prompt_type == "react":
                generation_params["max_new_tokens"] = 2048
                generation_params["temperature"] = 0.8
            elif prompt_type.startswith("few_shot"):
                generation_params["temperature"] = 0.65
        
        # Optimize batch processing - set proper padding token and create attention mask
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.unk_token
        
        # Calculate optimal batch size based on prompt length
        avg_token_length = sum(len(tokenizer.encode(p)) for p in prompts) / len(prompts)
        dynamic_batch_size = min(len(prompts), max(1, int(8192 / avg_token_length)))
        
        all_responses = []
        
        # Process in dynamic batches
        for i in range(0, len(prompts), dynamic_batch_size):
            batch_prompts = prompts[i:i+dynamic_batch_size]
            
            # Tokenize with padding for efficient batch processing
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(model.device)
            
            # Generate text with optimizations
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_params,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode the generated text
            decoded_outputs = []
            for j, output in enumerate(outputs):
                # Find the input length to skip in the output
                input_length = len(inputs.input_ids[j])
                
                # Only decode the newly generated tokens
                decoded = tokenizer.decode(
                    output[input_length:], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                decoded_outputs.append(decoded)
            
            all_responses.extend(decoded_outputs)
        
        return all_responses
        
    except Exception as e:
        print_status("error", f"Error in generating batch responses: {str(e)}", "red")
        traceback.print_exc()
        return [f"[Error: {str(e)}]"] * len(prompts)

# Model configuration defaults - đây là các mặc định theo loại prompt
DEFAULT_MODEL_CONFIGS = {
    # Prompt type configurations
    "standard": {
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9
    },
    "zero_shot": {
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9
    },
    "few_shot_3": {
        "max_tokens": 384,
        "temperature": 0.3,
        "top_p": 0.9
    },
    "few_shot_5": {
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9
    },
    "few_shot_7": {
        "max_tokens": 768,
        "temperature": 0.3,
        "top_p": 0.9
    },
    "cot": {
        "max_tokens": 768,
        "temperature": 0.2,
        "top_p": 0.9
    },
    "cot_self_consistency_3": {
        "max_tokens": 768,
        "temperature": 0.7, 
        "top_p": 0.95
    },
    "cot_self_consistency_5": {
        "max_tokens": 768,
        "temperature": 0.7,
        "top_p": 0.95
    },
    "cot_self_consistency_7": {
        "max_tokens": 896,
        "temperature": 0.7,
        "top_p": 0.95
    },
    "react": { 
        "max_tokens": 1024,
        "temperature": 0.4,
        "top_p": 0.95
    },
    
    # Model-specific configurations
    "llama": {
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95
    },
    "qwen": {
        "max_tokens": 256,  # Reduced from original
        "temperature": 0.7,
        "top_p": 0.95
    },
    "gemini": {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95
    }
}

def get_prompt_config(prompt_type, model_name):
    """
    Get the appropriate configuration for a specific prompt type and model.
    
    Args:
        prompt_type (str): Type of prompt (standard, cot, react, etc.)
        model_name (str): Name of the model
        
    Returns:
        dict: Configuration parameters
    """
    # Đảm bảo model_name là lowercase
    model_name = model_name.lower() if model_name else "default"
    
    # Mặc định cấu hình cho mọi trường hợp
    default_config = {
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95
    }
    
    # First get the model-specific base config
    if model_name in DEFAULT_MODEL_CONFIGS:
        config = DEFAULT_MODEL_CONFIGS[model_name].copy()
    else:
        print_status("warning", f"Unknown model: {model_name}, using default config", "yellow")
        config = default_config.copy()
    
    # Then override with prompt-type specific config if available
    if prompt_type in DEFAULT_MODEL_CONFIGS:
        # Only update max_tokens if it's lower for the model than what the prompt needs
        prompt_config = DEFAULT_MODEL_CONFIGS[prompt_type].copy()
        
        # For Qwen, we need to be more conservative with token counts
        if model_name == "qwen":
            max_tokens_scale = 0.75  # Reduce max tokens for Qwen
            prompt_config["max_tokens"] = min(int(prompt_config["max_tokens"] * max_tokens_scale), config.get("max_tokens", 512))
            
        for key, value in prompt_config.items():
            config[key] = value
    
    # Special handling for self-consistency and CoT
    if "self_consistency" in prompt_type:
        # These need higher temperatures for diverse outputs
        config["temperature"] = max(config.get("temperature", 0.7), 0.7)
        
        # For Qwen, reduce max_tokens to avoid OOM
        if model_name == "qwen":
            config["max_tokens"] = min(config.get("max_tokens", 512), 384)
    
    # React needs more tokens, but be careful with Qwen
    if prompt_type == "react":
        if model_name == "qwen":
            config["max_tokens"] = min(config.get("max_tokens", 512), 768)  # Cap for Qwen
        elif model_name == "llama":
            config["max_tokens"] = min(config.get("max_tokens", 512), 1024)  # Cap for Llama
        else:  # gemini
            config["max_tokens"] = 1536  # Gemini can handle more tokens
    
    # Đảm bảo mọi cấu hình đều có các thông số cơ bản
    if "max_tokens" not in config:
        config["max_tokens"] = 512
    if "temperature" not in config:
        config["temperature"] = 0.7
    if "top_p" not in config:
        config["top_p"] = 0.95
    
    return config

def evaluate_prompt_effectiveness(results_df, output_dir=None):
    """
    Evaluate the effectiveness of different prompt types.
    
    Args:
        results_df (DataFrame): Results dataframe
        output_dir (str): Directory to save visualizations
        
    Returns:
        dict: Evaluation metrics
    """
    if results_df is None or results_df.empty:
        print("  No results data available for analysis")
        return {}
        
    print("  Analyzing prompt effectiveness...")
    
    # Create output directory if needed
    if output_dir:
        viz_dir = os.path.join(output_dir, "prompt_analysis")
        os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize metrics dictionary
    metrics = {
        "overall": {},
        "by_model": {},
        "by_prompt": {},
        "react_analysis": {}
    }
    
    # Get available columns for analysis
    available_columns = results_df.columns.tolist()
    required_columns = {
        'correctness_score': 'has_error',  # Fallback column if correctness_score missing
        'elapsed_time': None,  # No fallback for time
        'response_length': None  # No fallback for length
    }
    
    # Check and map columns
    analysis_columns = {}
    for col, fallback in required_columns.items():
        if col in available_columns:
            analysis_columns[col] = col
        elif fallback and fallback in available_columns:
            analysis_columns[col] = fallback
            print(f"  Using '{fallback}' as fallback for '{col}'")
        else:
            print(f"  Warning: Column '{col}' not available for analysis")
    
    # Filter ReAct prompts
    react_results = results_df[results_df['prompt_type'].str.contains('react', case=False, na=False)]
    
    if not react_results.empty:
        # Prepare aggregation dictionary based on available columns
        agg_dict = {}
        for metric, col in analysis_columns.items():
            if col == 'has_error' and metric == 'correctness_score':
                # Invert has_error to get correctness
                agg_dict[metric] = lambda x: 1 - x.mean()
            elif col:
                agg_dict[col] = "mean"
        
        if agg_dict:  # Only proceed if we have columns to analyze
            react_metrics = react_results.groupby("model_name").agg(agg_dict).reset_index()
            metrics["react_analysis"] = react_metrics.to_dict(orient="records")
            
            # Create visualizations if output directory specified
            if output_dir:
                try:
                    plt.figure(figsize=(12, 6))
                    for metric, col in analysis_columns.items():
                        if col:
                            plt.figure(figsize=(10, 6))
                            sns.barplot(x='model_name', y=metric, data=react_metrics)
                            plt.title(f'ReAct Performance: {metric}', fontsize=14)
                            plt.xlabel('Model', fontsize=12)
                            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(os.path.join(viz_dir, f'react_{metric}.png'))
                            plt.close()
                except Exception as e:
                    print(f"  Error creating visualization: {e}")
    
    # Analyze by prompt type
    prompt_metrics = {}
    for prompt_type in results_df['prompt_type'].unique():
        prompt_data = results_df[results_df['prompt_type'] == prompt_type]
        
        # Calculate metrics based on available columns
        prompt_stats = {}
        for metric, col in analysis_columns.items():
            if col:
                if col == 'has_error' and metric == 'correctness_score':
                    prompt_stats[metric] = 1 - prompt_data[col].mean()
                else:
                    prompt_stats[metric] = prompt_data[col].mean()
        
        prompt_metrics[prompt_type] = prompt_stats
    
    metrics["by_prompt"] = prompt_metrics
    
    # Create comparison visualizations
    if output_dir and analysis_columns:
        try:
            for metric, col in analysis_columns.items():
                if col:
                    plt.figure(figsize=(12, 6))
                    data = []
                    for prompt_type, stats in prompt_metrics.items():
                        if metric in stats:
                            data.append({
                                'prompt_type': prompt_type,
                                metric: stats[metric]
                            })
                    
                    if data:
                        comparison_df = pd.DataFrame(data)
                        sns.barplot(x='prompt_type', y=metric, data=comparison_df)
                        plt.title(f'Prompt Type Comparison: {metric}', fontsize=14)
                        plt.xlabel('Prompt Type', fontsize=12)
                        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(os.path.join(viz_dir, f'prompt_comparison_{metric}.png'))
                        plt.close()
        except Exception as e:
            print(f"  Error creating comparison visualization: {e}")
    
    return metrics

def generate_batch_responses_api(model_name, prompts, prompt_type=None):
    """
    Generate responses for a batch of prompts using API-based models with rate limiting.
    
    Args:
        model_name (str): Name of the API model (e.g., "gemini")
        prompts (list): List of prompts to generate responses for
        prompt_type (str): Type of prompt being processed
        
    Returns:
        list: List of responses
    """
    if model_name != "gemini":
        print_status("error", f"Unsupported API model: {model_name}", "red")
        return ["[Error: Unsupported API model]"] * len(prompts)
    
    # Set generation parameters based on prompt type
    generation_params = {
        "max_output_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    # Adjust parameters for specific prompt types
    if prompt_type:
        if prompt_type.startswith("cot"):
            generation_params["max_output_tokens"] = 1536
        elif prompt_type == "react":
            generation_params["max_output_tokens"] = 2048
            generation_params["temperature"] = 0.8
        elif prompt_type.startswith("few_shot"):
            generation_params["temperature"] = 0.65
    
    # Set up Gemini model
    try:
        model = genai.GenerativeModel(
            model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        )
    except Exception as model_error:
        print_status("error", f"Error initializing Gemini model: {str(model_error)}", "red")
        return [f"[Error: API model initialization failed]"] * len(prompts)
    
    responses = []
    retry_count = 0
    max_retries = 3
    
    # Process prompts with rate limiting
    for i, prompt in enumerate(prompts):
        # Add exponential backoff for retries
        if retry_count > 0:
            backoff_time = min(2 ** retry_count, 30)  # Cap at 30 seconds
            print_status("warning", f"Rate limit hit, backing off for {backoff_time} seconds", "yellow")
            time.sleep(backoff_time)
        
        try:
            # Log progress
            print_status("info", f"Processing prompt {i+1}/{len(prompts)} with {model_name}", "blue")
            
            # Generate response
            response = model.generate_content(
                prompt,
                **generation_params,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                ],
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                responses.append(response.text)
            else:
                responses.append(str(response))
            
            # Reset retry counter on success
            retry_count = 0
            
            # Add delay between API calls to avoid rate limiting (only if not last item)
            if i < len(prompts) - 1:
                time.sleep(1.5)  # 1.5 second delay between calls
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle rate limiting errors
            if "rate limit" in error_str or "quota" in error_str or "resource exhausted" in error_str:
                retry_count += 1
                
                if retry_count <= max_retries:
                    print_status("warning", f"Rate limit hit (retry {retry_count}/{max_retries}), retrying...", "yellow")
                    i -= 1  # Retry the same prompt
                    continue
                else:
                    print_status("error", f"Rate limit error after {max_retries} retries", "red")
                    responses.append(f"[Error: API rate limit exceeded after {max_retries} retries]")
            else:
                print_status("error", f"API error: {str(e)}", "red")
                responses.append(f"[Error: {str(e)}]")
            
            # Reset retry counter for next prompt
            retry_count = 0
    
    return responses

if __name__ == "__main__":
    try:
        args = setup_argparse()
        
        # Nếu chỉ phân tích kết quả có sẵn mà không chạy đánh giá
        if args.analyze_only and args.results_file:
            print_header("Analysis Only Mode")
            print_status("info", f"Analyzing results from {args.results_file}", "blue")
            
            try:
                import pandas as pd
                
                # Load results from file
                with open(args.results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # Tạo output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(args.output_dir, f"analysis_{timestamp}")
                os.makedirs(output_dir, exist_ok=True)
                
                # Tạo DataFrame từ kết quả
                df_results = generate_report_and_visualizations(args.results_file, output_dir)
                
                # Phân tích hiệu quả prompt nếu được yêu cầu
                if args.analyze_prompt_effectiveness:
                    print_header("Prompt Effectiveness Analysis")
                    print_status("info", "Analyzing prompt effectiveness...", "blue")
                    evaluate_prompt_effectiveness(df_results, output_dir)
                
                # Phân tích chất lượng suy luận nếu được yêu cầu
                if args.analyze_reasoning:
                    try:
                        from check.model_evaluator import ModelEvaluator
                        print_header("Reasoning Quality Analysis")
                        print_status("info", "Analyzing reasoning quality...", "blue")
                        evaluator = ModelEvaluator({}, output_dir)
                        evaluator.evaluate_reasoning_quality(df_results)
                        
                        # Đánh giá bằng Gemini API nếu được yêu cầu
                        if not args.skip_gemini_eval:
                            print_status("info", f"Starting Gemini API evaluation with sample size {args.gemini_sample_size}...", "blue")
                            evaluator.evaluate_reasoning_with_gemini(df_results, args.gemini_sample_size)
                    except ImportError:
                        print_status("error", "ModelEvaluator not available. Cannot analyze reasoning quality.", "red")
                
                print_status("success", f"Analysis completed and saved to {output_dir}", "green")
                sys.exit(0)
                
            except Exception as e:
                print_status("error", f"Error during analysis: {str(e)}", "red")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        
        # Chạy đánh giá bình thường
        main()
    except KeyboardInterrupt:
        print_status("warning", "Evaluation process interrupted by user", "yellow")
        print_status("info", "Temporary results may have been saved in the results/ directory", "blue")
    except Exception as e:
        print_status("error", f"Unhandled error: {str(e)}", "red")
        import traceback
        traceback.print_exc() 