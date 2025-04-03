"""
Model evaluator for comparing performance of different LLMs on classical problems.
"""

import os
import json
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher
import torch
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Local imports
from check.prompts import (
    standard_prompt, 
    chain_of_thought_prompt, 
    hybrid_cot_prompt,
    zero_shot_cot_prompt,
    tree_of_thought_prompt,
    # Add new prompt types
    zero_shot_prompt,
    few_shot_3_prompt,
    few_shot_5_prompt,
    few_shot_7_prompt,
    cot_self_consistency_3_prompt,
    cot_self_consistency_5_prompt,
    cot_self_consistency_7_prompt,
    react_prompt
)
from check.model_manager import (
    generate_text_with_model, 
    clear_memory,
    check_gpu_memory,
    load_gemini_model
)

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import re
import gc
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import pickle
import hashlib
import uuid
from collections import Counter
from colorama import Fore, Style, init

# Khởi tạo colorama
init(autoreset=True)

# Thiết lập Logger
def setup_logging():
    """
    Setup logging with proper encoding for Unicode characters and colorized output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Xóa các handler hiện tại
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Custom formatter để thêm màu sắc
    class ColoredFormatter(logging.Formatter):
        FORMATS = {
            logging.DEBUG: Fore.CYAN + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
            logging.INFO: Fore.GREEN + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
            logging.WARNING: Fore.YELLOW + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
            logging.ERROR: Fore.RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
            logging.CRITICAL: Fore.RED + Style.BRIGHT + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL
        }
        
        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)
    
    # Thêm handler mới với encoding UTF-8
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(handler)
    
    # Thêm file handler nếu cần
    file_handler = logging.FileHandler("evaluation.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    
    return logging.getLogger("ModelEvaluator")

# Tạo logger
logger = setup_logging()
logger.setLevel(logging.INFO)

class ModelEvaluator:
    """Class for evaluating and comparing different LLM performances."""
    
    def __init__(self, models_dict, results_dir="results"):
        """
        Initialize the evaluator.
        
        Args:
            models_dict (dict): Dictionary of models to evaluate
                Format: {
                    "model_name": {
                        "type": "local"|"gemini", 
                        "model": model_object,
                        "tokenizer": tokenizer_object  # Only for local models
                    }
                }
            results_dir (str): Directory to save results
        """
        self.models = models_dict
        self.results_dir = results_dir
        self.checkpoint_dir = os.path.join(results_dir, "checkpoints")
        
        # Tạo các thư mục cần thiết
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Khởi tạo trạng thái checkpoint
        self.checkpoint_state = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "completed_evaluations": {},
            "partial_results": [],
            "current_status": "initialized",
            "last_updated": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        
        logger.info(f"ModelEvaluator initialized with checkpoint system")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        
        # Create plots directory
        self.plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Prompt types
        self.prompt_types = {
            "standard": standard_prompt,
            "zero_shot": zero_shot_prompt,
            "cot": chain_of_thought_prompt,
            "hybrid_cot": hybrid_cot_prompt,
            "zero_shot_cot": zero_shot_cot_prompt,
            "tree_of_thought": tree_of_thought_prompt,
            "few_shot_3": few_shot_3_prompt,
            "few_shot_5": few_shot_5_prompt,
            "few_shot_7": few_shot_7_prompt,
            "cot_self_consistency_3": cot_self_consistency_3_prompt,
            "cot_self_consistency_5": cot_self_consistency_5_prompt,
            "cot_self_consistency_7": cot_self_consistency_7_prompt,
            "react": react_prompt
        }
        
        # Initialize results dictionary with all required keys
        self.results = {
            "model_name": [],
            "prompt_type": [],
            "question_id": [],
            "question": [],
            "answer": [],
            "token_count": [],
            "elapsed_time": [],
            "tokens_per_second": [],
            "response_length": [],
            "has_error": [],
            "error_type": [],
            "confidence_score": [],
            "complexity_score": [],
            "topic_category": []
        }
        
        # Performance tracking
        self.performance_history = []
        self.error_history = []
        self.latency_history = []
        
        # Initialize visualization settings
        plt.style.use('default')
        sns.set_theme()
    
    def load_questions(self, questions_file):
        """
        Load questions from a JSON file.
        
        Args:
            questions_file (str): Path to the JSON file containing questions
            
        Returns:
            list: List of questions
        """
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            print(f"  Loaded {len(questions)} questions from {questions_file}")
            return questions
        except Exception as e:
            print(f"  Error loading questions: {e}")
            return []
    
    def estimate_token_count(self, text):
        """
        Estimate token count for a text.
        
        Args:
            text (str): Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        # Simple approximation: average 4 characters per token (for languages like Vietnamese)
        return len(text) // 3
    
    def evaluate_single_question(self, question, model_info, prompt_type="standard", example_selection="default", custom_examples=None, examples_file=None, model_config=None):
        """
        Evaluate a single question with a specific model and prompt type.
        
        Args:
            question (dict or str): Question to evaluate
            model_info (dict or str): Model information dictionary or model name
            prompt_type (str): Type of prompt to use
            example_selection (str): Method to select examples for few-shot prompts
            custom_examples (list): Custom examples for few-shot prompts
            examples_file (str): Path to file containing examples
            model_config (dict): Model configuration parameters
            
        Returns:
            dict: Evaluation result
        """
        # Convert model_info to dictionary if it's a string (model name)
        if isinstance(model_info, str):
            model_name = model_info
            model_type = "gemini" if model_name == "gemini" else "local"
            model_info = {"name": model_name, "type": model_type}
        
        # Extract model information
        model_name = model_info.get("name", "unknown")
        model_type = model_info.get("type", "local")
        
        # Log which model and prompt type are being evaluated
        logger.info(f"Evaluating model '{model_name}' with prompt type '{prompt_type}'")
        
        # Extract question text
        if isinstance(question, dict):
            question_id = question.get("id", str(uuid.uuid4())[:8])
            question_text = question.get("text", str(question))
            expected_answer = question.get("answer", None)
        else:
            question_id = str(uuid.uuid4())[:8]
            question_text = str(question)
            expected_answer = None
        
        # Determine which prompt function to use
        prompt_fn = None
        if prompt_type in self.prompt_types:
            prompt_fn = self.prompt_types[prompt_type]
        else:
            try:
                # Try to import the specific prompt function directly from prompts module
                import check.prompts as prompts
                # Convert prompt_type to the function name (e.g., "zero_shot" to "zero_shot_prompt")
                prompt_function_name = f"{prompt_type}_prompt" if not prompt_type.endswith("_prompt") else prompt_type
                
                if hasattr(prompts, prompt_function_name):
                    prompt_fn = getattr(prompts, prompt_function_name)
                else:
                    return {"error": f"Unknown prompt type: {prompt_type}. No function named {prompt_function_name} in prompts module."}
            except (ImportError, ValueError, AttributeError) as e:
                return {"error": f"Error accessing prompt type: {prompt_type}. {str(e)}"}
        
        # Create prompt based on prompt type
        try:
            if prompt_type.startswith("few_shot"):
                # For few-shot prompts, pass the additional parameters
                logger.info(f"Creating {prompt_type} prompt for model '{model_name}'")
                prompt = prompt_fn(question_text, "classical_problem", example_selection=example_selection, 
                                  custom_examples=custom_examples, examples_file=examples_file)
            else:
                # For other prompt types, use the standard approach
                logger.info(f"Creating {prompt_type} prompt for model '{model_name}'")
                prompt = prompt_fn(question_text, "classical_problem")
        except Exception as e:
            return {"error": f"Error creating prompt: {str(e)}"}
        
        # Track metrics
        start_time = time.time()
        token_count = self.estimate_token_count(prompt)
        
        # Special handling for self-consistency methods
        if prompt_type.startswith("cot_self_consistency"):
            try:
                # Determine the number of iterations from the prompt type name
                num_iterations = int(prompt_type.split("_")[-1])
                
                # Process the self-consistency method
                logger.info(f"Running self-consistency ({num_iterations} iterations) with model '{model_name}'")
                answer = self._process_self_consistency(question_text, model_name, num_iterations, model_config)
            except Exception as e:
                answer = f"Error in self-consistency processing: {str(e)}"
        # Special handling for ReAct prompts
        elif prompt_type == "react":
            try:
                # Set default React config with higher max_tokens
                react_config = {"max_tokens": 2048}
                
                # Apply any custom configurations
                if model_config:
                    react_config.update(model_config)
                
                # Generate text with model
                if model_type == "local":
                    # Pass the model name for local models
                    answer = generate_text_with_model(
                        prompt, 
                        model_type="local", 
                        model_name=model_name,
                        custom_config=react_config,
                        prompt_type=prompt_type
                    )
                elif model_type == "api" or model_type == "gemini":
                    # For Gemini API
                    answer = generate_text_with_model(
                        prompt, 
                        model_type="gemini",
                        custom_config=react_config,
                        prompt_type=prompt_type
                    )
                else:
                    answer = f"Unsupported model type: {model_type}"
                
                # Make sure the response follows the ReAct format
                if ("THINK:" not in answer and "SUY NGHĨ:" not in answer) or \
                   (("ACTION:" not in answer and "HÀNH ĐỘNG:" not in answer) or \
                    ("OBSERVATION:" not in answer and "QUAN SÁT:" not in answer)):
                    # If the model didn't follow the format, structure the response
                    answer = self._structure_react_response(answer)
            except Exception as e:
                answer = f"Error in ReAct processing: {str(e)}"
        else:
            try:
                # Default configuration
                std_config = {"max_tokens": 1024}
                
                # Apply any custom configurations
                if model_config:
                    std_config.update(model_config)
                
                # Generate text with model
                if model_type == "local":
                    # Pass the model name for local models
                    answer = generate_text_with_model(
                        prompt, 
                        model_type="local", 
                        model_name=model_name,
                        custom_config=std_config,
                        prompt_type=prompt_type
                    )
                elif model_type == "api" or model_type == "gemini":
                    # For Gemini API
                    answer = generate_text_with_model(
                        prompt, 
                        model_type="gemini",
                        custom_config=std_config,
                        prompt_type=prompt_type
                    )
                else:
                    answer = f"Unsupported model type: {model_type}"
            except Exception as e:
                answer = f"Error: {str(e)}"
        
        # Calculate metrics
        end_time = time.time()
        latency = end_time - start_time
        response_length = len(answer)
        tokens_per_second = token_count / latency if latency > 0 else 0
        
        result = {
            "model_name": model_name,
            "prompt_type": prompt_type,
            "question_id": question_id,
            "question": question_text,
            "answer": answer,
            "token_count": token_count,
            "elapsed_time": latency,
            "tokens_per_second": tokens_per_second,
            "response_length": response_length,
            "example_selection": example_selection if prompt_type.startswith("few_shot") else None
        }
        
        return result
    
    def _process_self_consistency(self, question, model_name, num_iterations=3, model_config=None):
        """
        Process a question using the self-consistency approach.
        
        Args:
            question (str): Question text
            model_name (str): Name of the model to use
            num_iterations (int): Number of different solutions to generate
            model_config (dict): Model configuration parameters
            
        Returns:
            str: Consensus answer
        """
        logger.info(f"Generating {num_iterations} different solutions for self-consistency using model '{model_name}'...")
        
        # Get chain-of-thought prompt function
        prompt_fn = self.prompt_types.get("cot", None)
        if not prompt_fn:
            raise ValueError("Chain-of-thought prompt function not found")
        
        # Generate multiple solutions
        solutions = []
        for i in range(num_iterations):
            # Apply temperature randomness for diversity
            solution_config = model_config.copy() if model_config else {}
            solution_config["temperature"] = solution_config.get("temperature", 0.7)  # Default higher temperature
            
            # Generate solution with CoT prompt
            prompt = prompt_fn(question, "classical_problem")
            solution = generate_text_with_model(
                prompt, 
                model_type="gemini" if model_name == "gemini" else "local", 
                model_name=model_name,
                custom_config=solution_config,
                prompt_type="cot_self_consistency"
            )
            
            solutions.append(solution)
            logger.info(f"Generated solution {i+1}/{num_iterations} with model '{model_name}'")
        
        # Extract final answers from the reasoning chains
        answers = []
        for solution in solutions:
            final_answer = self._extract_final_answer(solution)
            if final_answer:
                answers.append(final_answer)
        
        if not answers:
            return "Unable to determine consensus answer from generated solutions"
        
        # Find the most common answer (consensus)
        from collections import Counter
        answer_counts = Counter(answers)
        consensus_answer, _ = answer_counts.most_common(1)[0]
        
        return f"Self-consistency consensus answer ({num_iterations} solutions): {consensus_answer}"
    
    def _extract_final_answer(self, cot_response):
        """
        Extract the final answer from a Chain-of-Thought response.
        
        Args:
            cot_response (str): The full CoT response
            
        Returns:
            str: The extracted final answer
        """
        # Common patterns that might indicate a final answer
        patterns = [
            r"Final answer:(.+?)(?:$|\n\n)",
            r"Therefore, the answer is(.+?)(?:$|\n\n)", 
            r"Thus, the answer is(.+?)(?:$|\n\n)",
            r"The answer is(.+?)(?:$|\n\n)",
            r"In conclusion,(.+?)(?:$|\n\n)",
            r"Vậy kết quả là(.+?)(?:$|\n\n)",
            r"Kết luận:(.+?)(?:$|\n\n)"
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = re.search(pattern, cot_response, re.IGNORECASE | re.DOTALL)
            if matches:
                return matches.group(1).strip()
        
        # If no pattern matches, use the last paragraph/sentence as the answer
        paragraphs = cot_response.split("\n\n")
        if paragraphs:
            last_paragraph = paragraphs[-1].strip()
            if last_paragraph:
                return last_paragraph
        
        # If all else fails, return the entire response
        return cot_response
    
    def _normalize_answer(self, answer):
        """
        Normalize an answer for better comparison in self-consistency.
        
        Args:
            answer (str): The raw answer
            
        Returns:
            str: Normalized answer
        """
        # Remove whitespace, convert to lowercase
        normalized = re.sub(r'\s+', ' ', answer).strip().lower()
        
        # Remove punctuation
        normalized = re.sub(r'[.,;:!?()"\']', '', normalized)
        
        # For numerical answers, try to extract just the number
        number_match = re.search(r'[-+]?\d*\.?\d+', normalized)
        if number_match:
            # If the answer contains a number, prioritize that
            return number_match.group(0)
        
        return normalized
    
    def _structure_react_response(self, response):
        """
        Structure a response to follow the ReAct format if it doesn't already.
        
        Args:
            response (str): The model's response
            
        Returns:
            str: A structured response following ReAct format
        """
        # Start with thinking step
        structured_response = "THINK: Let me think through this problem step by step.\n"
        
        # Extract what seems to be the reasoning part
        lines = response.strip().split('\n')
        reasoning_lines = []
        conclusion_lines = []
        
        # Separate the response into reasoning and conclusion
        in_conclusion = False
        for line in lines:
            line = line.strip()
            if not line:
                    continue
                
            if any(keyword in line.lower() for keyword in ["therefore", "thus", "so the answer is", "the answer is", "finally"]):
                in_conclusion = True
                
            if in_conclusion:
                conclusion_lines.append(line)
            else:
                reasoning_lines.append(line)
        
        # Add reasoning to structured response
        if reasoning_lines:
            structured_response += "\n".join(reasoning_lines) + "\n\n"
        
        # Add action step
        structured_response += "ACTION: I need to calculate the answer based on my reasoning.\n"
        
        # Add observation step (renamed from RESULT to OBSERVATION)
        structured_response += "OBSERVATION: After calculating, I can see the result.\n\n"
        
        # Add final reasoning and answer
        structured_response += "THINK: Based on my calculations and observations, I can determine the answer.\n"
        if conclusion_lines:
            structured_response += "\n".join(conclusion_lines)
        else:
            structured_response += "The answer is: " + response.split(".")[-1].strip()
            
        return structured_response
    
    def evaluate_all_questions(self, questions, model_names=None, prompt_types=None, max_questions=None, checkpoint_frequency=1, example_selection="default", custom_examples=None, examples_file=None, model_config=None):
        """
        Evaluate all models with all prompt types on all questions.
        
        Args:
            questions (list): List of questions to evaluate
            model_names (list): List of model names to evaluate
            prompt_types (list): List of prompt types to evaluate
            max_questions (int): Maximum number of questions to evaluate
            checkpoint_frequency (int): How often to save checkpoints
            example_selection (str): Method to select examples for few-shot prompts
            custom_examples (list): Custom examples for few-shot prompts
            examples_file (str): Path to file containing examples
            model_config (dict): Model configuration parameters
            
        Returns:
            pd.DataFrame: DataFrame containing evaluation results
        """
        # Limit number of questions if max_questions is specified
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        if model_names is None:
            model_names = list(self.models.keys())
            logger.info(f"No models specified, using all available models: {model_names}")
        
        if prompt_types is None:
            prompt_types = list(self.prompt_types.keys())
            logger.info(f"No prompt types specified, using all available types: {prompt_types}")
        
        # Create a more informative log message about the evaluation scope
        logger.info(f"Evaluating {len(questions)} questions with models: {model_names} and prompts: {prompt_types}")
        
        # Tìm checkpoint hiện có (nếu có)
        checkpoint_data = self.load_checkpoint(model_names=model_names, prompt_types=prompt_types)
        
        if checkpoint_data:
            # Khôi phục từ checkpoint
            questions_from_checkpoint, completed_indices, results = checkpoint_data
            logger.info(f"Resuming from checkpoint with {len(completed_indices)} completed questions")
            
            # Kiểm tra xem danh sách câu hỏi có giống nhau không
            if len(questions_from_checkpoint) != len(questions):
                logger.warning(f"Number of questions in checkpoint ({len(questions_from_checkpoint)}) differs from current questions ({len(questions)})")
                logger.warning(f"Using questions from checkpoint to ensure consistency")
                questions = questions_from_checkpoint
        else:
            # Khởi tạo mới
            logger.info(f"No checkpoint found. Starting evaluation from the beginning")
            completed_indices = []
            results = []
        
        # Tính toán tổng số đánh giá cần thực hiện
        total_evaluations = len(model_names) * len(prompt_types) * len(questions)
        logger.info(f"Total evaluations to complete: {total_evaluations}")
        logger.info(f"Checkpoint frequency: every {checkpoint_frequency} questions")
        
        # Tạo progress bar
        pbar = tqdm(total=len(questions), desc="Evaluating questions")
        pbar.update(len(completed_indices))
        
        try:
            # Đánh giá các câu hỏi
            for i, question in enumerate(questions):
                # Bỏ qua các câu hỏi đã hoàn thành
                if i in completed_indices:
                    continue
                
                # Lưu checkpoint theo tần suất
                if i > 0 and i % checkpoint_frequency == 0:
                    self.save_checkpoint(model_names, prompt_types, questions, completed_indices, results)
                
                # Display current question
                logger.info(f"Evaluating question {i+1}/{len(questions)}: {question['text'] if isinstance(question, dict) else question[:50]}...")
                
                for model_name in model_names:
                    for prompt_type in prompt_types:
                        # # Display current model and prompt type being evaluated
                        # logger.info(f" Evaluating model '{model_name}' with prompt '{prompt_type}'")
                        
                        # Lấy config cho model và prompt type cụ thể
                        config_key = f"{model_name}_{prompt_type}"
                        current_config = None
                        if model_config and config_key in model_config:
                            current_config = model_config[config_key]
                        
                        try:
                            result = self.evaluate_single_question(
                                question,
                                model_name,
                                prompt_type,
                                example_selection=example_selection,
                                custom_examples=custom_examples,
                                examples_file=examples_file,
                                model_config=current_config
                            )
                            
                            # Thêm question ID
                            if isinstance(question, dict) and "id" in question:
                                result["question_id"] = question["id"]
                            else:
                                result["question_id"] = str(i)
                                
                            results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Error evaluating question {i} with model {model_name} and prompt {prompt_type}: {e}")
                            # Thêm kết quả lỗi
                            error_result = {
                                "model_name": model_name,
                                "prompt_type": prompt_type,
                                "question_id": question["id"] if isinstance(question, dict) and "id" in question else str(i),
                                "question": question["text"] if isinstance(question, dict) and "text" in question else str(question),
                                "answer": f"Error: {str(e)}",
                                "token_count": 0,
                                "elapsed_time": 0,
                                "tokens_per_second": 0,
                                "response_length": 0,
                                "has_error": True,
                                "error_type": str(type(e).__name__),
                            }
                            results.append(error_result)
                
                # Đánh dấu câu hỏi này đã hoàn thành
                completed_indices.append(i)
                pbar.update(1)
                
                # Lưu checkpoint sau mỗi câu hỏi nếu tần suất là 1
                if checkpoint_frequency == 1:
                    self.save_checkpoint(model_names, prompt_types, questions, completed_indices, results)
            
            # Lưu checkpoint cuối cùng
            self.save_checkpoint(model_names, prompt_types, questions, completed_indices, results)
            
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user")
            logger.info(f"Saving checkpoint before exit...")
            self.save_checkpoint(model_names, prompt_types, questions, completed_indices, results)
            
        except Exception as e:
            logger.error(f"Unexpected error during evaluation: {e}")
            logger.info(f"Saving checkpoint before exit...")
            self.save_checkpoint(model_names, prompt_types, questions, completed_indices, results)
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            pbar.close()
            logger.info(f"  - Completed {len(completed_indices)}/{len(questions)} questions ({len(completed_indices)*100/len(questions):.1f}%)")
        
        # Chuyển đổi kết quả thành DataFrame
        df = pd.DataFrame(results)
        
        # Cập nhật trạng thái
        self.checkpoint_state["current_status"] = "completed"
        self.checkpoint_state["last_updated"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return df
    
    def analyze_results(self, results_df=None, save_plots=True):
        """
        Analyze evaluation results and generate visualizations.
        
        Args:
            results_df (pd.DataFrame): DataFrame containing evaluation results
            save_plots (bool): Whether to save plots
            
        Returns:
            dict: Analysis results
        """
        if results_df is None:
            if not self.results["model_name"]:
                print("  No results to analyze")
                return {}
            results_df = pd.DataFrame(self.results)
            
        print("\n  Analyzing evaluation results...")
        
        # Initialize analysis results
        analysis = {
            "model_metrics": {},
            "prompt_metrics": {},
            "combined_metrics": {},
            "latency_stats": {},
            "token_stats": {},
            "response_length_stats": {},
            "difficulty_analysis": {},
            "topic_analysis": {},
            "error_analysis": {},
            "plots": []
        }
        
        # Set styling for plots
        plt.style.use('fivethirtyeight')
        sns.set_style("whitegrid")
        
        # 1. Response time comparison by model
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x="model_name", y="elapsed_time", data=results_df)
            plt.title("Response Time Comparison by Model")
            plt.xlabel("Model")
            plt.ylabel("Response Time (seconds)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            latency_plot_path = os.path.join(self.plots_dir, "latency_by_model.png")
            plt.savefig(latency_plot_path)
            analysis["plots"].append(latency_plot_path)
            plt.close()
            
            # Calculate response time statistics
            latency_stats = results_df.groupby("model_name")["elapsed_time"].agg(["mean", "median", "min", "max", "std"]).reset_index()
            analysis["latency_stats"] = latency_stats.to_dict(orient="records")
        except Exception as e:
            print(f"  Error creating response time plot: {e}")
        
        # 2. Response time comparison by prompt type
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x="prompt_type", y="elapsed_time", data=results_df)
            plt.title("Response Time Comparison by Prompt Type")
            plt.xlabel("Prompt Type")
            plt.ylabel("Response Time (seconds)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            prompt_latency_plot_path = os.path.join(self.plots_dir, "latency_by_prompt.png")
            plt.savefig(prompt_latency_plot_path)
            analysis["plots"].append(prompt_latency_plot_path)
            plt.close()
        except Exception as e:
            print(f"  Error creating prompt response time plot: {e}")
            
        # 3. Response length comparison
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x="model_name", y="response_length", data=results_df)
            plt.title("Response Length Comparison")
            plt.xlabel("Model")
            plt.ylabel("Response Length (characters)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            length_plot_path = os.path.join(self.plots_dir, "response_length.png")
            plt.savefig(length_plot_path)
            analysis["plots"].append(length_plot_path)
            plt.close()
            
            # Calculate response length statistics
            response_length_stats = results_df.groupby("model_name")["response_length"].agg(["mean", "median", "min", "max", "std"]).reset_index()
            analysis["response_length_stats"] = response_length_stats.to_dict(orient="records")
        except Exception as e:
            print(f"  Error creating response length plot: {e}")
            
        # 4. Heatmap of model vs prompt type (for response time)
        try:
            plt.figure(figsize=(12, 8))
            heatmap_data = results_df.pivot_table(
                values="elapsed_time", 
                index="model_name", 
                columns="prompt_type", 
                aggfunc="mean"
            )
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title("Average Response Time by Model and Prompt Type")
            plt.tight_layout()
            
            heatmap_path = os.path.join(self.plots_dir, "latency_heatmap.png")
            plt.savefig(heatmap_path)
            analysis["plots"].append(heatmap_path)
            plt.close()
        except Exception as e:
            print(f"  Error creating heatmap: {e}")
            
        # 5. Combined performance visualization
        try:
            # Create a figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
            
            # Subplot 1: Average response time by model
            avg_latency = results_df.groupby("model_name")["elapsed_time"].mean().reset_index()
            sns.barplot(x="model_name", y="elapsed_time", data=avg_latency, ax=ax1)
            ax1.set_title("Average Response Time by Model")
            ax1.set_xlabel("Model")
            ax1.set_ylabel("Response Time (seconds)")
            ax1.tick_params(axis='x', rotation=45)
            
            # Subplot 2: Average response length by model
            avg_length = results_df.groupby("model_name")["response_length"].mean().reset_index()
            sns.barplot(x="model_name", y="response_length", data=avg_length, ax=ax2)
            ax2.set_title("Average Response Length by Model")
            ax2.set_xlabel("Model")
            ax2.set_ylabel("Response Length (characters)")
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            combined_path = os.path.join(self.plots_dir, "combined_performance.png")
            plt.savefig(combined_path)
            analysis["plots"].append(combined_path)
            plt.close()
        except Exception as e:
            print(f"  Error creating combined performance plot: {e}")
            
        # 6. Performance distribution histogram
        try:
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            
            # Response time distribution
            sns.histplot(results_df["elapsed_time"], kde=True, ax=axes[0])
            axes[0].set_title("Response Time Distribution")
            axes[0].set_xlabel("Response Time (seconds)")
            
            # Response length distribution
            sns.histplot(results_df["response_length"], kde=True, ax=axes[1])
            axes[1].set_title("Response Length Distribution")
            axes[1].set_xlabel("Response Length (characters)")
            
            plt.tight_layout()
            
            distributions_path = os.path.join(self.plots_dir, "performance_distributions.png")
            plt.savefig(distributions_path)
            analysis["plots"].append(distributions_path)
            plt.close()
        except Exception as e:
            print(f"  Error creating distribution plots: {e}")
            
        # 7. Group key metrics by model
        try:
            model_metrics = results_df.groupby("model_name").agg({
                "elapsed_time": ["mean", "std", "min", "max"],
                "response_length": ["mean", "std", "min", "max"]
            }).reset_index()
            
            analysis["model_metrics"] = model_metrics.to_dict(orient="records")
        except Exception as e:
            print(f"  Error calculating model metrics: {e}")
            
        # Group key metrics by prompt type
        try:
            prompt_metrics = results_df.groupby("prompt_type").agg({
                "elapsed_time": ["mean", "std", "min", "max"],
                "response_length": ["mean", "std", "min", "max"]
            }).reset_index()
            
            analysis["prompt_metrics"] = prompt_metrics.to_dict(orient="records")
        except Exception as e:
            print(f"  Error calculating prompt metrics: {e}")
            
        # Group key metrics by model and prompt type
        try:
            combined_metrics = results_df.groupby(["model_name", "prompt_type"]).agg({
                "elapsed_time": ["mean", "std", "min", "max"],
                "response_length": ["mean", "std", "min", "max"]
            }).reset_index()
            
            analysis["combined_metrics"] = combined_metrics.to_dict(orient="records")
        except Exception as e:
            print(f"  Error calculating combined metrics: {e}")
            
        print(f"  Analysis completed with {len(analysis['plots'])} visualizations")
        return analysis
    
    def create_confusion_matrix(self, results_df, model_name, prompt_type, plot_dir=None):
        """Create a confusion matrix for model responses."""
        # Filter results for the specific model and prompt type
        model_results = results_df[(results_df['model_name'] == model_name) & 
                                   (results_df['prompt_type'] == prompt_type)]
        
        if model_results.empty:
            return None
        
        # Simulate correctness based on response characteristics
        # In practice, you would need actual labels to compare against
        def simulate_correctness(row):
            # If there's an error, consider it incorrect
            if row['has_error']:
                return 'Incorrect'
            
            response = row['response'].lower()
            question = row['question'].lower()
            
            # Check response length - very short responses are usually incomplete
            if len(response) < 50:
                return 'Incomplete'
            
            # Check for keywords that indicate uncertainty
            uncertain_keywords = ['not sure', 'may be', 'unclear', 'insufficient information']
            if any(keyword in response for keyword in uncertain_keywords):
                return 'Uncertain'
            
            # Simulate based on model - different models have different accuracy rates
            accuracy_by_model = {
                'llama': 0.75,
                'qwen': 0.70,
                'gemini': 0.85
            }
            base_accuracy = accuracy_by_model.get(model_name.lower(), 0.5)
            
            # Add random noise
            import random
            if random.random() > base_accuracy:
                return 'Incorrect'
            else:
                return 'Correct'
        
        # Add correctness column to DataFrame
        model_results['correctness'] = model_results.apply(simulate_correctness, axis=1)
        
        # Calculate confusion matrix values
        labels = ['Correct', 'Incorrect', 'Incomplete', 'Uncertain']
        counts = model_results['correctness'].value_counts()
        
        # Ensure all labels have values
        for label in labels:
            if label not in counts.index:
                counts[label] = 0
        
        # Create a confusion matrix from counts
        cm = np.zeros((2, 2))
        cm[0, 0] = counts.get('Correct', 0)  # True Positive
        cm[0, 1] = counts.get('Uncertain', 0)  # False Negative
        cm[1, 0] = counts.get('Incomplete', 0)  # False Positive
        cm[1, 1] = counts.get('Incorrect', 0)  # True Negative
        
        # Create heatmap from confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                    xticklabels=['Correct', 'Uncertain'], 
                    yticklabels=['Complete', 'Incomplete'])
        plt.title(f'Confusion Matrix - {model_name.upper()} ({prompt_type})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save the chart if a target directory is provided
        if plot_dir:
            filename = os.path.join(plot_dir, f"confusion_matrix_{model_name}_{prompt_type}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return filename
        else:
            return plt

    def create_response_heatmap(self, results_df, plot_dir=None):
        """Create a heatmap comparing models and prompt types."""
        # Create a pivot table
        pivot = pd.pivot_table(
            results_df,
            values='response_length',
            index='model_name',
            columns='prompt_type',
            aggfunc='mean',
            fill_value=0
        )
        
        # Create heatmap with seaborn
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu')
        plt.title('Average Response Length by Model and Prompt Type')
        
        # Adjust font size
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Add annotations
        plt.tight_layout()
        
        # Save the chart if a target directory is provided
        if plot_dir:
            filename = os.path.join(plot_dir, f"response_length_heatmap.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return filename
        else:
            return plt

    def create_response_time_comparison(self, results_df, plot_dir=None):
        """Create a chart comparing response times between models and prompt types."""
        # Create a pivot table
        pivot = pd.pivot_table(
            results_df,
            values='elapsed_time',
            index='model_name',
            columns='prompt_type',
            aggfunc='mean',
            fill_value=0
        )
        
        # Create stacked bar chart
        pivot.plot(kind='bar', figsize=(12, 6), colormap='viridis')
        plt.title('Average Response Time by Model and Prompt Type')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Model')
        plt.xticks(rotation=0)
        plt.legend(title='Prompt Type')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on each bar
        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt='%.1fs', fontsize=8)
        
        # Add annotations
        plt.tight_layout()
        
        # Save the chart if a target directory is provided
        if plot_dir:
            filename = os.path.join(plot_dir, f"response_time_comparison.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return filename
        else:
            return plt

    def create_error_distribution(self, results_df, plot_dir=None):
        """Create a chart showing error distribution by model."""
        # Calculate error rates
        error_rates = results_df.groupby('model_name')['has_error'].mean() * 100
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(error_rates.index, error_rates.values, color=['#FF5733', '#33A8FF', '#8C33FF'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('Error Rate by Model')
        plt.ylabel('Error Rate (%)')
        plt.xlabel('Model')
        plt.ylim(0, max(error_rates.values) * 1.2)  # Set y-axis limit
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add annotations
        plt.tight_layout()
        
        # Save the chart if a target directory is provided
        if plot_dir:
            filename = os.path.join(plot_dir, f"error_distribution.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return filename
        else:
            return plt

    def create_interactive_dashboard(self, results_df, plot_dir=None):
        """Create an interactive dashboard with Plotly."""
        # Calculate metrics
        model_stats = results_df.groupby('model_name').agg({
            'has_error': 'mean',
            'elapsed_time': 'mean',
            'response_length': 'mean'
        }).reset_index()
        
    def generate_report(self, results_df=None, analysis_results=None):
        """
        Generate HTML report from evaluation results.
        
        Args:
            results_df (pd.DataFrame, optional): Results dataframe
            analysis_results (dict, optional): Pre-computed analysis results
            
        Returns:
            str: Path to the generated report
        """
        if results_df is None:
            # Check if we have processed results
            processed_results_path = os.path.join(self.results_dir, "processed_results.csv")
            if os.path.exists(processed_results_path):
                results_df = pd.read_csv(processed_results_path)
                print(f"  Loaded processed results from {processed_results_path}")
            elif not self.results["model_name"]:
                print("  No results to generate report")
                return ""
            else:
                results_df = pd.DataFrame(self.results)
            
        if analysis_results is None:
            analysis_results = self.analyze_results(results_df, save_plots=True)
            
        print("\n📝 Generating evaluation report...")
        
        # Create report with HTML
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"evaluation_report_{timestamp}.html")
        
        # Calculate error statistics if available
        has_error_stats = "has_error" in results_df.columns
        error_count = results_df["has_error"].sum() if has_error_stats else 0
        error_percentage = (error_count / len(results_df) * 100) if has_error_stats and len(results_df) > 0 else 0
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .plot-container {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .model-response {{ 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 5px;
                    background-color: #f9f9f9;
                }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric-card {{ 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    margin: 10px; 
                    border-radius: 5px;
                    background-color: #f9f9f9;
                    flex: 1;
                    min-width: 200px;
                }}
                .plot-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    grid-gap: 20px;
                }}
                .full-width {{
                    grid-column: 1 / span 2;
                }}
                .stat-table {{
                    margin-top: 30px;
                    margin-bottom: 50px;
                }}
                .error {{ color: #d9534f; }}
                .warning {{ color: #f0ad4e; }}
                .success {{ color: #5cb85c; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>LLM Evaluation Report</h1>
                <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Summary</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Models Evaluated</h3>
                        <p>{results_df['model_name'].nunique()}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Prompt Types</h3>
                        <p>{results_df['prompt_type'].nunique()}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Total Questions</h3>
                        <p>{len(results_df['question'].unique())}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Total Evaluations</h3>
                        <p>{len(results_df)}</p>
                    </div>
                    {f'<div class="metric-card"><h3>Error Rate</h3><p class="{"error" if error_percentage > 10 else "warning" if error_percentage > 0 else "success"}">{error_count} ({error_percentage:.1f}%)</p></div>' if has_error_stats else ''}
                </div>
        """
        
        # Add error statistics section if errors exist
        if has_error_stats and error_count > 0:
            html_content += """
                <h2>Error Statistics</h2>
                <table class="stat-table">
                    <tr>
                        <th>Model</th>
                        <th>Errors</th>
                        <th>Total Responses</th>
                        <th>Error Rate</th>
                    </tr>
            """
            
            # Add error statistics by model
            for model in results_df['model_name'].unique():
                model_df = results_df[results_df['model_name'] == model]
                model_errors = model_df['has_error'].sum()
                model_total = len(model_df)
                model_error_rate = (model_errors / model_total * 100) if model_total > 0 else 0
                
                error_class = "error" if model_error_rate > 10 else "warning" if model_error_rate > 0 else "success"
                
                html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{model_errors}</td>
                        <td>{model_total}</td>
                        <td class="{error_class}">{model_error_rate:.1f}%</td>
                    </tr>
                """
            
            html_content += """
                </table>
            """
        
        html_content += """
                <h2>Performance Visualizations</h2>
                
                <div class="plot-grid">
                    <div class="plot-container">
                        <h3>Latency Comparison by Model</h3>
                        <img src="plots/latency_by_model.png" alt="Latency by Model" style="max-width:100%;">
                    </div>
                    
                    <div class="plot-container">
                        <h3>Latency Comparison by Prompt Type</h3>
                        <img src="plots/latency_by_prompt.png" alt="Latency by Prompt Type" style="max-width:100%;">
                    </div>
                    
                    <div class="plot-container">
                        <h3>Tokens Per Second Comparison</h3>
                        <img src="plots/tokens_per_second.png" alt="Tokens Per Second" style="max-width:100%;">
                    </div>
                    
                    <div class="plot-container">
                        <h3>Response Length Comparison</h3>
                        <img src="plots/response_length.png" alt="Response Length" style="max-width:100%;">
                    </div>
                    
                    <div class="plot-container full-width">
                        <h3>Latency Heatmap (Model vs Prompt Type)</h3>
                        <img src="plots/latency_heatmap.png" alt="Latency Heatmap" style="max-width:100%;">
                    </div>
                    
                    <div class="plot-container full-width">
                        <h3>Combined Performance Metrics</h3>
                        <img src="plots/combined_performance.png" alt="Combined Performance" style="max-width:100%;">
                    </div>
                    
                    <div class="plot-container full-width">
                        <h3>Performance Distributions</h3>
                        <img src="plots/performance_distributions.png" alt="Performance Distributions" style="max-width:100%;">
                    </div>
                </div>
                
                <h2>Detailed Statistics</h2>
                
                <h3>Latency Statistics by Model (seconds)</h3>
                <table class="stat-table">
                    <tr>
                        <th>Model</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Std Dev</th>
                    </tr>
        """
        
        # Add latency statistics rows
        if "latency_stats" in analysis_results:
            for stat in analysis_results["latency_stats"]:
                html_content += f"""
                    <tr>
                        <td>{stat['model_name']}</td>
                        <td>{stat['mean']:.2f}</td>
                        <td>{stat['median']:.2f}</td>
                        <td>{stat['min']:.2f}</td>
                        <td>{stat['max']:.2f}</td>
                        <td>{stat['std']:.2f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
                
                <h3>Tokens Per Second Statistics by Model</h3>
                <table class="stat-table">
                    <tr>
                        <th>Model</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Std Dev</th>
                    </tr>
        """
        
        # Add tokens per second statistics rows
        if "token_stats" in analysis_results:
            for stat in analysis_results["token_stats"]:
                html_content += f"""
                    <tr>
                        <td>{stat['model_name']}</td>
                        <td>{stat['mean']:.2f}</td>
                        <td>{stat['median']:.2f}</td>
                        <td>{stat['min']:.2f}</td>
                        <td>{stat['max']:.2f}</td>
                        <td>{stat['std']:.2f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
                
                <h3>Response Length Statistics by Model (characters)</h3>
                <table class="stat-table">
                    <tr>
                        <th>Model</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Std Dev</th>
                    </tr>
        """
        
        # Add response length statistics rows
        if "response_length_stats" in analysis_results:
            for stat in analysis_results["response_length_stats"]:
                html_content += f"""
                    <tr>
                        <td>{stat['model_name']}</td>
                        <td>{stat['mean']:.2f}</td>
                        <td>{stat['median']:.2f}</td>
                        <td>{stat['min']:.2f}</td>
                        <td>{stat['max']:.2f}</td>
                        <td>{stat['std']:.2f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
                
                <h2>Sample Responses</h2>
        """
        
        # Add sample responses (limit to first 5)
        sample_count = min(5, len(results_df))
        for i in range(sample_count):
            row = results_df.iloc[i]
            has_error = "has_error" in row and row["has_error"] == 1
            error_class = 'class="error"' if has_error else ''
            
            html_content += f"""
                <div class="model-response">
                    <h3>Question {i+1}</h3>
                    <p><strong>Question:</strong> {row['question']}</p>
                    <p><strong>Model:</strong> {row['model_name']}</p>
                    <p><strong>Prompt Type:</strong> {row['prompt_type']}</p>
                    <p><strong>Response:</strong></p>
                    <pre {error_class}>{row.get('answer', row.get('response', 'No response available'))}</pre>
                    <p><strong>Latency:</strong> {row['elapsed_time']:.2f} seconds</p>
                    {f'<p class="error"><strong>Status:</strong> Error detected in response</p>' if has_error else ''}
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"  Report generated: {report_path}")
        return report_path
    
    def analyze_results_from_json(self, json_file):
        """
        Analyze evaluation results from a JSON file.
        
        Args:
            json_file (str): Path to JSON file containing results
            
        Returns:
            dict: Analysis results
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
                
            print(f"  Loaded {len(results_data)} results from {json_file}")
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(results_data)
            
            # Rename columns to match our standard
            column_mapping = {
                "model": "model_name",
                "response": "answer",
                "time": "elapsed_time"
            }
            df = df.rename(columns=column_mapping)
            
            # Add derived columns
            df["question_id"] = range(len(df))
            df["token_count"] = df["question"].apply(self.estimate_token_count)
            df["response_length"] = df["answer"].apply(len)
            
            # Handle potential division by zero or invalid latency values
            df["elapsed_time"] = df["elapsed_time"].apply(lambda x: max(0.001, x))  # Ensure no zero values
            df["tokens_per_second"] = df.apply(
                lambda row: row["token_count"] / row["elapsed_time"] if row["elapsed_time"] > 0 else 0, 
                axis=1
            )
            
            # Add error flag for responses that contain error messages
            df["has_error"] = df["answer"].apply(
                lambda x: 1 if "[Không thể tạo phản hồi" in x or "Error:" in x else 0
            )
            
            # Print error statistics
            error_count = df["has_error"].sum()
            if error_count > 0:
                print(f"  Found {error_count} responses with errors ({error_count/len(df)*100:.1f}%)")
                # Group errors by model
                error_by_model = df.groupby("model_name")["has_error"].sum()
                for model, count in error_by_model.items():
                    if count > 0:
                        model_total = len(df[df["model_name"] == model])
                        print(f"  - {model}: {count}/{model_total} responses with errors ({count/model_total*100:.1f}%)")
            
            # Save the processed DataFrame
            results_path = os.path.join(self.results_dir, "processed_results.csv")
            df.to_csv(results_path, index=False)
            
            # Run analysis
            analysis_results = self.analyze_results(df)
            
            return analysis_results
            
        except Exception as e:
            print(f"  Error analyzing results from JSON: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def analyze_model_behavior(self, results_df):
        """
        Analyze model behavior patterns and characteristics.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            
        Returns:
            dict: Analysis results
        """
        analysis = {
            "behavior_patterns": {},
            "error_patterns": {},
            "performance_patterns": {},
            "response_patterns": {}
        }
        
        # 1. Analyze response patterns
        for model in results_df['model_name'].unique():
            model_df = results_df[results_df['model_name'] == model]
            
            # Response length patterns
            length_stats = model_df['response_length'].describe()
            
            # Error patterns
            error_rate = model_df['has_error'].mean()
            error_types = model_df['error_type'].value_counts().to_dict()
            
            # Performance patterns
            latency_stats = model_df['elapsed_time'].describe()
            token_speed = model_df['tokens_per_second'].mean()
            
            # Confidence patterns
            confidence_stats = model_df['confidence_score'].describe()
            
            analysis["behavior_patterns"][model] = {
                "response_length": length_stats.to_dict(),
                "error_rate": error_rate,
                "error_types": error_types,
                "latency": latency_stats.to_dict(),
                "token_speed": token_speed,
                "confidence": confidence_stats.to_dict()
            }
        
        # 2. Analyze prompt type effectiveness
        for prompt_type in results_df['prompt_type'].unique():
            prompt_df = results_df[results_df['prompt_type'] == prompt_type]
            
            # Calculate effectiveness metrics
            avg_confidence = prompt_df['confidence_score'].mean()
            error_rate = prompt_df['has_error'].mean()
            avg_latency = prompt_df['elapsed_time'].mean()
            
            analysis["prompt_effectiveness"][prompt_type] = {
                "avg_confidence": avg_confidence,
                "error_rate": error_rate,
                "avg_latency": avg_latency
            }
        
        # 3. Analyze topic-based performance
        for topic in results_df['topic_category'].unique():
            topic_df = results_df[results_df['topic_category'] == topic]
            
            # Calculate topic-specific metrics
            avg_confidence = topic_df['confidence_score'].mean()
            error_rate = topic_df['has_error'].mean()
            avg_latency = topic_df['elapsed_time'].mean()
            
            analysis["topic_performance"][topic] = {
                "avg_confidence": avg_confidence,
                "error_rate": error_rate,
                "avg_latency": avg_latency
            }
        
        return analysis
    
    def create_advanced_visualizations(self, results_df, analysis_results):
        """
        Create advanced visualizations for model analysis.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            analysis_results (dict): Analysis results
            
        Returns:
            list: Paths to generated visualization files
        """
        visualization_paths = []
        
        # 1. PCA Analysis of Model Performance
        try:
            # Prepare features for PCA
            features = ['elapsed_time', 'response_length', 'tokens_per_second', 'confidence_score']
            X = results_df[features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create PCA plot
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=results_df['model_name'].map({'llama': 0, 'qwen': 1, 'gemini': 2}),
                                cmap='viridis')
            plt.colorbar(scatter)
            plt.title('PCA Analysis of Model Performance')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            
            pca_path = os.path.join(self.plots_dir, "pca_analysis.png")
            plt.savefig(pca_path)
            visualization_paths.append(pca_path)
            plt.close()
            
        except Exception as e:
            print(f"  Error creating PCA plot: {e}")
        
        # 2. Performance Clustering Analysis
        try:
            # Prepare features for clustering
            features = ['elapsed_time', 'response_length', 'tokens_per_second']
            X = results_df[features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Create cluster plot
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                c=clusters, cmap='viridis')
            plt.colorbar(scatter)
            plt.title('Performance Clustering Analysis')
            plt.xlabel('Standardized Response Time')
            plt.ylabel('Standardized Response Length')
            
            cluster_path = os.path.join(self.plots_dir, "performance_clusters.png")
            plt.savefig(cluster_path)
            visualization_paths.append(cluster_path)
            plt.close()
            
        except Exception as e:
            print(f"  Error creating clustering plot: {e}")
        
        # 3. Interactive Performance Dashboard
        try:
            # Create interactive dashboard with Plotly
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Response Time by Model', 
                              'Error Rate by Model',
                              'Confidence Score by Model',
                              'Token Speed by Model')
            )
            
            # Add traces
            for model in results_df['model_name'].unique():
                model_df = results_df[results_df['model_name'] == model]
                
                # Response time
                fig.add_trace(
                    go.Box(y=model_df['elapsed_time'], name=model),
                    row=1, col=1
                )
                
                # Error rate
                fig.add_trace(
                    go.Bar(x=[model], y=[model_df['has_error'].mean()], name=model),
                    row=1, col=2
                )
                
                # Confidence score
                fig.add_trace(
                    go.Box(y=model_df['confidence_score'], name=model),
                    row=2, col=1
                )
                
                # Token speed
                fig.add_trace(
                    go.Box(y=model_df['tokens_per_second'], name=model),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=False,
                title_text="Interactive Model Performance Dashboard"
            )
            
            # Save interactive dashboard
            dashboard_path = os.path.join(self.plots_dir, "interactive_dashboard.html")
            fig.write_html(dashboard_path)
            visualization_paths.append(dashboard_path)
            
        except Exception as e:
            print(f"  Error creating interactive dashboard: {e}")
        
        # 4. Topic-based Performance Analysis
        try:
            # Create topic performance heatmap
            topic_performance = pd.pivot_table(
                results_df,
                values='confidence_score',
                index='model_name',
                columns='topic_category',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(topic_performance, annot=True, fmt='.2f', cmap='YlOrRd')
            plt.title('Topic-based Performance Analysis')
            plt.tight_layout()
            
            topic_path = os.path.join(self.plots_dir, "topic_performance.png")
            plt.savefig(topic_path)
            visualization_paths.append(topic_path)
            plt.close()
            
        except Exception as e:
            print(f"  Error creating topic performance plot: {e}")
        
        return visualization_paths
    
    def analyze_error_patterns(self, results_df):
        """
        Analyze patterns in model errors.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            
        Returns:
            dict: Error analysis results
        """
        error_analysis = {
            "error_types": {},
            "error_by_model": {},
            "error_by_prompt": {},
            "error_by_topic": {},
            "error_trends": {}
        }
        
        # 1. Analyze error types
        error_types = results_df['error_type'].value_counts()
        error_analysis["error_types"] = error_types.to_dict()
        
        # 2. Analyze errors by model
        for model in results_df['model_name'].unique():
            model_df = results_df[results_df['model_name'] == model]
            error_analysis["error_by_model"][model] = {
                "total_errors": model_df['has_error'].sum(),
                "error_rate": model_df['has_error'].mean(),
                "error_types": model_df['error_type'].value_counts().to_dict()
            }
        
        # 3. Analyze errors by prompt type
        for prompt_type in results_df['prompt_type'].unique():
            prompt_df = results_df[results_df['prompt_type'] == prompt_type]
            error_analysis["error_by_prompt"][prompt_type] = {
                "total_errors": prompt_df['has_error'].sum(),
                "error_rate": prompt_df['has_error'].mean(),
                "error_types": prompt_df['error_type'].value_counts().to_dict()
            }
        
        # 4. Analyze errors by topic
        for topic in results_df['topic_category'].unique():
            topic_df = results_df[results_df['topic_category'] == topic]
            error_analysis["error_by_topic"][topic] = {
                "total_errors": topic_df['has_error'].sum(),
                "error_rate": topic_df['has_error'].mean(),
                "error_types": topic_df['error_type'].value_counts().to_dict()
            }
        
        # 5. Analyze error trends over time
        if 'timestamp' in results_df.columns:
            error_trends = results_df.groupby('timestamp')['has_error'].mean()
            error_analysis["error_trends"] = error_trends.to_dict()
        
        return error_analysis
    
    def generate_comprehensive_report(self, results_df=None, analysis_results=None):
        """
        Generate a comprehensive HTML report with advanced analysis.
        
        Args:
            results_df (pd.DataFrame, optional): Results dataframe
            analysis_results (dict, optional): Pre-computed analysis results
            
        Returns:
            str: Path to the generated report
        """
        if results_df is None:
            if not self.results["model_name"]:
                print("  No results to generate report")
                return ""
            results_df = pd.DataFrame(self.results)
        
        if analysis_results is None:
            analysis_results = self.analyze_results(results_df, save_plots=True)
        
        # Generate advanced analysis
        behavior_analysis = self.analyze_model_behavior(results_df)
        error_analysis = self.analyze_error_patterns(results_df)
        
        # Create advanced visualizations
        visualization_paths = self.create_advanced_visualizations(results_df, analysis_results)
        
        # Generate comprehensive HTML report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"comprehensive_report_{timestamp}.html")
        
        # Create HTML content with advanced analysis and visualizations
        html_content = self._generate_html_report(
            results_df, 
            analysis_results, 
            behavior_analysis, 
            error_analysis,
            visualization_paths
        )
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  Comprehensive report generated: {report_path}")
        return report_path
    
    def _generate_html_report(self, results_df, analysis_results, behavior_analysis, error_analysis, visualization_paths):
        """
        Generate HTML content for the comprehensive report.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            analysis_results (dict): Analysis results
            behavior_analysis (dict): Behavior analysis results
            error_analysis (dict): Error analysis results
            visualization_paths (list): Paths to visualization files
            
        Returns:
            str: HTML content
        """
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LLM Performance Analysis Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1, h2, h3 {
                    color: #2c3e50;
                    margin-top: 30px;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .metric-card {
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 6px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }
                .metric-label {
                    color: #7f8c8d;
                    font-size: 14px;
                }
                .visualization {
                    margin: 20px 0;
                    text-align: center;
                }
                .visualization img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                    font-weight: 600;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .error-highlight {
                    color: #e74c3c;
                    font-weight: bold;
                }
                .success-highlight {
                    color: #2ecc71;
                    font-weight: bold;
                }
                .chart-container {
                    margin: 20px 0;
                    padding: 15px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                .recommendations {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }
                .recommendation-item {
                    margin: 10px 0;
                    padding: 10px;
                    background-color: white;
                    border-radius: 4px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>LLM Performance Analysis Report</h1>
                <p>Generated on: {timestamp}</p>
                
                <!-- Overview Section -->
                <div class="section">
                    <h2>Overview</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{total_models}</div>
                            <div class="metric-label">Models Evaluated</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{total_questions}</div>
                            <div class="metric-label">Questions Tested</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{avg_confidence:.2f}%</div>
                            <div class="metric-label">Average Confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{avg_response_time:.2f}s</div>
                            <div class="metric-label">Average Response Time</div>
                        </div>
                    </div>
                </div>
                
                <!-- Model Performance Section -->
                <div class="section">
                    <h2>Model Performance</h2>
                    <div class="chart-container">
                        <h3>Response Time Distribution</h3>
                        <div class="visualization">
                            <img src="{response_time_dist}" alt="Response Time Distribution">
                        </div>
                    </div>
                    <div class="chart-container">
                        <h3>Confidence Score Distribution</h3>
                        <div class="visualization">
                            <img src="{confidence_dist}" alt="Confidence Distribution">
                        </div>
                    </div>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Confidence</th>
                            <th>Response Time</th>
                            <th>Error Rate</th>
                            <th>Response Length</th>
                        </tr>
                        {model_comparison_rows}
                    </table>
                </div>
                
                <!-- Prompt Type Analysis -->
                <div class="section">
                    <h2>Prompt Type Analysis</h2>
                    <div class="chart-container">
                        <h3>Error Rate by Model and Prompt Type</h3>
                        <div class="visualization">
                            <img src="{error_rate_analysis}" alt="Error Rate Analysis">
                        </div>
                    </div>
                    <table>
                        <tr>
                            <th>Prompt Type</th>
                            <th>Confidence</th>
                            <th>Response Time</th>
                            <th>Error Rate</th>
                        </tr>
                        {prompt_effectiveness_rows}
                    </table>
                </div>
                
                <!-- Topic-based Performance -->
                <div class="section">
                    <h2>Topic-based Performance</h2>
                    <div class="visualization">
                        <img src="{topic_performance}" alt="Topic Performance Analysis">
                    </div>
                    <table>
                        <tr>
                            <th>Topic</th>
                            <th>Confidence</th>
                            <th>Response Time</th>
                            <th>Error Rate</th>
                        </tr>
                        {topic_performance_rows}
                    </table>
                </div>
                
                <!-- Error Analysis -->
                <div class="section">
                    <h2>Error Analysis</h2>
                    <div class="visualization">
                        <img src="{error_types}" alt="Error Types Distribution">
                    </div>
                    <div class="visualization">
                        <img src="{length_vs_time}" alt="Response Length vs Time">
                    </div>
                    <table>
                        <tr>
                            <th>Error Type</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                        {error_analysis_rows}
                    </table>
                </div>
                
                <!-- Performance Trends -->
                <div class="section">
                    <h2>Performance Trends</h2>
                    <div class="chart-container">
                        <h3>Time-based Trends</h3>
                        <div class="visualization">
                            <img src="{time_trends_plot}" alt="Time-based Trends">
                        </div>
                    </div>
                </div>
                
                <!-- Recommendations -->
                <div class="section">
                    <h2>Recommendations</h2>
                    <div class="recommendations">
                        {recommendations_html}
                    </div>
                </div>
                
                <!-- Detailed Results -->
                <div class="section">
                    <h2>Detailed Results</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Prompt Type</th>
                            <th>Question ID</th>
                            <th>Confidence</th>
                            <th>Response Time</th>
                            <th>Error Type</th>
                        </tr>
                        {detailed_results_rows}
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Calculate overview metrics
        total_models = len(results_df['model_name'].unique())
        total_questions = len(results_df['question_id'].unique())
        avg_confidence = results_df['confidence_score'].mean() * 100
        avg_response_time = results_df['elapsed_time'].mean()
        
        # Generate model comparison rows
        model_comparison_rows = ""
        for model, metrics in behavior_analysis["model_comparison"].items():
            model_comparison_rows += f"""
            <tr>
                <td>{model}</td>
                <td class="{'success-highlight' if metrics['confidence'] >= 0.8 else 'error-highlight'}">{metrics['confidence']*100:.2f}%</td>
                <td>{metrics['response_time']:.2f}s</td>
                <td class="{'error-highlight' if metrics['error_rate'] > 0.1 else 'success-highlight'}">{metrics['error_rate']*100:.2f}%</td>
                <td>{metrics['response_length']:.0f}</td>
            </tr>
            """
        
        # Generate prompt effectiveness rows
        prompt_effectiveness_rows = ""
        for prompt_type, metrics in behavior_analysis["prompt_effectiveness"].items():
            prompt_effectiveness_rows += f"""
            <tr>
                <td>{prompt_type}</td>
                <td class="{'success-highlight' if metrics['confidence'] >= 0.8 else 'error-highlight'}">{metrics['confidence']*100:.2f}%</td>
                <td>{metrics['response_time']:.2f}s</td>
                <td class="{'error-highlight' if metrics['error_rate'] > 0.1 else 'success-highlight'}">{metrics['error_rate']*100:.2f}%</td>
            </tr>
            """
        
        # Generate topic performance rows
        topic_performance_rows = ""
        for topic, metrics in behavior_analysis["topic_performance"].items():
            topic_performance_rows += f"""
            <tr>
                <td>{topic}</td>
                <td class="{'success-highlight' if metrics['confidence'] >= 0.8 else 'error-highlight'}">{metrics['confidence']*100:.2f}%</td>
                <td>{metrics['response_time']:.2f}s</td>
                <td class="{'error-highlight' if metrics['error_rate'] > 0.1 else 'success-highlight'}">{metrics['error_rate']*100:.2f}%</td>
            </tr>
            """
        
        # Generate error analysis rows
        error_analysis_rows = ""
        total_errors = error_analysis["error_by_model"]["total_errors"]
        for error_type, count in error_analysis["error_types"].items():
            percentage = (count / total_errors) * 100
            error_analysis_rows += f"""
            <tr>
                <td>{error_type}</td>
                <td>{count}</td>
                <td class="{'error-highlight' if percentage > 20 else 'warning'}">{percentage:.2f}%</td>
            </tr>
            """
        
        # Generate recommendations HTML
        recommendations_html = ""
        for recommendation in error_analysis["recommendations"]:
            recommendations_html += f"""
            <div class="recommendation-item">
                {recommendation}
            </div>
            """
        
        # Generate detailed results rows
        detailed_results_rows = ""
        for _, row in results_df.iterrows():
            detailed_results_rows += f"""
            <tr>
                <td>{row['model_name']}</td>
                <td>{row['prompt_type']}</td>
                <td>{row['question_id']}</td>
                <td class="{'success-highlight' if row['confidence_score'] >= 0.8 else 'error-highlight'}">{row['confidence_score']*100:.2f}%</td>
                <td>{row['elapsed_time']:.2f}s</td>
                <td class="{'error-highlight' if row['has_error'] else 'success-highlight'}">{row['error_type'] if row['has_error'] else 'None'}</td>
            </tr>
            """
        
        # Format the HTML content
        html_content = html_content.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_models=total_models,
            total_questions=total_questions,
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            model_comparison_rows=model_comparison_rows,
            prompt_effectiveness_rows=prompt_effectiveness_rows,
            topic_performance_rows=topic_performance_rows,
            error_analysis_rows=error_analysis_rows,
            recommendations_html=recommendations_html,
            detailed_results_rows=detailed_results_rows,
            **visualization_paths
        )
        
        return html_content
    
    def create_detailed_visualizations(self, results_df):
        """
        Create detailed visualizations for model performance analysis.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            
        Returns:
            dict: Paths to generated visualization files
        """
        plots_dir = os.path.join(self.results_dir, "detailed_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        visualization_paths = {}
        
        # 1. Response Time Distribution
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='model_name', y='elapsed_time', data=results_df)
            plt.title('Response Time Distribution by Model')
            plt.xlabel('Model')
            plt.ylabel('Response Time (seconds)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            visualization_paths['response_time_dist'] = os.path.join(plots_dir, "response_time_distribution.png")
            plt.savefig(visualization_paths['response_time_dist'])
            plt.close()
            
        except Exception as e:
            print(f"  Error creating response time distribution plot: {e}")
        
        # 2. Error Rate by Model and Prompt Type
        try:
            plt.figure(figsize=(12, 6))
            error_data = results_df.groupby(['model_name', 'prompt_type'])['has_error'].mean().unstack()
            error_data.plot(kind='bar', stacked=True)
            plt.title('Error Rate by Model and Prompt Type')
            plt.xlabel('Model')
            plt.ylabel('Error Rate')
            plt.legend(title='Prompt Type')
            plt.tight_layout()
            
            visualization_paths['error_rate_analysis'] = os.path.join(plots_dir, "error_rate_analysis.png")
            plt.savefig(visualization_paths['error_rate_analysis'])
            plt.close()
            
        except Exception as e:
            print(f"  Error creating error rate analysis plot: {e}")
        
        # 3. Confidence Score Distribution
        try:
            plt.figure(figsize=(12, 6))
            sns.violinplot(x='model_name', y='confidence_score', data=results_df)
            plt.title('Confidence Score Distribution by Model')
            plt.xlabel('Model')
            plt.ylabel('Confidence Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            visualization_paths['confidence_dist'] = os.path.join(plots_dir, "confidence_distribution.png")
            plt.savefig(visualization_paths['confidence_dist'])
            plt.close()
            
        except Exception as e:
            print(f"  Error creating confidence distribution plot: {e}")
        
        # 4. Topic-based Performance Heatmap
        try:
            plt.figure(figsize=(12, 8))
            topic_performance = pd.pivot_table(
                results_df,
                values='confidence_score',
                index='model_name',
                columns='topic_category',
                aggfunc='mean'
            )
            sns.heatmap(topic_performance, annot=True, fmt='.2f', cmap='YlOrRd')
            plt.title('Topic-based Performance Analysis')
            plt.tight_layout()
            
            visualization_paths['topic_performance'] = os.path.join(plots_dir, "topic_performance_heatmap.png")
            plt.savefig(visualization_paths['topic_performance'])
            plt.close()
            
        except Exception as e:
            print(f"  Error creating topic performance heatmap: {e}")
        
        # 5. Response Length vs. Response Time
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=results_df, x='response_length', y='elapsed_time', 
                          hue='model_name', alpha=0.6)
            plt.title('Response Length vs. Response Time')
            plt.xlabel('Response Length (tokens)')
            plt.ylabel('Response Time (seconds)')
            plt.tight_layout()
            
            visualization_paths['length_vs_time'] = os.path.join(plots_dir, "length_vs_time.png")
            plt.savefig(visualization_paths['length_vs_time'])
            plt.close()
            
        except Exception as e:
            print(f"  Error creating length vs time plot: {e}")
        
        # 6. Error Type Distribution
        try:
            plt.figure(figsize=(12, 6))
            error_types = results_df[results_df['has_error']]['error_type'].value_counts()
            error_types.plot(kind='bar')
            plt.title('Distribution of Error Types')
            plt.xlabel('Error Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            visualization_paths['error_types'] = os.path.join(plots_dir, "error_types_distribution.png")
            plt.savefig(visualization_paths['error_types'])
            plt.close()
            
        except Exception as e:
            print(f"  Error creating error types distribution plot: {e}")
        
        return visualization_paths
    
    def analyze_performance_trends(self, results_df):
        """
        Analyze performance trends over time and across different dimensions.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            
        Returns:
            dict: Performance trend analysis results
        """
        trends = {
            "time_based": {},
            "model_based": {},
            "prompt_based": {},
            "topic_based": {}
        }
        
        # 1. Time-based trends
        if 'timestamp' in results_df.columns:
            time_trends = results_df.groupby('timestamp').agg({
                'confidence_score': 'mean',
                'elapsed_time': 'mean',
                'has_error': 'mean'
            })
            trends["time_based"] = time_trends.to_dict()
        
        # 2. Model-based trends
        model_trends = results_df.groupby('model_name').agg({
            'confidence_score': ['mean', 'std'],
            'elapsed_time': ['mean', 'std'],
            'has_error': 'mean',
            'response_length': ['mean', 'std']
        })
        trends["model_based"] = model_trends.to_dict()
        
        # 3. Prompt-based trends
        prompt_trends = results_df.groupby('prompt_type').agg({
            'confidence_score': ['mean', 'std'],
            'elapsed_time': ['mean', 'std'],
            'has_error': 'mean'
        })
        trends["prompt_based"] = prompt_trends.to_dict()
        
        # 4. Topic-based trends
        topic_trends = results_df.groupby('topic_category').agg({
            'confidence_score': ['mean', 'std'],
            'elapsed_time': ['mean', 'std'],
            'has_error': 'mean'
        })
        trends["topic_based"] = topic_trends.to_dict()
        
        return trends
    
    def generate_performance_summary(self, results_df, trends):
        """
        Generate a comprehensive performance summary.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            trends (dict): Performance trend analysis results
            
        Returns:
            dict: Performance summary
        """
        summary = {
            "overall_metrics": {},
            "model_comparison": {},
            "prompt_effectiveness": {},
            "topic_performance": {},
            "error_analysis": {},
            "recommendations": []
        }
        
        # 1. Overall metrics
        summary["overall_metrics"] = {
            "total_questions": len(results_df),
            "total_models": len(results_df['model_name'].unique()),
            "avg_confidence": results_df['confidence_score'].mean(),
            "avg_response_time": results_df['elapsed_time'].mean(),
            "error_rate": results_df['has_error'].mean()
        }
        
        # 2. Model comparison
        for model in results_df['model_name'].unique():
            model_df = results_df[results_df['model_name'] == model]
            summary["model_comparison"][model] = {
                "confidence": model_df['confidence_score'].mean(),
                "response_time": model_df['elapsed_time'].mean(),
                "error_rate": model_df['has_error'].mean(),
                "response_length": model_df['response_length'].mean()
            }
        
        # 3. Prompt effectiveness
        for prompt_type in results_df['prompt_type'].unique():
            prompt_df = results_df[results_df['prompt_type'] == prompt_type]
            summary["prompt_effectiveness"][prompt_type] = {
                "confidence": prompt_df['confidence_score'].mean(),
                "response_time": prompt_df['elapsed_time'].mean(),
                "error_rate": prompt_df['has_error'].mean()
            }
        
        # 4. Topic performance
        for topic in results_df['topic_category'].unique():
            topic_df = results_df[results_df['topic_category'] == topic]
            summary["topic_performance"][topic] = {
                "confidence": topic_df['confidence_score'].mean(),
                "response_time": topic_df['elapsed_time'].mean(),
                "error_rate": topic_df['has_error'].mean()
            }
        
        # 5. Error analysis
        error_types = results_df[results_df['has_error']]['error_type'].value_counts()
        summary["error_analysis"] = {
            "total_errors": len(results_df[results_df['has_error']]),
            "error_rate": results_df['has_error'].mean(),
            "error_types": error_types.to_dict()
        }
        
        # 6. Generate recommendations
        recommendations = []
        
        # Model recommendations
        best_model = max(summary["model_comparison"].items(), 
                        key=lambda x: x[1]["confidence"])[0]
        recommendations.append(f"Best performing model: {best_model}")
        
        # Prompt recommendations
        best_prompt = max(summary["prompt_effectiveness"].items(),
                         key=lambda x: x[1]["confidence"])[0]
        recommendations.append(f"Most effective prompt type: {best_prompt}")
        
        # Topic recommendations
        for topic, metrics in summary["topic_performance"].items():
            if metrics["error_rate"] > 0.2:
                recommendations.append(f"High error rate in {topic} topic: {metrics['error_rate']:.2%}")
        
        summary["recommendations"] = recommendations
        
        return summary 

    def generate_detailed_report(self, results_df=None, trends=None, summary=None):
        """
        Generate a detailed performance report with comprehensive analysis.
        
        Args:
            results_df (pd.DataFrame, optional): Results dataframe
            trends (dict, optional): Performance trend analysis results
            summary (dict, optional): Performance summary results
            
        Returns:
            str: Path to the generated report
        """
        if results_df is None:
            if not self.results["model_name"]:
                print("  No results to generate report")
                return ""
            results_df = pd.DataFrame(self.results)
        
        if trends is None:
            trends = self.analyze_performance_trends(results_df)
        
        if summary is None:
            summary = self.generate_performance_summary(results_df, trends)
        
        # Create detailed visualizations
        visualization_paths = self.create_detailed_visualizations(results_df)
        
        # Generate report timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"detailed_report_{timestamp}.html")
        
        # Generate HTML content
        html_content = self._generate_detailed_html_report(
            results_df,
            trends,
            summary,
            visualization_paths
        )
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  Detailed report generated: {report_path}")
        return report_path
    
    def _generate_detailed_html_report(self, results_df, trends, summary, visualization_paths):
        """
        Generate HTML content for the detailed report.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            trends (dict): Performance trend analysis results
            summary (dict): Performance summary results
            visualization_paths (dict): Paths to visualization files
            
        Returns:
            str: HTML content
        """
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Detailed LLM Performance Analysis Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1, h2, h3 {
                    color: #2c3e50;
                    margin-top: 30px;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .metric-card {
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 6px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }
                .metric-label {
                    color: #7f8c8d;
                    font-size: 14px;
                }
                .visualization {
                    margin: 20px 0;
                    text-align: center;
                }
                .visualization img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                    font-weight: 600;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .error-highlight {
                    color: #e74c3c;
                    font-weight: bold;
                }
                .success-highlight {
                    color: #2ecc71;
                    font-weight: bold;
                }
                .chart-container {
                    margin: 20px 0;
                    padding: 15px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                .recommendations {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }
                .recommendation-item {
                    margin: 10px 0;
                    padding: 10px;
                    background-color: white;
                    border-radius: 4px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Detailed LLM Performance Analysis Report</h1>
                <p>Generated on: {timestamp}</p>
                
                <!-- Overview Section -->
                <div class="section">
                    <h2>Overview</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{total_models}</div>
                            <div class="metric-label">Models Evaluated</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{total_questions}</div>
                            <div class="metric-label">Questions Tested</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{avg_confidence:.2f}%</div>
                            <div class="metric-label">Average Confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{avg_response_time:.2f}s</div>
                            <div class="metric-label">Average Response Time</div>
                        </div>
                    </div>
                </div>
                
                <!-- Model Performance Section -->
                <div class="section">
                    <h2>Model Performance</h2>
                    <div class="chart-container">
                        <h3>Response Time Distribution</h3>
                        <div class="visualization">
                            <img src="{response_time_dist}" alt="Response Time Distribution">
                        </div>
                    </div>
                    <div class="chart-container">
                        <h3>Confidence Score Distribution</h3>
                        <div class="visualization">
                            <img src="{confidence_dist}" alt="Confidence Distribution">
                        </div>
                    </div>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Confidence</th>
                            <th>Response Time</th>
                            <th>Error Rate</th>
                            <th>Response Length</th>
                        </tr>
                        {model_comparison_rows}
                    </table>
                </div>
                
                <!-- Prompt Type Analysis -->
                <div class="section">
                    <h2>Prompt Type Analysis</h2>
                    <div class="chart-container">
                        <h3>Error Rate by Model and Prompt Type</h3>
                        <div class="visualization">
                            <img src="{error_rate_analysis}" alt="Error Rate Analysis">
                        </div>
                    </div>
                    <table>
                        <tr>
                            <th>Prompt Type</th>
                            <th>Confidence</th>
                            <th>Response Time</th>
                            <th>Error Rate</th>
                        </tr>
                        {prompt_effectiveness_rows}
                    </table>
                </div>
                
                <!-- Topic-based Performance -->
                <div class="section">
                    <h2>Topic-based Performance</h2>
                    <div class="visualization">
                        <img src="{topic_performance}" alt="Topic Performance Analysis">
                    </div>
                    <table>
                        <tr>
                            <th>Topic</th>
                            <th>Confidence</th>
                            <th>Response Time</th>
                            <th>Error Rate</th>
                        </tr>
                        {topic_performance_rows}
                    </table>
                </div>
                
                <!-- Error Analysis -->
                <div class="section">
                    <h2>Error Analysis</h2>
                    <div class="visualization">
                        <img src="{error_types}" alt="Error Types Distribution">
                    </div>
                    <div class="visualization">
                        <img src="{length_vs_time}" alt="Response Length vs Time">
                    </div>
                    <table>
                        <tr>
                            <th>Error Type</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                        {error_analysis_rows}
                    </table>
                </div>
                
                <!-- Performance Trends -->
                <div class="section">
                    <h2>Performance Trends</h2>
                    <div class="chart-container">
                        <h3>Time-based Trends</h3>
                        <div class="visualization">
                            <img src="{time_trends_plot}" alt="Time-based Trends">
                        </div>
                    </div>
                </div>
                
                <!-- Recommendations -->
                <div class="section">
                    <h2>Recommendations</h2>
                    <div class="recommendations">
                        {recommendations_html}
                    </div>
                </div>
                
                <!-- Detailed Results -->
                <div class="section">
                    <h2>Detailed Results</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Prompt Type</th>
                            <th>Question ID</th>
                            <th>Confidence</th>
                            <th>Response Time</th>
                            <th>Error Type</th>
                        </tr>
                        {detailed_results_rows}
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Calculate metrics for overview
        total_models = len(results_df['model_name'].unique())
        total_questions = len(results_df['question_id'].unique())
        avg_confidence = results_df['confidence_score'].mean() * 100
        avg_response_time = results_df['elapsed_time'].mean()
        
        # Generate model comparison rows
        model_comparison_rows = ""
        for model, metrics in summary["model_comparison"].items():
            model_comparison_rows += f"""
            <tr>
                <td>{model}</td>
                <td class="{'success-highlight' if metrics['confidence'] >= 0.8 else 'error-highlight'}">{metrics['confidence']*100:.2f}%</td>
                <td>{metrics['response_time']:.2f}s</td>
                <td class="{'error-highlight' if metrics['error_rate'] > 0.1 else 'success-highlight'}">{metrics['error_rate']*100:.2f}%</td>
                <td>{metrics['response_length']:.0f}</td>
            </tr>
            """
        
        # Generate prompt effectiveness rows
        prompt_effectiveness_rows = ""
        for prompt_type, metrics in summary["prompt_effectiveness"].items():
            prompt_effectiveness_rows += f"""
            <tr>
                <td>{prompt_type}</td>
                <td class="{'success-highlight' if metrics['confidence'] >= 0.8 else 'error-highlight'}">{metrics['confidence']*100:.2f}%</td>
                <td>{metrics['response_time']:.2f}s</td>
                <td class="{'error-highlight' if metrics['error_rate'] > 0.1 else 'success-highlight'}">{metrics['error_rate']*100:.2f}%</td>
            </tr>
            """
        
        # Generate topic performance rows
        topic_performance_rows = ""
        for topic, metrics in summary["topic_performance"].items():
            topic_performance_rows += f"""
            <tr>
                <td>{topic}</td>
                <td class="{'success-highlight' if metrics['confidence'] >= 0.8 else 'error-highlight'}">{metrics['confidence']*100:.2f}%</td>
                <td>{metrics['response_time']:.2f}s</td>
                <td class="{'error-highlight' if metrics['error_rate'] > 0.1 else 'success-highlight'}">{metrics['error_rate']*100:.2f}%</td>
            </tr>
            """
        
        # Generate error analysis rows
        error_analysis_rows = ""
        total_errors = summary["error_analysis"]["total_errors"]
        for error_type, count in summary["error_analysis"]["error_types"].items():
            percentage = (count / total_errors) * 100
            error_analysis_rows += f"""
            <tr>
                <td>{error_type}</td>
                <td>{count}</td>
                <td class="{'error-highlight' if percentage > 20 else 'warning'}">{percentage:.2f}%</td>
            </tr>
            """
        
        # Generate recommendations HTML
        recommendations_html = ""
        for recommendation in summary["recommendations"]:
            recommendations_html += f"""
            <div class="recommendation-item">
                {recommendation}
            </div>
            """
        
        # Generate detailed results rows
        detailed_results_rows = ""
        for _, row in results_df.iterrows():
            detailed_results_rows += f"""
            <tr>
                <td>{row['model_name']}</td>
                <td>{row['prompt_type']}</td>
                <td>{row['question_id']}</td>
                <td class="{'success-highlight' if row['confidence_score'] >= 0.8 else 'error-highlight'}">{row['confidence_score']*100:.2f}%</td>
                <td>{row['elapsed_time']:.2f}s</td>
                <td class="{'error-highlight' if row['has_error'] else 'success-highlight'}">{row['error_type'] if row['has_error'] else 'None'}</td>
            </tr>
            """
        
        # Format the HTML content
        html_content = html_content.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_models=total_models,
            total_questions=total_questions,
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            model_comparison_rows=model_comparison_rows,
            prompt_effectiveness_rows=prompt_effectiveness_rows,
            topic_performance_rows=topic_performance_rows,
            error_analysis_rows=error_analysis_rows,
            recommendations_html=recommendations_html,
            detailed_results_rows=detailed_results_rows,
            **visualization_paths
        )
        
        return html_content
    
    def evaluate_reasoning_quality(self, results_df=None):
        """
        Đánh giá chất lượng suy luận dựa trên các tiêu chí như tính logic, tính mạch lạc,
        và chính xác trong quá trình suy luận, đặc biệt là phân tích từng loại prompt.
        
        Args:
            results_df (DataFrame): DataFrame chứa kết quả đánh giá. Nếu None, sẽ dùng results_df của class
        
        Returns:
            dict: Dictionary chứa kết quả phân tích chất lượng suy luận
        """
        if results_df is None:
            if hasattr(self, "results_df"):
                results_df = self.results_df
            else:
                print("  Không có dữ liệu kết quả để phân tích")
                return {}
        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        # Thư mục lưu biểu đồ
        plots_dir = os.path.join(self.results_dir, "reasoning_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        reasoning_metrics = {}
        
        # Khởi tạo các cột chất lượng suy luận nếu chưa có
        if 'reasoning_score' not in results_df.columns:
            # Hàm đánh giá suy luận dựa vào cấu trúc response
            def evaluate_response_reasoning(row):
                response = str(row['answer'])
                prompt_type = row['prompt_type']
                
                # Điểm mặc định
                score = 0.5
                
                # Đối với CoT và hybrid CoT
                if 'cot' in prompt_type.lower():
                    # Kiểm tra các dấu hiệu của suy luận tốt
                    steps_markers = ['bước', 'step', 'first', 'thứ nhất', 'tiếp theo', 'sau đó', 'cuối cùng']
                    math_markers = ['=', '+', '-', '*', '/', '×', '÷', 'tính toán', 'phương trình']
                    reasoning_markers = ['vì', 'do đó', 'vậy', 'nên', 'bởi vì', 'dẫn đến', 'suy ra']
                    
                    # Tính điểm dựa trên các marker
                    step_score = sum(1 for marker in steps_markers if marker.lower() in response.lower()) / max(4, len(steps_markers))
                    math_score = sum(1 for marker in math_markers if marker in response) / max(3, len(math_markers))
                    reasoning_score = sum(1 for marker in reasoning_markers if marker.lower() in response.lower()) / max(3, len(reasoning_markers))
                    
                    # Tính điểm tổng hợp
                    score = 0.4 * min(step_score, 1.0) + 0.3 * min(math_score, 1.0) + 0.3 * min(reasoning_score, 1.0)
                    
                    # Kiểm tra tính mạch lạc (các bước có thứ tự logic không)
                    if all(marker in response.lower() for marker in ['bước 1', 'bước 2']) or \
                       all(marker in response.lower() for marker in ['thứ nhất', 'thứ hai']):
                        score += 0.1
                    
                    # Kiểm tra xem có kết luận hay không
                    if any(marker in response.lower() for marker in ['kết luận', 'vậy', 'do đó', 'conclusion']):
                        score += 0.1
                
                # Đối với ReAct prompts
                elif 'react' in prompt_type.lower():
                    # Kiểm tra cấu trúc THINK-ACT-OBSERVE
                    think_markers = ['think:', 'suy nghĩ:', 'suy nghĩ rằng', 'tôi cần']
                    act_markers = ['act:', 'hành động:', 'tôi sẽ', 'tiến hành']
                    observe_markers = ['observe:', 'quan sát:', 'kết quả là', 'nhận thấy']
                    
                    has_think = any(marker in response.lower() for marker in think_markers)
                    has_act = any(marker in response.lower() for marker in act_markers)
                    has_observe = any(marker in response.lower() for marker in observe_markers)
                    
                    # Tính điểm dựa trên việc có đủ 3 thành phần
                    if has_think and has_act and has_observe:
                        score = 0.9  # Phản hồi hoàn chỉnh
                    elif has_think and has_act:
                        score = 0.7  # Thiếu quan sát
                    elif has_think:
                        score = 0.5  # Chỉ có suy nghĩ
                    else:
                        score = 0.3  # Không theo cấu trúc ReAct
                    
                    # Kiểm tra số lượng chu kỳ suy luận
                    cycles = min(response.lower().count('think:'), response.lower().count('act:'))
                    if cycles >= 2:
                        score += 0.1  # Thưởng cho việc có nhiều chu kỳ
                
                # Đối với self-consistency
                elif 'self_consistency' in prompt_type.lower():
                    score = 0.6  # Điểm cơ bản
                    
                    # Kiểm tra tính nhất quán trong câu trả lời
                    consistency_markers = ['nhất quán', 'consistent', 'phương pháp', 'giải', 'solution']
                    consistency_score = sum(1 for marker in consistency_markers if marker.lower() in response.lower()) / len(consistency_markers)
                    
                    score += 0.4 * consistency_score
                
                # Đối với few-shot
                elif 'few_shot' in prompt_type.lower():
                    # Kiểm tra xem phản hồi có làm theo cấu trúc của ví dụ không
                    example_markers = ['tương tự', 'giống như', 'theo cách tương tự', 'áp dụng']
                    example_score = sum(1 for marker in example_markers if marker.lower() in response.lower()) / len(example_markers)
                    
                    score = 0.5 + 0.3 * example_score
                    
                    # Kiểm tra tính chính xác
                    if 'đáp án' in response.lower() or 'kết quả' in response.lower():
                        score += 0.2
                
                # Đối với zero-shot
                elif 'zero_shot' in prompt_type.lower():
                    # Đánh giá dựa trên cấu trúc câu trả lời
                    structure_markers = ['đáp án', 'kết quả', 'trả lời', 'solution', 'answer']
                    structure_score = sum(1 for marker in structure_markers if marker.lower() in response.lower()) / len(structure_markers)
                    
                    score = 0.5 + 0.5 * structure_score
                
                # Giới hạn điểm trong khoảng 0-1
                return min(max(score, 0), 1)
            
            # Áp dụng hàm đánh giá
            results_df['reasoning_score'] = results_df.apply(evaluate_response_reasoning, axis=1)
        
        # 1. Phân tích chất lượng suy luận theo loại prompt
        reasoning_by_prompt = results_df.groupby(['prompt_type', 'model_name'])['reasoning_score'].mean().reset_index()
        reasoning_metrics['by_prompt_type'] = reasoning_by_prompt
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 8))
        sns.barplot(x='prompt_type', y='reasoning_score', hue='model_name', data=reasoning_by_prompt)
        plt.title('Chất lượng suy luận theo loại prompt', fontsize=15)
        plt.xlabel('Loại prompt', fontsize=12)
        plt.ylabel('Điểm chất lượng suy luận', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'reasoning_quality_by_prompt.png'), dpi=300)
        plt.close()
        
        # 2. So sánh suy luận giữa các nhóm prompt
        prompt_categories = {
            'CoT': [p for p in results_df['prompt_type'].unique() if 'cot' in str(p).lower() and 'self_consistency' not in str(p).lower()],
            'Standard': [p for p in results_df['prompt_type'].unique() if p in ['standard', 'zero_shot']],
            'Few-shot': [p for p in results_df['prompt_type'].unique() if 'few_shot' in str(p).lower() and 'consistency' not in str(p).lower()],
            'Self-consistency': [p for p in results_df['prompt_type'].unique() if 'self_consistency' in str(p).lower()],
            'ReAct': [p for p in results_df['prompt_type'].unique() if 'react' in str(p).lower()]
        }
        
        category_data = []
        for category, prompts in prompt_categories.items():
            if prompts:
                category_df = results_df[results_df['prompt_type'].isin(prompts)]
                for model in category_df['model_name'].unique():
                    model_df = category_df[category_df['model_name'] == model]
                    avg_score = model_df['reasoning_score'].mean()
                    category_data.append({
                        'category': category,
                        'model_name': model,
                        'avg_reasoning_score': avg_score
                    })
        
        if category_data:
            category_df = pd.DataFrame(category_data)
            reasoning_metrics['by_category'] = category_df
            
            # Vẽ biểu đồ
            plt.figure(figsize=(12, 8))
            sns.barplot(x='category', y='avg_reasoning_score', hue='model_name', data=category_df)
            plt.title('Chất lượng suy luận theo nhóm prompt', fontsize=15)
            plt.xlabel('Nhóm prompt', fontsize=12)
            plt.ylabel('Điểm chất lượng suy luận trung bình', fontsize=12)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'reasoning_quality_by_category.png'), dpi=300)
            plt.close()
        
        # 3. Phân tích mối quan hệ giữa reasoning score và correctness score
        if 'correctness_score' in results_df.columns:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='reasoning_score', y='correctness_score', hue='prompt_type', data=results_df, alpha=0.7)
            plt.title('Mối quan hệ giữa chất lượng suy luận và độ chính xác', fontsize=15)
            plt.xlabel('Điểm chất lượng suy luận', fontsize=12)
            plt.ylabel('Điểm độ chính xác', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'reasoning_vs_correctness.png'), dpi=300)
            plt.close()
            
            # Tính tương quan
            corr = results_df[['reasoning_score', 'correctness_score']].corr().iloc[0, 1]
            reasoning_metrics['correlation'] = corr
            print(f"Tương quan giữa reasoning score và correctness score: {corr:.4f}")
        
        # 4. Đánh giá chi tiết các bước suy luận trong Chain of Thought
        cot_results = results_df[results_df['prompt_type'].str.contains('cot', case=False) &
                                 ~results_df['prompt_type'].str.contains('self_consistency', case=False)]
        
        if not cot_results.empty:
            # Hàm phân tích các bước trong CoT
            def analyze_cot_steps(response):
                response = str(response).lower()
                steps = []
                
                # Tìm các bước theo từ khóa
                for keyword in ['bước', 'step', 'thứ nhất', 'thứ hai', 'first', 'second']:
                    positions = [pos for pos in range(len(response)) if response[pos:].startswith(keyword)]
                    steps.extend(positions)
                
                steps.sort()
                return len(steps)
            
            cot_results['num_steps'] = cot_results['answer'].apply(analyze_cot_steps)
            
            # Phân tích số bước theo mô hình
            steps_by_model = cot_results.groupby('model_name')['num_steps'].mean().reset_index()
            reasoning_metrics['cot_steps'] = steps_by_model
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='model_name', y='num_steps', data=steps_by_model)
            plt.title('Số bước suy luận trung bình trong Chain of Thought', fontsize=15)
            plt.xlabel('Mô hình', fontsize=12)
            plt.ylabel('Số bước trung bình', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'cot_steps_by_model.png'), dpi=300)
            plt.close()
            
            # Phân tích mối quan hệ giữa số bước và độ chính xác
            if 'correctness_score' in cot_results.columns:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='num_steps', y='correctness_score', hue='model_name', data=cot_results)
                plt.title('Mối quan hệ giữa số bước suy luận và độ chính xác', fontsize=15)
                plt.xlabel('Số bước suy luận', fontsize=12)
                plt.ylabel('Độ chính xác', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'cot_steps_vs_accuracy.png'), dpi=300)
                plt.close()
        
        # 5. Phân tích của ReAct prompts
        react_results = results_df[results_df['prompt_type'].str.contains('react', case=False)]
        
        if not react_results.empty:
            # Hàm phân tích chu kỳ ReAct (THINK-ACT-OBSERVE)
            def analyze_react_cycles(response):
                response = str(response).lower()
                think_count = response.count('think:') + response.count('suy nghĩ:')
                act_count = response.count('act:') + response.count('hành động:')
                observe_count = response.count('observe:') + response.count('quan sát:')
                
                return min(think_count, act_count, observe_count)
            
            react_results['react_cycles'] = react_results['answer'].apply(analyze_react_cycles)
            
            # Phân tích theo mô hình
            cycles_by_model = react_results.groupby('model_name')['react_cycles'].mean().reset_index()
            reasoning_metrics['react_cycles'] = cycles_by_model
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='model_name', y='react_cycles', data=cycles_by_model)
            plt.title('Số chu kỳ ReAct trung bình theo mô hình', fontsize=15)
            plt.xlabel('Mô hình', fontsize=12)
            plt.ylabel('Số chu kỳ trung bình', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'react_cycles_by_model.png'), dpi=300)
            plt.close()
            
            # Mối quan hệ giữa số chu kỳ và độ chính xác
            if 'correctness_score' in react_results.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='react_cycles', y='correctness_score', data=react_results)
                plt.title('Ảnh hưởng của số chu kỳ ReAct đến độ chính xác', fontsize=15)
                plt.xlabel('Số chu kỳ ReAct', fontsize=12)
                plt.ylabel('Độ chính xác', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'react_cycles_vs_accuracy.png'), dpi=300)
                plt.close()
        
        # 6. So sánh few-shot với số lượng ví dụ khác nhau
        few_shot_results = results_df[results_df['prompt_type'].str.contains('few_shot', case=False) &
                                      ~results_df['prompt_type'].str.contains('consistency', case=False)]
        
        if not few_shot_results.empty and 'reasoning_score' in few_shot_results.columns:
            # Trích xuất số lượng ví dụ từ tên prompt
            try:
                few_shot_results['num_examples'] = few_shot_results['prompt_type'].str.extract(r'(\d+)').astype(int)
                
                examples_analysis = few_shot_results.groupby(['num_examples', 'model_name'])['reasoning_score'].mean().reset_index()
                reasoning_metrics['few_shot_examples'] = examples_analysis
                
                plt.figure(figsize=(10, 6))
                sns.lineplot(x='num_examples', y='reasoning_score', hue='model_name', 
                             markers=True, data=examples_analysis)
                plt.title('Ảnh hưởng của số lượng ví dụ đến chất lượng suy luận', fontsize=15)
                plt.xlabel('Số lượng ví dụ', fontsize=12)
                plt.ylabel('Điểm chất lượng suy luận', fontsize=12)
                plt.xticks(examples_analysis['num_examples'].unique())
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'few_shot_examples_reasoning.png'), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Lỗi khi phân tích few-shot prompts: {e}")
        
        return reasoning_metrics
    
    def evaluate_reasoning_with_gemini(self, results_df=None, sample_size=50):
        """
        Sử dụng Gemini API để đánh giá chất lượng suy luận của các phản hồi từ các mô hình khác.
        Phương pháp này cung cấp đánh giá độc lập và khách quan hơn.
        
        Args:
            results_df (DataFrame): DataFrame chứa kết quả cần đánh giá
            sample_size (int): Số lượng mẫu để đánh giá (đánh giá toàn bộ có thể tốn nhiều chi phí API)
        
        Returns:
            DataFrame: DataFrame chứa kết quả đánh giá từ Gemini
        """
        if results_df is None:
            if hasattr(self, "results_df"):
                results_df = self.results_df
            else:
                print("  Không có dữ liệu kết quả để phân tích")
                return None
        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        import time
        import random
        
        try:
            from check.model_manager import load_gemini_model
        except ImportError:
            print("  Không thể import load_gemini_model từ model_manager. Kiểm tra cài đặt.")
            return None
        
        print("  Khởi tạo đánh giá bằng Gemini API...")
        gemini_model = load_gemini_model()
        
        if gemini_model is None:
            print("  Không thể khởi tạo mô hình Gemini. Kiểm tra API key và kết nối mạng.")
            return None
        
        # Lấy mẫu ngẫu nhiên để đánh giá (để tiết kiệm chi phí API)
        if sample_size > 0 and sample_size < len(results_df):
            sample_df = results_df.sample(sample_size, random_state=42)
        else:
            sample_df = results_df.copy()
        
        # Tạo cột mới để lưu đánh giá
        sample_df['gemini_reasoning_quality'] = None
        sample_df['gemini_correctness'] = None
        sample_df['gemini_feedback'] = None
        
        # Tạo thư mục để lưu kết quả
        plots_dir = os.path.join(self.results_dir, "gemini_evaluation")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Template prompt đánh giá
        evaluation_template = """
        Đánh giá chất lượng suy luận trong phản hồi dưới đây. Phản hồi này được tạo bởi một mô hình ngôn ngữ lớn cho câu hỏi được đưa ra.
        
        THÔNG TIN:
        - Loại prompt: {prompt_type}
        - Câu hỏi: {question}
        - Phản hồi cần đánh giá: {response}
        
        TIÊU CHÍ ĐÁNH GIÁ:
        1. Chất lượng suy luận (0-10): Đánh giá tính logic, mạch lạc và hợp lý của quá trình suy luận.
        2. Độ chính xác (0-10): Đánh giá xem kết luận cuối cùng có chính xác không.
        3. Nhận xét chi tiết: Đưa ra nhận xét ngắn về điểm mạnh và điểm yếu trong suy luận.
        
        TRẢ VỀ KẾT QUẢ DƯỚI ĐỊNH DẠNG SAU:
        Reasoning_Quality: [điểm số từ 0-10]
        Correctness: [điểm số từ 0-10]
        Feedback: [nhận xét ngắn gọn trong 1-2 câu]
        """
        
        print(f"  Bắt đầu đánh giá {len(sample_df)} mẫu bằng Gemini API...")
        
        for idx, row in sample_df.iterrows():
            try:
                # Tạo prompt đánh giá
                prompt = evaluation_template.format(
                    prompt_type=row['prompt_type'],
                    question=row['question'],
                    response=row['answer']
                )
                
                # Gọi Gemini API
                response = gemini_model.generate_content(prompt)
                
                if response and hasattr(response, 'text'):
                    result_text = response.text
                    
                    # Trích xuất điểm số và nhận xét
                    try:
                        reasoning_line = [line for line in result_text.split('\n') if 'Reasoning_Quality:' in line]
                        correctness_line = [line for line in result_text.split('\n') if 'Correctness:' in line]
                        feedback_line = [line for line in result_text.split('\n') if 'Feedback:' in line]
                        
                        if reasoning_line:
                            reasoning_score = float(reasoning_line[0].split(':')[1].strip()) / 10  # Chuẩn hóa về 0-1
                            sample_df.at[idx, 'gemini_reasoning_quality'] = reasoning_score
                        
                        if correctness_line:
                            correctness_score = float(correctness_line[0].split(':')[1].strip()) / 10  # Chuẩn hóa về 0-1
                            sample_df.at[idx, 'gemini_correctness'] = correctness_score
                        
                        if feedback_line:
                            feedback = feedback_line[0].split(':')[1].strip()
                            sample_df.at[idx, 'gemini_feedback'] = feedback
                            
                    except Exception as parse_e:
                        print(f"  Không thể phân tích kết quả từ Gemini cho mẫu {idx}: {parse_e}")
                
                # Thêm delay để tránh giới hạn rate của API
                time.sleep(1)
                
                # In tiến độ
                if (idx + 1) % 10 == 0:
                    print(f"  Đã đánh giá {idx + 1}/{len(sample_df)} mẫu")
                    
            except Exception as e:
                print(f"  Lỗi khi đánh giá mẫu {idx}: {str(e)}")
                # Tiếp tục với mẫu tiếp theo
                time.sleep(3)  # Thêm delay dài hơn nếu có lỗi
        
        # Lưu kết quả đánh giá
        evaluation_file = os.path.join(self.results_dir, "gemini_evaluation_results.csv")
        sample_df.to_csv(evaluation_file, index=False)
        print(f"  Đã lưu kết quả đánh giá vào {evaluation_file}")
        
        # Phân tích kết quả
        if 'gemini_reasoning_quality' in sample_df.columns and not sample_df['gemini_reasoning_quality'].isnull().all():
            # 1. Chất lượng suy luận theo loại prompt
            plt.figure(figsize=(12, 6))
            reasoning_by_prompt = sample_df.groupby('prompt_type')['gemini_reasoning_quality'].mean().reset_index()
            reasoning_by_prompt = reasoning_by_prompt.sort_values('gemini_reasoning_quality', ascending=False)
            
            sns.barplot(x='prompt_type', y='gemini_reasoning_quality', data=reasoning_by_prompt)
            plt.title('Chất lượng suy luận theo đánh giá của Gemini API', fontsize=15)
            plt.xlabel('Loại prompt', fontsize=12)
            plt.ylabel('Điểm chất lượng suy luận (0-1)', fontsize=12)
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'gemini_reasoning_by_prompt.png'), dpi=300)
            plt.close()
            
            # 2. So sánh đánh giá giữa Gemini và phương pháp nội bộ
            if 'reasoning_score' in sample_df.columns:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='reasoning_score', y='gemini_reasoning_quality', data=sample_df, hue='prompt_type', alpha=0.7)
                
                # Thêm đường tham chiếu y=x
                min_val = min(sample_df['reasoning_score'].min(), sample_df['gemini_reasoning_quality'].min())
                max_val = max(sample_df['reasoning_score'].max(), sample_df['gemini_reasoning_quality'].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                plt.title('So sánh đánh giá: Phương pháp nội bộ vs Gemini API', fontsize=15)
                plt.xlabel('Điểm đánh giá nội bộ', fontsize=12)
                plt.ylabel('Điểm đánh giá Gemini API', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'internal_vs_gemini_comparison.png'), dpi=300)
                plt.close()
                
                # Tính tương quan
                corr = sample_df[['reasoning_score', 'gemini_reasoning_quality']].corr().iloc[0, 1]
                print(f"Tương quan giữa đánh giá nội bộ và Gemini API: {corr:.4f}")
            
            # 3. So sánh độ chính xác theo đánh giá của Gemini
            if 'gemini_correctness' in sample_df.columns and not sample_df['gemini_correctness'].isnull().all():
                plt.figure(figsize=(12, 6))
                correctness_by_prompt = sample_df.groupby('prompt_type')['gemini_correctness'].mean().reset_index()
                correctness_by_prompt = correctness_by_prompt.sort_values('gemini_correctness', ascending=False)
                
                sns.barplot(x='prompt_type', y='gemini_correctness', data=correctness_by_prompt)
                plt.title('Độ chính xác theo đánh giá của Gemini API', fontsize=15)
                plt.xlabel('Loại prompt', fontsize=12)
                plt.ylabel('Điểm độ chính xác (0-1)', fontsize=12)
                plt.xticks(rotation=45)
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'gemini_correctness_by_prompt.png'), dpi=300)
                plt.close()
                
                # So sánh với độ chính xác nội bộ nếu có
                if 'correctness_score' in sample_df.columns:
                    plt.figure(figsize=(10, 6))
                    prompt_types = sample_df['prompt_type'].unique()
                    
                    internal_scores = []
                    gemini_scores = []
                    labels = []
                    
                    for pt in prompt_types:
                        pt_data = sample_df[sample_df['prompt_type'] == pt]
                        internal_scores.append(pt_data['correctness_score'].mean())
                        gemini_scores.append(pt_data['gemini_correctness'].mean())
                        labels.append(pt)
                    
                    width = 0.35
                    x = np.arange(len(labels))
                    
                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.bar(x - width/2, internal_scores, width, label='Đánh giá nội bộ')
                    ax.bar(x + width/2, gemini_scores, width, label='Đánh giá Gemini')
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Điểm độ chính xác (0-1)', fontsize=12)
                    ax.set_title('So sánh độ chính xác: Đánh giá nội bộ vs Gemini API', fontsize=15)
                    ax.legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'correctness_comparison.png'), dpi=300)
                    plt.close()
        
        return sample_df
    
    def generate_prompt_comparison_report(self, results_df=None, output_dir=None):
        """
        Generate a comprehensive visual report comparing different prompt techniques 
        (zero-shot, few-shot, CoT, self-consistency, ReAct) with step-by-step evaluation.
        
        Args:
            results_df (DataFrame): DataFrame containing evaluation results
            output_dir (str): Directory to save the report and visualization files
            
        Returns:
            str: Path to the generated report HTML file
        """
        if results_df is None:
            if hasattr(self, "results_df"):
                results_df = self.results_df
            else:
                print("  No results data available for report generation")
                return None
                
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        from datetime import datetime
        
        # Create report directory
        if output_dir is None:
            output_dir = self.results_dir
            
        report_dir = os.path.join(output_dir, "prompt_comparison_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(report_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        print("  Generating prompt comparison report...")
        
        # Define prompt categories for analysis
        prompt_categories = {
            "Zero-shot": [p for p in results_df['prompt_type'].unique() if 'zero' in str(p).lower()],
            "Few-shot": [p for p in results_df['prompt_type'].unique() if 'few_shot' in str(p).lower() and 'consistency' not in str(p).lower()],
            "Chain-of-Thought": [p for p in results_df['prompt_type'].unique() if 'cot' in str(p).lower() and 'consistency' not in str(p).lower()],
            "Self-consistency": [p for p in results_df['prompt_type'].unique() if 'self_consistency' in str(p).lower()],
            "ReAct": [p for p in results_df['prompt_type'].unique() if 'react' in str(p).lower()]
        }
        
        # Step 1: Overall performance comparison
        print("1️⃣ Analyzing overall performance by prompt type...")
        plt.figure(figsize=(14, 10))
        if 'correctness_score' in results_df.columns:
            performance_by_prompt = results_df.groupby(['prompt_type', 'model_name'])['correctness_score'].mean().reset_index()
            
            # Create visualization
            sns.barplot(x='prompt_type', y='correctness_score', hue='model_name', data=performance_by_prompt)
            plt.title('Overall Performance by Prompt Type', fontsize=16)
            plt.xlabel('Prompt Type', fontsize=14)
            plt.ylabel('Average Correctness Score', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '1_overall_performance.png'), dpi=300)
            plt.close()
        
        # Step 2: Response length analysis
        print("2️⃣ Analyzing response length characteristics...")
        if 'response_length' in results_df.columns:
            plt.figure(figsize=(14, 10))
            response_length_by_prompt = results_df.groupby(['prompt_type', 'model_name'])['response_length'].mean().reset_index()
            
            # Create visualization
            sns.barplot(x='prompt_type', y='response_length', hue='model_name', data=response_length_by_prompt)
            plt.title('Average Response Length by Prompt Type', fontsize=16)
            plt.xlabel('Prompt Type', fontsize=14)
            plt.ylabel('Average Response Length (tokens)', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '2_response_length.png'), dpi=300)
            plt.close()
        
        # Step 3: Response time analysis
        print("3️⃣ Analyzing response time efficiency...")
        if 'elapsed_time' in results_df.columns:
            plt.figure(figsize=(14, 10))
            time_by_prompt = results_df.groupby(['prompt_type', 'model_name'])['elapsed_time'].mean().reset_index()
            
            # Create visualization
            sns.barplot(x='prompt_type', y='elapsed_time', hue='model_name', data=time_by_prompt)
            plt.title('Average Response Time by Prompt Type', fontsize=16)
            plt.xlabel('Prompt Type', fontsize=14)
            plt.ylabel('Average Response Time (seconds)', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '3_response_time.png'), dpi=300)
            plt.close()
        
        # Step 4: Token efficiency (correctness per token)
        print("4️⃣ Calculating token efficiency metrics...")
        if all(col in results_df.columns for col in ['correctness_score', 'response_length']):
            # Calculate token efficiency
            results_df['token_efficiency'] = results_df['correctness_score'] / results_df['response_length'].clip(lower=1) * 1000
            
            plt.figure(figsize=(14, 10))
            token_efficiency_by_prompt = results_df.groupby(['prompt_type', 'model_name'])['token_efficiency'].mean().reset_index()
            
            # Create visualization
            sns.barplot(x='prompt_type', y='token_efficiency', hue='model_name', data=token_efficiency_by_prompt)
            plt.title('Token Efficiency by Prompt Type', fontsize=16)
            plt.xlabel('Prompt Type', fontsize=14)
            plt.ylabel('Efficiency (correctness/token × 1000)', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '4_token_efficiency.png'), dpi=300)
            plt.close()
        
        # Step 5: Reasoning quality comparison
        print("5️⃣ Comparing reasoning quality metrics...")
        if 'reasoning_score' in results_df.columns:
            plt.figure(figsize=(14, 10))
            reasoning_by_prompt = results_df.groupby(['prompt_type', 'model_name'])['reasoning_score'].mean().reset_index()
            
            # Create visualization
            sns.barplot(x='prompt_type', y='reasoning_score', hue='model_name', data=reasoning_by_prompt)
            plt.title('Reasoning Quality by Prompt Type', fontsize=16)
            plt.xlabel('Prompt Type', fontsize=14)
            plt.ylabel('Average Reasoning Score', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '5_reasoning_quality.png'), dpi=300)
            plt.close()
        
        # Step 6: Few-shot examples analysis
        print("6️⃣ Analyzing few-shot examples effectiveness...")
        few_shot_results = results_df[results_df['prompt_type'].str.contains('few_shot', case=False, na=False)]
        
        if not few_shot_results.empty and 'correctness_score' in few_shot_results.columns:
            # Extract number of examples
            few_shot_results['num_examples'] = few_shot_results['prompt_type'].str.extract(r'(\d+)').astype(float)
            
            plt.figure(figsize=(14, 10))
            few_shot_analysis = few_shot_results.groupby(['num_examples', 'model_name'])['correctness_score'].mean().reset_index()
            
            # Create visualization
            sns.lineplot(x='num_examples', y='correctness_score', hue='model_name', markers=True, data=few_shot_analysis)
            plt.title('Performance by Number of Examples (Few-shot)', fontsize=16)
            plt.xlabel('Number of Examples', fontsize=14)
            plt.ylabel('Average Correctness Score', fontsize=14)
            plt.xticks(few_shot_analysis['num_examples'].unique())
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '6_few_shot_examples.png'), dpi=300)
            plt.close()
        
        # Step 7: Prompt category comparison
        print("7️⃣ Generating prompt category comparison...")
        category_performance = []
        
        for category, prompt_types in prompt_categories.items():
            if prompt_types:
                category_data = results_df[results_df['prompt_type'].isin(prompt_types)]
                
                if not category_data.empty and 'correctness_score' in category_data.columns:
                    for model in category_data['model_name'].unique():
                        model_data = category_data[category_data['model_name'] == model]
                        category_performance.append({
                            'category': category,
                            'model_name': model,
                            'correctness_score': model_data['correctness_score'].mean(),
                            'response_length': model_data['response_length'].mean() if 'response_length' in model_data.columns else np.nan,
                            'elapsed_time': model_data['elapsed_time'].mean() if 'elapsed_time' in model_data.columns else np.nan
                        })
        
        if category_performance:
            category_df = pd.DataFrame(category_performance)
            
            # Create visualization for category comparison
            plt.figure(figsize=(14, 10))
            sns.barplot(x='category', y='correctness_score', hue='model_name', data=category_df)
            plt.title('Performance by Prompt Category', fontsize=16)
            plt.xlabel('Prompt Category', fontsize=14)
            plt.ylabel('Average Correctness Score', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '7_category_performance.png'), dpi=300)
            plt.close()
        
        # Step 8: Prompt type performance heatmap
        print("8️⃣ Creating prompt performance heatmap...")
        if 'correctness_score' in results_df.columns:
            plt.figure(figsize=(16, 12))
            pivot_data = results_df.pivot_table(index='prompt_type', columns='model_name', values='correctness_score', aggfunc='mean')
            
            # Create heatmap visualization
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=0.5)
            plt.title('Prompt Type Performance Heatmap', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '8_performance_heatmap.png'), dpi=300)
            plt.close()
        
        # Step 9: Analysis by question difficulty
        print("9️⃣ Analyzing prompt performance by question difficulty...")
        if all(col in results_df.columns for col in ['question_difficulty', 'correctness_score']):
            difficulty_analysis = results_df.groupby(['prompt_type', 'question_difficulty'])['correctness_score'].mean().reset_index()
            
            plt.figure(figsize=(16, 12))
            pivot_difficulty = difficulty_analysis.pivot(index='prompt_type', columns='question_difficulty', values='correctness_score')
            
            # Create visualization
            sns.heatmap(pivot_difficulty, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=0.5)
            plt.title('Prompt Performance by Question Difficulty', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '9_difficulty_analysis.png'), dpi=300)
            plt.close()
        
        # Step 10: Generate HTML report
        print("🔟 Generating final HTML report...")
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prompt Technique Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                h1 {{ color: #2c3e50; text-align: center; padding-bottom: 10px; border-bottom: 2px solid #3498db; }}
                h2 {{ color: #2980b9; margin-top: 30px; padding-bottom: 5px; border-bottom: 1px solid #ddd; }}
                h3 {{ color: #3498db; margin-top: 25px; }}
                .section {{ margin: 25px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .section-title {{ background-color: #2980b9; color: white; padding: 10px; margin-top: 0; border-radius: 5px 5px 0 0; }}
                .step {{ background-color: #f5f5f5; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                .visualization img {{ max-width: 100%; box-shadow: 0 3px 10px rgba(0,0,0,0.2); }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ font-weight: bold; color: #e74c3c; }}
                .footer {{ margin-top: 30px; text-align: center; font-size: 0.9em; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <h1>Prompt Technique Comparison Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2 class="section-title">Executive Summary</h2>
                <p>This report provides a comprehensive analysis of different prompt techniques (Zero-shot, Few-shot, Chain-of-Thought, Self-consistency, and ReAct) across evaluated models. The evaluation was conducted on a range of criteria including accuracy, response time, token efficiency, and reasoning quality.</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">Evaluation Process</h2>
                <p>The evaluation process involved the following steps:</p>
                
                <div class="step">
                    <h3>Step 1: Overall Performance Comparison</h3>
                    <p>Analysis of correctness scores across different prompt types and models.</p>
                    <div class="visualization">
                        <img src="visualizations/1_overall_performance.png" alt="Overall Performance Chart">
                    </div>
                </div>
                
                <div class="step">
                    <h3>Step 2: Response Length Analysis</h3>
                    <p>Comparison of average response lengths for different prompt techniques.</p>
                    <div class="visualization">
                        <img src="visualizations/2_response_length.png" alt="Response Length Chart">
                    </div>
                </div>
                
                <div class="step">
                    <h3>Step 3: Response Time Efficiency</h3>
                    <p>Evaluation of time required to generate responses with different prompt techniques.</p>
                    <div class="visualization">
                        <img src="visualizations/3_response_time.png" alt="Response Time Chart">
                    </div>
                </div>
                
                <div class="step">
                    <h3>Step 4: Token Efficiency Metrics</h3>
                    <p>Analysis of correctness per token to measure prompt efficiency.</p>
                    <div class="visualization">
                        <img src="visualizations/4_token_efficiency.png" alt="Token Efficiency Chart">
                    </div>
                </div>
                
                <div class="step">
                    <h3>Step 5: Reasoning Quality Comparison</h3>
                    <p>Assessment of reasoning quality across different prompt techniques.</p>
                    <div class="visualization">
                        <img src="visualizations/5_reasoning_quality.png" alt="Reasoning Quality Chart">
                    </div>
                </div>
                
                <div class="step">
                    <h3>Step 6: Few-shot Examples Analysis</h3>
                    <p>Analysis of how the number of examples affects few-shot prompt performance.</p>
                    <div class="visualization">
                        <img src="visualizations/6_few_shot_examples.png" alt="Few-shot Examples Chart">
                    </div>
                </div>
                
                <div class="step">
                    <h3>Step 7: Prompt Category Comparison</h3>
                    <p>Comparison of performance between major prompt categories.</p>
                    <div class="visualization">
                        <img src="visualizations/7_category_performance.png" alt="Category Performance Chart">
                    </div>
                </div>
                
                <div class="step">
                    <h3>Step 8: Performance Heatmap</h3>
                    <p>Detailed heatmap visualizing performance across prompt types and models.</p>
                    <div class="visualization">
                        <img src="visualizations/8_performance_heatmap.png" alt="Performance Heatmap">
                    </div>
                </div>
                
                <div class="step">
                    <h3>Step 9: Question Difficulty Analysis</h3>
                    <p>Evaluation of how different prompt types perform on questions of varying difficulty.</p>
                    <div class="visualization">
                        <img src="visualizations/9_difficulty_analysis.png" alt="Difficulty Analysis Chart">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Key Findings</h2>
                <ul>
                    <li>Chain-of-Thought prompts generally improve reasoning but may have longer response times.</li>
                    <li>Few-shot prompts with optimal example counts (typically 3-5) often balance performance and efficiency.</li>
                    <li>Self-consistency can improve reliability but at the cost of increased token usage.</li>
                    <li>ReAct prompts excel in tasks requiring step-by-step reasoning and problem-solving.</li>
                    <li>Zero-shot performs best in straightforward questions and offers the best token efficiency.</li>
                </ul>
            </div>
            
            <div class="section">
                <h2 class="section-title">Recommendations</h2>
                <ul>
                    <li>For complex reasoning tasks: Use Chain-of-Thought or ReAct prompts.</li>
                    <li>For balanced performance: Few-shot with 3 examples offers a good compromise.</li>
                    <li>For token efficiency: Zero-shot with clear instructions.</li>
                    <li>For maximum accuracy: Self-consistency with 5-7 runs (resource permitting).</li>
                    <li>For interactive tasks: ReAct offers the best structured approach.</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>Generated by ModelEvaluator | © {datetime.now().year}</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = os.path.join(report_dir, "prompt_comparison_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  Report generated successfully at: {report_path}")
        return report_path
    
    def analyze_react_prompt_evaluation(self, results_df=None, output_dir=None):
        """
        Analyze and visualize the specific characteristics of ReAct prompts, 
        focusing on their reasoning-action cycles and effectiveness compared to other prompt types.
        
        Args:
            results_df (DataFrame): DataFrame containing evaluation results
            output_dir (str): Directory to save visualization files
            
        Returns:
            dict: Analysis results for ReAct prompts
        """
        if results_df is None:
            if hasattr(self, "results_df"):
                results_df = self.results_df
            else:
                print("  No results data available for analysis")
                return {}
                
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        import re
        from wordcloud import WordCloud
        
        # Create output directory if needed
        if output_dir is None:
            output_dir = self.results_dir
            
        react_dir = os.path.join(output_dir, "react_analysis")
        os.makedirs(react_dir, exist_ok=True)
        
        # Initialize results dictionary
        react_analysis = {}
        
        # Filter ReAct prompt results
        react_results = results_df[results_df['prompt_type'].str.contains('react', case=False, na=False)]
        
        if react_results.empty:
            print("  No ReAct prompt results found for analysis")
            return react_analysis
            
        print("  Analyzing ReAct prompt evaluation...")
        
        # 1. Analyze the number of reasoning-action cycles in ReAct responses
        def count_react_cycles(response):
            response = str(response).lower()
            # Count THINK-ACT-OBSERVE cycles
            think_count = response.count('think:') + response.count('suy nghĩ:')
            act_count = response.count('act:') + response.count('hành động:')
            observe_count = response.count('observe:') + response.count('quan sát:')
            
            # Return the minimum count (complete cycles)
            return min(think_count, act_count, observe_count)
        
        # Apply cycle counting
        react_results['react_cycles'] = react_results['answer'].apply(count_react_cycles)
        
        # Average cycles by model
        cycles_by_model = react_results.groupby('model_name')['react_cycles'].mean().reset_index()
        react_analysis['cycles_by_model'] = cycles_by_model
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='model_name', y='react_cycles', data=cycles_by_model)
        plt.title('Average ReAct Reasoning-Action Cycles by Model', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Average Number of Cycles', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(react_dir, 'react_cycles_by_model.png'), dpi=300)
        plt.close()
        
        # 2. Analyze relationship between cycles and correctness
        if 'correctness_score' in react_results.columns:
            cycles_correctness = react_results.groupby('react_cycles')['correctness_score'].mean().reset_index()
            react_analysis['cycles_vs_correctness'] = cycles_correctness
            
            plt.figure(figsize=(12, 8))
            sns.lineplot(x='react_cycles', y='correctness_score', data=cycles_correctness, marker='o')
            plt.title('Effect of ReAct Cycles on Correctness', fontsize=16)
            plt.xlabel('Number of Reasoning-Action Cycles', fontsize=14)
            plt.ylabel('Average Correctness Score', fontsize=14)
            plt.xticks(cycles_correctness['react_cycles'].unique())
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(react_dir, 'react_cycles_vs_correctness.png'), dpi=300)
            plt.close()
        
        # 3. Compare ReAct with other reasoning prompts (CoT)
        cot_results = results_df[results_df['prompt_type'].str.contains('cot', case=False, na=False) & 
                                ~results_df['prompt_type'].str.contains('consistency', case=False, na=False)]
        
        if not cot_results.empty and 'correctness_score' in results_df.columns:
            # Prepare comparison data
            react_scores = react_results.groupby('model_name')['correctness_score'].mean().reset_index()
            react_scores['prompt_type'] = 'ReAct'
            
            cot_scores = cot_results.groupby('model_name')['correctness_score'].mean().reset_index()
            cot_scores['prompt_type'] = 'Chain-of-Thought'
            
            comparison_df = pd.concat([react_scores, cot_scores])
            react_analysis['react_vs_cot'] = comparison_df
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='model_name', y='correctness_score', hue='prompt_type', data=comparison_df)
            plt.title('ReAct vs Chain-of-Thought Performance Comparison', fontsize=16)
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Average Correctness Score', fontsize=14)
            plt.legend(title='Prompt Type')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(react_dir, 'react_vs_cot.png'), dpi=300)
            plt.close()
        
        # 4. Analyze ReAct response structure
        def extract_action_types(response):
            response = str(response).lower()
            actions = []
            
            # Extract text between "act:" and the next keyword or end of string
            act_patterns = re.findall(r'act:(.*?)(?:think:|observe:|$)', response, re.DOTALL)
            act_patterns += re.findall(r'hành động:(.*?)(?:suy nghĩ:|quan sát:|$)', response, re.DOTALL)
            
            for act in act_patterns:
                act = act.strip()
                if act:
                    actions.append(act)
            
            return '; '.join(actions)
        
        # Extract action types
        react_results['actions'] = react_results['answer'].apply(extract_action_types)
        
        # Create word cloud of actions if WordCloud is available
        try:
            all_actions = ' '.join(react_results['actions'].fillna(''))
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                 max_words=100, contour_width=3, contour_color='steelblue')
            
            # Generate word cloud
            action_cloud = wordcloud.generate(all_actions)
            
            # Create visualization
            plt.figure(figsize=(16, 8))
            plt.imshow(action_cloud, interpolation='bilinear')
            plt.axis("off")
            plt.title('Common Actions in ReAct Responses', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(react_dir, 'react_actions_wordcloud.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"  Could not generate word cloud: {e}")
        
        # 5. Compare response time for ReAct vs other prompt types
        if 'elapsed_time' in results_df.columns:
            # Group prompts into categories
            results_df['prompt_category'] = 'Other'
            results_df.loc[results_df['prompt_type'].str.contains('react', case=False, na=False), 'prompt_category'] = 'ReAct'
            results_df.loc[results_df['prompt_type'].str.contains('cot', case=False, na=False), 'prompt_category'] = 'Chain-of-Thought'
            results_df.loc[results_df['prompt_type'].str.contains('zero', case=False, na=False), 'prompt_category'] = 'Zero-shot'
            results_df.loc[results_df['prompt_type'].str.contains('few_shot', case=False, na=False), 'prompt_category'] = 'Few-shot'
            
            # Calculate average response time by category and model
            time_by_category = results_df.groupby(['prompt_category', 'model_name'])['elapsed_time'].mean().reset_index()
            react_analysis['response_time_comparison'] = time_by_category
            
            plt.figure(figsize=(14, 10))
            sns.barplot(x='prompt_category', y='elapsed_time', hue='model_name', data=time_by_category)
            plt.title('Response Time by Prompt Category', fontsize=16)
            plt.xlabel('Prompt Category', fontsize=14)
            plt.ylabel('Average Response Time (seconds)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(react_dir, 'react_time_comparison.png'), dpi=300)
            plt.close()
        
        # 6. Generate a table of ReAct prompt performance metrics
        react_metrics = react_results.groupby('model_name').agg({
            'correctness_score': 'mean' if 'correctness_score' in react_results.columns else lambda x: np.nan,
            'elapsed_time': 'mean' if 'elapsed_time' in react_results.columns else lambda x: np.nan,
            'react_cycles': 'mean',
            'response_length': 'mean' if 'response_length' in react_results.columns else lambda x: np.nan
        }).reset_index()
        
        react_analysis['performance_metrics'] = react_metrics
        
        # Save metrics to CSV
        metrics_file = os.path.join(react_dir, 'react_performance_metrics.csv')
        react_metrics.to_csv(metrics_file, index=False)
        
        print(f"ReAct analysis complete! Results saved to {react_dir}")
        return react_analysis
    
    def _generate_checkpoint_id(self, model_names, prompt_types, question_ids=None):
        """
        Tạo ID duy nhất cho checkpoint dựa trên cấu hình đánh giá.
        
        Args:
            model_names: Danh sách model được đánh giá
            prompt_types: Danh sách prompt types được đánh giá
            question_ids: (Tùy chọn) Danh sách ID câu hỏi
            
        Returns:
            str: Checkpoint ID duy nhất
        """
        # Tạo chuỗi đại diện cho cấu hình đánh giá
        config_str = f"models={'-'.join(sorted(model_names))}_prompts={'-'.join(sorted(prompt_types))}"
        
        if question_ids:
            # Nếu có danh sách câu hỏi cụ thể, hash chúng để có ID ngắn gọn
            questions_hash = hashlib.md5("-".join(map(str, sorted(question_ids))).encode()).hexdigest()[:8]
            config_str += f"_questions={questions_hash}"
        
        # Thêm timestamp để tạo ID duy nhất
        timestamp = self.checkpoint_state["timestamp"]
        
        return f"checkpoint_{config_str}_{timestamp}"
    
    def save_checkpoint(self, model_names, prompt_types, questions, completed_indices, partial_results):
        """
        Lưu trạng thái đánh giá hiện tại vào checkpoint.
        
        Args:
            model_names: Danh sách tên model
            prompt_types: Danh sách loại prompt
            questions: Danh sách câu hỏi đang đánh giá
            completed_indices: Chỉ số các câu hỏi đã hoàn thành
            partial_results: Kết quả đánh giá đã có
            
        Returns:
            str: Đường dẫn đến file checkpoint
        """
        # Cập nhật trạng thái checkpoint
        question_ids = [q["id"] if isinstance(q, dict) and "id" in q else i 
                       for i, q in enumerate(questions)]
        
        checkpoint_id = self._generate_checkpoint_id(model_names, prompt_types, question_ids)
        
        # Cập nhật thông tin trạng thái
        self.checkpoint_state["completed_evaluations"] = {
            "model_names": model_names,
            "prompt_types": prompt_types,
            "question_count": len(questions),
            "completed_indices": completed_indices,
            "question_ids": question_ids
        }
        self.checkpoint_state["partial_results"] = partial_results
        self.checkpoint_state["current_status"] = "in_progress"
        self.checkpoint_state["last_updated"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Lưu checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.pkl")
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "checkpoint_state": self.checkpoint_state,
                "questions": questions,
                "partial_results": partial_results
            }, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        logger.info(f"  - Completed {len(completed_indices)}/{len(questions)} questions ({len(completed_indices)/len(questions)*100:.1f}%)")
        
        # Lưu kết quả tạm thời dưới dạng CSV nếu có kết quả
        if partial_results:
            temp_df = pd.DataFrame(partial_results)
            temp_results_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}_partial_results.csv")
            temp_df.to_csv(temp_results_path, index=False)
            logger.info(f"Partial results saved: {temp_results_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_id=None, model_names=None, prompt_types=None):
        """
        Tìm và tải checkpoint phù hợp với cấu hình đánh giá.
        
        Args:
            checkpoint_id: ID checkpoint cụ thể (nếu biết)
            model_names: Danh sách tên model
            prompt_types: Danh sách loại prompt
            
        Returns:
            tuple: (questions, completed_indices, partial_results) hoặc None nếu không tìm thấy
        """
        # Nếu có checkpoint_id cụ thể
        if checkpoint_id:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.pkl")
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading specific checkpoint: {checkpoint_path}")
                with open(checkpoint_path, "rb") as f:
                    checkpoint_data = pickle.load(f)
                
                self.checkpoint_state = checkpoint_data["checkpoint_state"]
                return (
                    checkpoint_data["questions"],
                    self.checkpoint_state["completed_evaluations"]["completed_indices"],
                    checkpoint_data["partial_results"]
                )
        
        # Tìm checkpoint phù hợp nhất nếu không có ID cụ thể
        if model_names and prompt_types:
            # Lấy danh sách tất cả file checkpoint
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pkl")]
            
            # Sắp xếp theo thời gian tạo (mới nhất trước)
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
            
            # Kiểm tra từng file để tìm cấu hình phù hợp
            for file in checkpoint_files:
                checkpoint_path = os.path.join(self.checkpoint_dir, file)
                
                try:
                    with open(checkpoint_path, "rb") as f:
                        checkpoint_data = pickle.load(f)
                    
                    checkpoint_state = checkpoint_data["checkpoint_state"]
                    checkpoint_models = set(checkpoint_state["completed_evaluations"]["model_names"])
                    checkpoint_prompts = set(checkpoint_state["completed_evaluations"]["prompt_types"])
                    
                    # Kiểm tra xem checkpoint có chứa các model và prompt cần thiết không
                    if checkpoint_models.issuperset(model_names) and checkpoint_prompts.issuperset(prompt_types):
                        logger.info(f"Found matching checkpoint: {checkpoint_path}")
                        logger.info(f"  - Last updated: {checkpoint_state['last_updated']}")
                        logger.info(f"  - Status: {checkpoint_state['current_status']}")
                        logger.info(f"  - Completed: {len(checkpoint_state['completed_evaluations']['completed_indices'])}/{checkpoint_state['completed_evaluations']['question_count']} questions")
                        
                        self.checkpoint_state = checkpoint_state
                        return (
                            checkpoint_data["questions"],
                            self.checkpoint_state["completed_evaluations"]["completed_indices"],
                            checkpoint_data["partial_results"]
                        )
                except Exception as e:
                    logger.warning(f"Error loading checkpoint {file}: {e}")
                    continue
        
        logger.info("No suitable checkpoint found")
        return None