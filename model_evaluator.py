"""
Model evaluator for comparing performance of different LLMs on classical problems.
"""

import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher
import torch
import re

# Local imports
from prompts import (
    standard_prompt, 
    chain_of_thought_prompt, 
    hybrid_cot_prompt,
    zero_shot_cot_prompt,
    tree_of_thought_prompt
)
from model_manager import (
    generate_text_with_local_model, 
    generate_text_with_gemini,
    clear_memory,
    check_gpu_memory
)


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
        os.makedirs(results_dir, exist_ok=True)
        
        # Prompt types
        self.prompt_types = {
            "standard": standard_prompt,
            "cot": chain_of_thought_prompt,
            "hybrid_cot": hybrid_cot_prompt,
            "zero_shot_cot": zero_shot_cot_prompt,
            "tree_of_thought": tree_of_thought_prompt
        }
        
        # Metrics
        self.results = {
            "model_name": [],
            "prompt_type": [],
            "question_id": [],
            "question": [],
            "answer": [],
            "token_count": [],
            "latency": [],  # Generation time in seconds
            "tokens_per_second": [],
            "response_length": []
        }
    
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
            print(f"✅ Loaded {len(questions)} questions from {questions_file}")
            return questions
        except Exception as e:
            print(f"❌ Error loading questions: {e}")
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
    
    def evaluate_single_question(self, question, model_info, prompt_type="standard"):
        """
        Evaluate a single question with a specific model and prompt type.
        
        Args:
            question (str): Question to evaluate
            model_info (dict): Model information dictionary
            prompt_type (str): Type of prompt to use
            
        Returns:
            dict: Result metrics
        """
        model_name = model_info["name"]
        model_type = model_info["type"]
        prompt_fn = self.prompt_types[prompt_type]
        
        # Create prompt
        prompt = prompt_fn(question, "classical_problem")
        
        # Track metrics
        start_time = time.time()
        token_count = self.estimate_token_count(prompt)
        
        # Generate answer
        try:
            if model_type == "local":
                model = model_info["model"]
                tokenizer = model_info["tokenizer"]
                answer = generate_text_with_local_model(prompt, model, tokenizer)
            elif model_type == "gemini":
                model = model_info["model"]
                answer = generate_text_with_gemini(prompt, model)
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
            "question": question,
            "answer": answer,
            "token_count": token_count,
            "latency": latency,
            "tokens_per_second": tokens_per_second,
            "response_length": response_length
        }
        
        return result
    
    def evaluate_all_questions(self, questions, model_names=None, prompt_types=None, max_questions=None):
        """
        Evaluate all questions with specified models and prompt types.
        
        Args:
            questions (list): List of questions to evaluate
            model_names (list, optional): List of model names to evaluate (default: all)
            prompt_types (list, optional): List of prompt types to use (default: all)
            max_questions (int, optional): Maximum number of questions to evaluate
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
            
        if not model_names:
            model_names = list(self.models.keys())
            
        if not prompt_types:
            prompt_types = list(self.prompt_types.keys())
        
        print(f"🔍 Evaluating {len(questions)} questions with {len(model_names)} models and {len(prompt_types)} prompt types")
        
        for q_id, question in enumerate(tqdm(questions, desc="Questions")):
            for model_name in model_names:
                if model_name not in self.models:
                    print(f"⚠️ Model {model_name} not found in available models")
                    continue
                
                model_info = self.models[model_name]
                
                for prompt_type in prompt_types:
                    print(f"\n📝 Evaluating Question {q_id+1} with {model_name} using {prompt_type} prompt")
                    
                    # Check GPU memory before inference
                    if torch.cuda.is_available():
                        check_gpu_memory()
                    
                    # Evaluate question
                    result = self.evaluate_single_question(question, model_info, prompt_type)
                    
                    # Add question ID to result
                    result["question_id"] = q_id
                    
                    # Append to results
                    for key, value in result.items():
                        if key in self.results:
                            self.results[key].append(value)
                    
                    # Clear memory after each evaluation
                    clear_memory()
        
        # Convert results to dataframe
        results_df = pd.DataFrame(self.results)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"✅ Results saved to {results_path}")
        
        return results_df
    
    def analyze_results(self, results_df=None, save_plots=True):
        """
        Analyze results and generate visualizations.
        
        Args:
            results_df (pd.DataFrame, optional): Results dataframe
            save_plots (bool): Whether to save plots to files
            
        Returns:
            dict: Analysis metrics
        """
        if results_df is None:
            if not self.results["model_name"]:
                print("❌ No results to analyze")
                return {}
            results_df = pd.DataFrame(self.results)
        
        print("\n📊 Analyzing results...")
        
        # Create directory for plots
        plots_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Performance metrics by model and prompt type
        performance_df = results_df.groupby(['model_name', 'prompt_type']).agg({
            'latency': ['mean', 'std'],
            'tokens_per_second': ['mean', 'std'],
            'response_length': ['mean', 'std']
        }).reset_index()
        
        # 2. Generate plots
        # Plot 1: Average latency by model and prompt type
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df, 
            x='model_name', 
            y='latency', 
            hue='prompt_type',
            palette='viridis'
        )
        plt.title('Average Latency by Model and Prompt Type')
        plt.xlabel('Model')
        plt.ylabel('Latency (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plots_dir, 'latency_comparison.png'))
            
        # Plot 2: Tokens per second by model
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df, 
            x='model_name', 
            y='tokens_per_second',
            palette='viridis'
        )
        plt.title('Tokens per Second by Model')
        plt.xlabel('Model')
        plt.ylabel('Tokens per Second')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plots_dir, 'tokens_per_second.png'))
        
        # Plot 3: Response length by model and prompt type
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df, 
            x='model_name', 
            y='response_length', 
            hue='prompt_type',
            palette='viridis'
        )
        plt.title('Average Response Length by Model and Prompt Type')
        plt.xlabel('Model')
        plt.ylabel('Response Length (characters)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plots_dir, 'response_length.png'))
        
        # Plot 4: Heatmap of relative performance
        # Normalize metrics for fair comparison
        heatmap_data = results_df.pivot_table(
            index='model_name', 
            columns='prompt_type', 
            values='latency', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm_r', fmt='.2f')
        plt.title('Average Latency Heatmap (seconds)')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plots_dir, 'latency_heatmap.png'))
            
        # Save analysis results
        analysis_results = {
            "performance_metrics": performance_df.to_dict(),
            "plot_paths": {
                "latency_comparison": os.path.join(plots_dir, 'latency_comparison.png'),
                "tokens_per_second": os.path.join(plots_dir, 'tokens_per_second.png'),
                "response_length": os.path.join(plots_dir, 'response_length.png'),
                "latency_heatmap": os.path.join(plots_dir, 'latency_heatmap.png')
            }
        }
        
        print("✅ Analysis completed")
        
        return analysis_results
    
    def generate_report(self, results_df=None, analysis_results=None):
        """
        Generate a comprehensive HTML report of the evaluation.
        
        Args:
            results_df (pd.DataFrame, optional): Results dataframe
            analysis_results (dict, optional): Analysis metrics
            
        Returns:
            str: Path to the generated report
        """
        if results_df is None:
            if not self.results["model_name"]:
                print("❌ No results to generate report")
                return ""
            results_df = pd.DataFrame(self.results)
            
        if analysis_results is None:
            analysis_results = self.analyze_results(results_df, save_plots=True)
            
        print("\n📝 Generating evaluation report...")
        
        # Create report with HTML
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"evaluation_report_{timestamp}.html")
        
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
            </style>
        </head>
        <body>
            <div class="container">
                <h1>LLM Evaluation Report</h1>
                <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Performance Summary</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Models Evaluated</h3>
                        <p>{len(results_df['model_name'].unique())}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Questions Processed</h3>
                        <p>{len(results_df['question_id'].unique())}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Prompt Types</h3>
                        <p>{len(results_df['prompt_type'].unique())}</p>
                    </div>
                </div>
                
                <h2>Performance Visualizations</h2>
                <div class="plot-container">
                    <h3>Latency Comparison</h3>
                    <img src="plots/latency_comparison.png" alt="Latency Comparison" style="max-width:100%;">
                </div>
                
                <div class="plot-container">
                    <h3>Tokens per Second</h3>
                    <img src="plots/tokens_per_second.png" alt="Tokens per Second" style="max-width:100%;">
                </div>
                
                <div class="plot-container">
                    <h3>Response Length Comparison</h3>
                    <img src="plots/response_length.png" alt="Response Length" style="max-width:100%;">
                </div>
                
                <div class="plot-container">
                    <h3>Latency Heatmap</h3>
                    <img src="plots/latency_heatmap.png" alt="Latency Heatmap" style="max-width:100%;">
                </div>
                
                <h2>Sample Responses</h2>
        """
        
        # Add sample responses for comparison
        sample_questions = results_df['question_id'].unique()[:3]  # Take first 3 questions
        
        for q_id in sample_questions:
            q_results = results_df[results_df['question_id'] == q_id]
            question = q_results['question'].iloc[0]
            
            html_content += f"""
                <h3>Question {q_id + 1}</h3>
                <div class="model-response">
                    <p><strong>Question:</strong> {question}</p>
                </div>
            """
            
            for _, row in q_results.iterrows():
                model_name = row['model_name']
                prompt_type = row['prompt_type']
                answer = row['answer'].replace('\n', '<br>')
                latency = row['latency']
                
                html_content += f"""
                <div class="model-response">
                    <h4>{model_name} with {prompt_type} prompt</h4>
                    <p><strong>Latency:</strong> {latency:.2f} seconds</p>
                    <p><strong>Response:</strong><br>{answer}</p>
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✅ Report generated: {report_path}")
        
        return report_path 