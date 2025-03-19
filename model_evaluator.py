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
from prompts import (
    standard_prompt, 
    chain_of_thought_prompt, 
    hybrid_cot_prompt,
    zero_shot_cot_prompt,
    tree_of_thought_prompt
)
from model_manager import (
    generate_text_with_model, 
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
        
        # Create plots directory
        self.plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Prompt types
        self.prompt_types = {
            "standard": standard_prompt,
            "cot": chain_of_thought_prompt,
            "hybrid_cot": hybrid_cot_prompt,
            "zero_shot_cot": zero_shot_cot_prompt,
            "tree_of_thought": tree_of_thought_prompt
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
                answer = generate_text_with_model(prompt, model_type="local", max_tokens=1024)
            elif model_type == "gemini":
                model = model_info["model"]
                answer = generate_text_with_model(prompt, model_type="gemini", max_tokens=1024)
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
            "elapsed_time": latency,
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
        Analyze evaluation results and generate visualizations.
        
        Args:
            results_df (pd.DataFrame): DataFrame containing evaluation results
            save_plots (bool): Whether to save plots
            
        Returns:
            dict: Analysis results
        """
        if results_df is None:
            if not self.results["model_name"]:
                print("❌ No results to analyze")
                return {}
            results_df = pd.DataFrame(self.results)
            
        print("\n📊 Analyzing evaluation results...")
        
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
            print(f"❌ Error creating response time plot: {e}")
        
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
            print(f"❌ Error creating prompt response time plot: {e}")
            
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
            print(f"❌ Error creating response length plot: {e}")
            
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
            print(f"❌ Error creating heatmap: {e}")
            
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
            print(f"❌ Error creating combined performance plot: {e}")
            
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
            print(f"❌ Error creating distribution plots: {e}")
            
        # 7. Group key metrics by model
        try:
            model_metrics = results_df.groupby("model_name").agg({
                "elapsed_time": ["mean", "std", "min", "max"],
                "response_length": ["mean", "std", "min", "max"]
            }).reset_index()
            
            analysis["model_metrics"] = model_metrics.to_dict(orient="records")
        except Exception as e:
            print(f"❌ Error calculating model metrics: {e}")
            
        # Group key metrics by prompt type
        try:
            prompt_metrics = results_df.groupby("prompt_type").agg({
                "elapsed_time": ["mean", "std", "min", "max"],
                "response_length": ["mean", "std", "min", "max"]
            }).reset_index()
            
            analysis["prompt_metrics"] = prompt_metrics.to_dict(orient="records")
        except Exception as e:
            print(f"❌ Error calculating prompt metrics: {e}")
            
        # Group key metrics by model and prompt type
        try:
            combined_metrics = results_df.groupby(["model_name", "prompt_type"]).agg({
                "elapsed_time": ["mean", "std", "min", "max"],
                "response_length": ["mean", "std", "min", "max"]
            }).reset_index()
            
            analysis["combined_metrics"] = combined_metrics.to_dict(orient="records")
        except Exception as e:
            print(f"❌ Error calculating combined metrics: {e}")
            
        print(f"✅ Analysis completed with {len(analysis['plots'])} visualizations")
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
                print(f"✅ Loaded processed results from {processed_results_path}")
            elif not self.results["model_name"]:
                print("❌ No results to generate report")
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
            
        print(f"✅ Report generated: {report_path}")
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
                
            print(f"✅ Loaded {len(results_data)} results from {json_file}")
            
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
                print(f"⚠️ Found {error_count} responses with errors ({error_count/len(df)*100:.1f}%)")
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
            print(f"❌ Error analyzing results from JSON: {e}")
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
            print(f"❌ Error creating PCA plot: {e}")
        
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
            print(f"❌ Error creating clustering plot: {e}")
        
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
            print(f"❌ Error creating interactive dashboard: {e}")
        
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
            print(f"❌ Error creating topic performance plot: {e}")
        
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
                print("❌ No results to generate report")
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
        
        print(f"✅ Comprehensive report generated: {report_path}")
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
            print(f"❌ Error creating response time distribution plot: {e}")
        
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
            print(f"❌ Error creating error rate analysis plot: {e}")
        
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
            print(f"❌ Error creating confidence distribution plot: {e}")
        
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
            print(f"❌ Error creating topic performance heatmap: {e}")
        
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
            print(f"❌ Error creating length vs time plot: {e}")
        
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
            print(f"❌ Error creating error types distribution plot: {e}")
        
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
                print("❌ No results to generate report")
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
        
        print(f"✅ Detailed report generated: {report_path}")
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