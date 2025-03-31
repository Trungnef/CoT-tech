"""
HTML report generation for visualization of evaluation results.
"""

import os
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Class for generating HTML reports with visualizations from evaluation results.
    """
    
    def __init__(self, result_dir: Union[str, Path]):
        """
        Initialize the report generator.
        
        Args:
            result_dir: Directory containing evaluation results
        """
        self.result_dir = Path(result_dir)
        self.plots_dir = self.result_dir / "plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # HTML template parts
        self._html_header = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LLM Evaluation Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9fafb;
                }
                
                h1 {
                    color: #1a56db;
                    border-bottom: 2px solid #e5e7eb;
                    padding-bottom: 10px;
                    margin-top: 30px;
                }
                
                h2 {
                    color: #2563eb;
                    margin-top: 25px;
                }
                
                h3 {
                    color: #3b82f6;
                    margin-top: 20px;
                }
                
                .container {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }
                
                .plot-container {
                    margin-top: 20px;
                    margin-bottom: 30px;
                }
                
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                
                th, td {
                    border: 1px solid #e5e7eb;
                    padding: 12px 15px;
                    text-align: left;
                }
                
                th {
                    background-color: #f3f4f6;
                    font-weight: bold;
                }
                
                tr:nth-child(even) {
                    background-color: #f9fafb;
                }
                
                .highlight {
                    background-color: #fffbeb;
                    padding: 15px;
                    border-radius: 6px;
                    border-left: 4px solid #fbbf24;
                    margin: 20px 0;
                }
                
                .metric-card {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    padding: 15px;
                    margin-bottom: 15px;
                    display: flex;
                    flex-direction: column;
                }
                
                .metric-title {
                    font-size: 14px;
                    color: #6b7280;
                    margin-bottom: 5px;
                }
                
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #1e40af;
                }
                
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }
                
                .tabs {
                    display: flex;
                    border-bottom: 1px solid #e5e7eb;
                    margin-bottom: 20px;
                }
                
                .tab {
                    padding: 10px 20px;
                    cursor: pointer;
                    border-bottom: 2px solid transparent;
                }
                
                .tab.active {
                    border-bottom: 2px solid #3b82f6;
                    color: #3b82f6;
                    font-weight: bold;
                }
                
                .tab-content {
                    display: none;
                }
                
                .tab-content.active {
                    display: block;
                }
            </style>
        </head>
        <body>
        """
        
        self._html_footer = """
        <script>
            function switchTab(tabName, element) {
                // Hide all tab contents
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Show selected tab content and mark tab as active
                document.getElementById(tabName).classList.add('active');
                element.classList.add('active');
            }
        </script>
        </body>
        </html>
        """
        
        # Title and config info
        self._report_title = '<h1 class="text-3xl font-bold mb-6">LLM Evaluation Report</h1>'
    
    def load_results(self, results_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load evaluation results from a CSV or JSON file.
        
        Args:
            results_path: Path to results file (default: looks for results.csv in result_dir)
            
        Returns:
            pd.DataFrame: DataFrame containing results
        """
        if results_path is None:
            # Try to find results.csv in result_dir
            csv_path = self.result_dir / "results.csv"
            if csv_path.exists():
                results_path = csv_path
            else:
                # Try JSON if CSV not found
                json_path = self.result_dir / "results.json"
                if json_path.exists():
                    results_path = json_path
                else:
                    raise FileNotFoundError(f"No results file found in {self.result_dir}")
        
        results_path = Path(results_path)
        
        if results_path.suffix.lower() == ".csv":
            df = pd.read_csv(results_path)
        elif results_path.suffix.lower() == ".json":
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {results_path.suffix}")
        
        logger.info(f"Loaded results from {results_path} with {len(df)} entries")
        return df
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load evaluation configuration.
        
        Args:
            config_path: Path to config file (default: looks for config.json in result_dir)
            
        Returns:
            dict: Configuration dictionary
        """
        if config_path is None:
            config_path = self.result_dir / "config.json"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def generate_report(self, results_df: Optional[pd.DataFrame] = None) -> str:
        """
        Generate an HTML report with visualizations.
        
        Args:
            results_df: DataFrame containing results (if None, tries to load from file)
            
        Returns:
            str: Path to the generated HTML report
        """
        if results_df is None:
            results_df = self.load_results()
        
        config = self.load_config()
        
        # Generate HTML report
        html_parts = [self._html_header, self._report_title]
        
        # Add configuration summary
        html_parts.append(self._generate_config_summary(config))
        
        # Add high-level metrics and cards
        html_parts.append(self._generate_overview_section(results_df))
        
        # Add model comparison
        html_parts.append(self._generate_model_comparison(results_df))
        
        # Add prompt type comparison
        html_parts.append(self._generate_prompt_comparison(results_df))
        
        # Add interactive visualizations
        html_parts.append(self._generate_visualizations(results_df))
        
        # Close HTML
        html_parts.append(self._html_footer)
        
        # Combine all parts
        html_content = "\n".join(html_parts)
        
        # Write to file
        report_path = self.result_dir / "evaluation_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {report_path}")
        
        return str(report_path)
    
    def _generate_config_summary(self, config: Dict[str, Any]) -> str:
        """Generate HTML for configuration summary."""
        timestamp = config.get("timestamp", "N/A")
        models = ", ".join(config.get("models", ["N/A"]))
        prompt_types = ", ".join(config.get("prompt_types", ["N/A"]))
        num_questions = config.get("num_questions", "N/A")
        parallel = "Yes" if config.get("parallel", False) else "No"
        
        html = f"""
        <div class="container">
            <h2 class="text-xl font-semibold mb-4">Evaluation Configuration</h2>
            <table>
                <tr>
                    <th>Timestamp</th>
                    <td>{timestamp}</td>
                </tr>
                <tr>
                    <th>Models Evaluated</th>
                    <td>{models}</td>
                </tr>
                <tr>
                    <th>Prompt Types</th>
                    <td>{prompt_types}</td>
                </tr>
                <tr>
                    <th>Number of Questions</th>
                    <td>{num_questions}</td>
                </tr>
                <tr>
                    <th>Parallel Execution</th>
                    <td>{parallel}</td>
                </tr>
            </table>
        </div>
        """
        
        return html
    
    def _generate_overview_section(self, df: pd.DataFrame) -> str:
        """Generate high-level overview metrics."""
        # Calculate key metrics
        total_evaluations = len(df)
        total_models = df['model_name'].nunique()
        total_prompt_types = df['prompt_type'].nunique()
        avg_correctness = df['correctness_score'].mean() if 'correctness_score' in df.columns else 0
        avg_reasoning = df['reasoning_score'].mean() if 'reasoning_score' in df.columns else 0
        
        best_model = df.groupby('model_name')['total_score'].mean().idxmax() if 'total_score' in df.columns else "N/A"
        best_prompt = df.groupby('prompt_type')['total_score'].mean().idxmax() if 'total_score' in df.columns else "N/A"
        
        # Format as percentage
        avg_correctness_pct = f"{avg_correctness * 100:.1f}%"
        avg_reasoning_pct = f"{avg_reasoning * 100:.1f}%"
        
        html = f"""
        <div class="container">
            <h2 class="text-xl font-semibold mb-4">Evaluation Overview</h2>
            
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-title">Total Evaluations</div>
                    <div class="metric-value">{total_evaluations}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Models Evaluated</div>
                    <div class="metric-value">{total_models}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Prompt Types</div>
                    <div class="metric-value">{total_prompt_types}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Avg. Correctness</div>
                    <div class="metric-value">{avg_correctness_pct}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Avg. Reasoning</div>
                    <div class="metric-value">{avg_reasoning_pct}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Best Model</div>
                    <div class="metric-value">{best_model}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Best Prompt Type</div>
                    <div class="metric-value">{best_prompt}</div>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_model_comparison(self, df: pd.DataFrame) -> str:
        """Generate model comparison section."""
        # Create model comparison figure
        if 'total_score' in df.columns:
            fig = px.bar(
                df.groupby('model_name')['total_score'].mean().reset_index(),
                x='model_name',
                y='total_score',
                title='Model Performance Comparison',
                labels={'model_name': 'Model', 'total_score': 'Average Score'},
                color='model_name',
                template='plotly_white'
            )
            fig.update_layout(xaxis_title="Model", yaxis_title="Average Score")
            
            # Save plot
            model_comp_path = self.plots_dir / "model_comparison.html"
            fig.write_html(str(model_comp_path))
            
            html = f"""
            <div class="container">
                <h2 class="text-xl font-semibold mb-4">Model Comparison</h2>
                <div class="plot-container">
                    <iframe src="plots/model_comparison.html" width="100%" height="500px" frameborder="0"></iframe>
                </div>
            </div>
            """
        else:
            html = """
            <div class="container">
                <h2 class="text-xl font-semibold mb-4">Model Comparison</h2>
                <div class="highlight">No score data available for model comparison</div>
            </div>
            """
        
        return html
    
    def _generate_prompt_comparison(self, df: pd.DataFrame) -> str:
        """Generate prompt type comparison section."""
        # Create prompt comparison figure
        if 'total_score' in df.columns:
            # Group by model and prompt type
            prompt_comp_df = df.groupby(['model_name', 'prompt_type'])['total_score'].mean().reset_index()
            
            fig = px.bar(
                prompt_comp_df,
                x='prompt_type',
                y='total_score',
                color='model_name',
                barmode='group',
                title='Prompt Type Performance by Model',
                labels={
                    'prompt_type': 'Prompt Type', 
                    'total_score': 'Average Score',
                    'model_name': 'Model'
                },
                template='plotly_white'
            )
            fig.update_layout(xaxis_title="Prompt Type", yaxis_title="Average Score")
            
            # Save plot
            prompt_comp_path = self.plots_dir / "prompt_comparison.html"
            fig.write_html(str(prompt_comp_path))
            
            html = f"""
            <div class="container">
                <h2 class="text-xl font-semibold mb-4">Prompt Type Comparison</h2>
                <div class="plot-container">
                    <iframe src="plots/prompt_comparison.html" width="100%" height="500px" frameborder="0"></iframe>
                </div>
            </div>
            """
        else:
            html = """
            <div class="container">
                <h2 class="text-xl font-semibold mb-4">Prompt Type Comparison</h2>
                <div class="highlight">No score data available for prompt comparison</div>
            </div>
            """
        
        return html
    
    def _generate_visualizations(self, df: pd.DataFrame) -> str:
        """Generate interactive visualizations section."""
        html = """
        <div class="container">
            <h2 class="text-xl font-semibold mb-4">Interactive Visualizations</h2>
            
            <div class="tabs">
                <div class="tab active" onclick="switchTab('tab-radar', this)">Radar Chart</div>
                <div class="tab" onclick="switchTab('tab-heatmap', this)">Heatmap</div>
                <div class="tab" onclick="switchTab('tab-bubble', this)">Bubble Chart</div>
            </div>
            
            <div id="tab-radar" class="tab-content active">
        """
        
        # Radar chart for metrics by model
        metrics = ['correctness_score', 'reasoning_score', 'confidence_score']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if available_metrics:
            # Prepare data
            radar_df = df.groupby('model_name')[available_metrics].mean().reset_index()
            
            # Create radar chart
            fig = go.Figure()
            
            for i, model in enumerate(radar_df['model_name']):
                fig.add_trace(go.Scatterpolar(
                    r=radar_df.loc[i, available_metrics].values.tolist(),
                    theta=available_metrics,
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Model Performance Metrics",
                template="plotly_white"
            )
            
            # Save plot
            radar_path = self.plots_dir / "radar_chart.html"
            fig.write_html(str(radar_path))
            
            html += f"""
                <div class="plot-container">
                    <iframe src="plots/radar_chart.html" width="100%" height="500px" frameborder="0"></iframe>
                </div>
            """
        else:
            html += """
                <div class="highlight">No metric data available for radar chart</div>
            """
        
        html += """
            </div>
            
            <div id="tab-heatmap" class="tab-content">
        """
        
        # Heatmap for model-prompt performance
        if 'total_score' in df.columns:
            # Create pivot table
            pivot = df.pivot_table(
                values='total_score',
                index='model_name',
                columns='prompt_type',
                aggfunc='mean'
            ).round(3)
            
            # Create heatmap
            fig = px.imshow(
                pivot,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Prompt Type", y="Model", color="Score"),
                title="Model-Prompt Performance Heatmap"
            )
            
            # Save plot
            heatmap_path = self.plots_dir / "heatmap.html"
            fig.write_html(str(heatmap_path))
            
            html += f"""
                <div class="plot-container">
                    <iframe src="plots/heatmap.html" width="100%" height="500px" frameborder="0"></iframe>
                </div>
            """
        else:
            html += """
                <div class="highlight">No score data available for heatmap</div>
            """
        
        html += """
            </div>
            
            <div id="tab-bubble" class="tab-content">
        """
        
        # Bubble chart for correctness vs reasoning
        if all(m in df.columns for m in ['correctness_score', 'reasoning_score', 'response_length']):
            bubble_df = df.groupby(['model_name', 'prompt_type']).agg({
                'correctness_score': 'mean',
                'reasoning_score': 'mean',
                'response_length': 'mean',
                'question_id': 'count'
            }).reset_index()
            
            bubble_df.rename(columns={'question_id': 'count'}, inplace=True)
            
            fig = px.scatter(
                bubble_df,
                x='correctness_score',
                y='reasoning_score',
                size='response_length',
                color='model_name',
                symbol='prompt_type',
                hover_name='model_name',
                text='prompt_type',
                labels={
                    'correctness_score': 'Correctness Score',
                    'reasoning_score': 'Reasoning Score',
                    'response_length': 'Avg Response Length',
                    'model_name': 'Model',
                    'prompt_type': 'Prompt Type'
                },
                title="Correctness vs Reasoning by Model and Prompt Type",
                template="plotly_white"
            )
            
            # Save plot
            bubble_path = self.plots_dir / "bubble_chart.html"
            fig.write_html(str(bubble_path))
            
            html += f"""
                <div class="plot-container">
                    <iframe src="plots/bubble_chart.html" width="100%" height="500px" frameborder="0"></iframe>
                </div>
            """
        else:
            html += """
                <div class="highlight">Not enough metric data available for bubble chart</div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html 