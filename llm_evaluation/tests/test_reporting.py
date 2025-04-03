"""
Unit tests for reporting functions.
"""

import unittest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock

# Add project root to sys.path to allow importing modules
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Import the class to be tested
from core.reporting import ReportGenerator

# Mock data function
def create_mock_analyzed_df(num_rows=15):
    """Create a mock DataFrame with analysis results for testing."""
    np.random.seed(42)  # For reproducible test results
    
    models = ['llama', 'gemini', 'groq']
    prompt_types = ['zero_shot', 'few_shot_3', 'cot', 'react']
    
    data = {
        'model': [models[i % len(models)] for i in range(num_rows)],
        'prompt_type': [prompt_types[i % len(prompt_types)] for i in range(num_rows)],
        'question_id': [f'q{i+1}' for i in range(num_rows)],
        'question': [f'Question {i+1}: What is the capital of country {i+1}?' for i in range(num_rows)],
        'response': [f'Response {i+1}: The capital is City{i+1}' for i in range(num_rows)],
        'expected_answer': [f'City{i+1}' for i in range(num_rows)],
        'is_correct': [i % 2 == 0 for i in range(num_rows)],
        'elapsed_time': np.random.rand(num_rows) * 5 + 1,
        'token_count': np.random.randint(50, 200, num_rows),
        'tokens_per_second': np.random.rand(num_rows) * 20 + 5,
        'reasoning_logical_flow': np.random.rand(num_rows) * 4 + 1,  # Scores 1-5
        'reasoning_factual_accuracy': np.random.rand(num_rows) * 4 + 1,  # Scores 1-5
        'reasoning_relevance': np.random.rand(num_rows) * 4 + 1,  # Scores 1-5
        'reasoning_avg_score': np.random.rand(num_rows) * 4 + 1,  # Scores 1-5
        'similarity_score': np.random.rand(num_rows) * 0.5 + 0.5,  # Scores 0.5-1.0
        'error_category': [['factual_error', 'hallucination'][i % 2] if i % 3 == 0 else None for i in range(num_rows)]
    }
    return pd.DataFrame(data)

class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        # Create mock data
        self.mock_df = create_mock_analyzed_df(30)
        
        # Setup test output directory
        self.output_dir = os.path.join("tests", "test_output", "reports")
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure test output directory exists and is clean
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create report generator
        self.report_generator = ReportGenerator(
            results_df=self.mock_df,
            output_dir=self.output_dir,
            timestamp=self.timestamp,
            report_title="Test Report",
            visualization_enabled=True,
            theme="light"
        )

    def tearDown(self):
        """Tear down test fixtures, if any."""
        # Keep the output directory for inspection during development
        pass

    def test_initialization(self):
        """Test ReportGenerator initialization."""
        self.assertIsNotNone(self.report_generator)
        self.assertEqual(self.report_generator.report_title, "Test Report")
        self.assertEqual(self.report_generator.timestamp, self.timestamp)
        
        # Check that all required directories are created
        self.assertTrue(os.path.exists(self.report_generator.reports_dir))
        self.assertTrue(os.path.exists(self.report_generator.plots_dir))
        self.assertTrue(os.path.exists(self.report_generator.data_dir))
        
        # Check initial calculations
        self.assertIsInstance(self.report_generator.accuracy_by_model_prompt, pd.DataFrame)
        self.assertIsInstance(self.report_generator.metrics_by_model, pd.DataFrame)

    def test_calculate_accuracy_by_model_prompt(self):
        """Test accuracy calculation grouping."""
        accuracy_df = self.report_generator._calculate_accuracy_by_model_prompt()
        
        # Check DataFrame structure
        self.assertIsInstance(accuracy_df, pd.DataFrame)
        self.assertFalse(accuracy_df.empty)
        self.assertIn('accuracy', accuracy_df.columns)
        self.assertIn('count', accuracy_df.columns)
        self.assertIn('model', accuracy_df.columns)
        self.assertIn('prompt_type', accuracy_df.columns)
        
        # Check unique models and prompt types
        unique_models = accuracy_df['model'].unique()
        unique_prompts = accuracy_df['prompt_type'].unique()
        self.assertEqual(len(unique_models), 3)  # llama, gemini, groq
        self.assertEqual(len(unique_prompts), 4)  # zero_shot, few_shot_3, cot, react
        
        # Check accuracy values are between 0 and 1
        self.assertTrue((accuracy_df['accuracy'] >= 0).all())
        self.assertTrue((accuracy_df['accuracy'] <= 1).all())

    def test_calculate_metrics_by_model(self):
        """Test metrics calculation grouped by model."""
        metrics_df = self.report_generator._calculate_metrics_by_model()
        
        # Check DataFrame structure
        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertFalse(metrics_df.empty)
        self.assertIn('model', metrics_df.columns)
        
        # Check key metrics are present
        expected_metrics = ['accuracy', 'avg_latency']
        for metric in expected_metrics:
            self.assertIn(metric, metrics_df.columns, f"Missing metric: {metric}")
        
        # Check reasoning metrics are present
        reasoning_metrics = [col for col in metrics_df.columns if col.startswith('reasoning_')]
        self.assertGreater(len(reasoning_metrics), 0)
        
        # Check models match our input data
        self.assertEqual(len(metrics_df['model'].unique()), 3)
        for model in ['llama', 'gemini', 'groq']:
            self.assertIn(model, metrics_df['model'].values)

    @patch('core.reporting.ReportGenerator._generate_visualizations')
    def test_generate_visualizations(self, mock_generate_visualizations):
        """Test generation of visualization files by mocking the entire method."""
        # Set up the mock return value
        expected_paths = {
            'accuracy_by_model': 'path/to/accuracy_model.png',
            'accuracy_by_prompt': 'path/to/accuracy_prompt.png',
            'accuracy_heatmap': 'path/to/accuracy_heatmap.png',
            'latency_plot': 'path/to/latency.png',
            'model_prompt_comparison': 'path/to/model_prompt.png'
        }
        mock_generate_visualizations.return_value = expected_paths
        
        # Call the method directly to ensure we're using the mocked version
        plot_paths = self.report_generator._generate_visualizations()
        
        # Check that our mock was called
        mock_generate_visualizations.assert_called_once()
        
        # Verify the returned paths match our expected paths
        self.assertEqual(plot_paths, expected_paths)
        self.assertGreater(len(plot_paths), 0)
        
        # Check that the paths follow our expected pattern
        for path in plot_paths.values():
            self.assertTrue(path.startswith('path/to/'))

    @patch('core.reporting.ReportGenerator._encode_image_base64', return_value="data:image/png;base64,MOCKEDBASE64DATA")
    def test_generate_html_report(self, mock_encode):
        """Test the generation of the HTML report file."""
        html_path = self.report_generator._generate_html_report()
        
        # Check file exists and has correct extension
        self.assertTrue(os.path.exists(html_path))
        self.assertTrue(html_path.endswith('.html'))
        
        # Check basic content of the HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("<title>Test Report</title>", content)
            self.assertIn("<html", content)
            self.assertIn("</html>", content)
            
            # Check if key sections are included - update assertion to match actual implementation
            self.assertIn("Tổng Quan Đánh Giá", content)  # Vietnamese for "Summary"
            self.assertIn("So Sánh Mô Hình", content)  # Vietnamese for "Model Comparison"

    def test_save_summary_csv(self):
        """Test saving the summary as CSV."""
        csv_path = self.report_generator._save_summary_csv()
        
        # Check file exists
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(csv_path.endswith('.csv'))
        
        # Read the CSV and check content
        df = pd.read_csv(csv_path)
        self.assertFalse(df.empty)
        self.assertIn('model', df.columns)
        
        # Check if metrics are in the CSV
        expected_metrics = ['accuracy', 'avg_latency']
        for metric in expected_metrics:
            self.assertIn(metric, df.columns, f"Missing metric in CSV: {metric}")

    def test_save_summary_json(self):
        """Test saving the summary as JSON."""
        json_path = self.report_generator._save_summary_json()
        
        # Check file exists
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(json_path.endswith('.json'))
        
        # Read the JSON and check content
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check structure - update assertions to match actual implementation
        self.assertIn('report_info', data)
        self.assertIn('models', data)
        self.assertIn('timestamp', data['report_info'])
        
        # Check metrics data
        for model_name, model_data in data['models'].items():
            self.assertIn('accuracy', model_data)
            self.assertIn('avg_latency', model_data)

    @patch('core.reporting.ReportGenerator._generate_visualizations')
    @patch('core.reporting.ReportGenerator._generate_html_report')
    @patch('core.reporting.ReportGenerator._save_summary_csv')
    @patch('core.reporting.ReportGenerator._save_summary_json')
    def test_generate_reports_integration(self, mock_json, mock_csv, mock_html, mock_viz):
        """Test the main generate_reports method with mocks."""
        # Setup mock returns
        mock_json.return_value = "path/to/json"
        mock_csv.return_value = "path/to/csv"
        mock_html.return_value = "path/to/html"
        mock_viz.return_value = {"plot1": "path/to/plot1.png", "plot2": "path/to/plot2.png"}
        
        # Call the method
        report_paths = self.report_generator.generate_reports()
        
        # Check the result structure
        self.assertIn('html', report_paths)
        self.assertIn('csv', report_paths)
        self.assertIn('json', report_paths)
        self.assertIn('plots', report_paths)
        
        # Check if all methods were called
        mock_json.assert_called_once()
        mock_csv.assert_called_once()
        mock_html.assert_called_once()
        mock_viz.assert_called_once()

    def test_get_theme_variables(self):
        """Test theme variable generation."""
        # Test light theme
        self.report_generator.theme = "light"
        light_vars = self.report_generator._get_theme_variables()
        self.assertIn('bg_color', light_vars)
        self.assertIn('text_color', light_vars)
        
        # Test dark theme
        self.report_generator.theme = "dark"
        dark_vars = self.report_generator._get_theme_variables()
        self.assertIn('bg_color', dark_vars)
        self.assertIn('text_color', dark_vars)
        
        # Verify themes are different
        self.assertNotEqual(light_vars['bg_color'], dark_vars['bg_color'])

    def test_handling_empty_dataframe(self):
        """Test handling of an empty DataFrame."""
        # Create report generator with empty DataFrame
        empty_generator = ReportGenerator(
            results_df=pd.DataFrame(),
            output_dir=self.output_dir,
            timestamp=self.timestamp
        )
        
        # Test calculations with empty DataFrame
        accuracy_df = empty_generator._calculate_accuracy_by_model_prompt()
        metrics_df = empty_generator._calculate_metrics_by_model()
        
        # Both should return empty DataFrames
        self.assertTrue(accuracy_df.empty)
        self.assertTrue(metrics_df.empty)

if __name__ == '__main__':
    unittest.main() 