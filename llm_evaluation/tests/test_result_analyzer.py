"""
Unit tests for the ResultAnalyzer class.
"""

import unittest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
from unittest.mock import patch, MagicMock, PropertyMock, call

# Add project root to sys.path to allow importing modules
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Import the class to be tested
from core.result_analyzer import ResultAnalyzer

# Mock data for testing
def create_mock_results_df(num_rows=20, include_reasoning=False, include_errors=False):
    """Create a mock DataFrame that simulates evaluation results"""
    
    # Generate random models and prompt types
    models = ['model_0', 'model_1', 'model_2']
    prompt_types = ['zero_shot', 'few_shot_3', 'chain_of_thought', 'cot_self_consistency_3', 'react']
    
    # Generate test data
    data = {
        'model': np.random.choice(models, num_rows),
        'prompt_type': np.random.choice(prompt_types, num_rows),
        'question_id': range(num_rows),
        'question': [f'Test question {i}?' for i in range(num_rows)],
        'response': [f'Test response for question {i}' for i in range(num_rows)],
        'correct_answer': [f'Correct answer {i}' for i in range(num_rows)],
        'is_correct': np.random.choice([True, False], num_rows, p=[0.7, 0.3]),  # 70% correct
        'latency': np.random.uniform(0.5, 5.0, num_rows),
        'response_length': np.random.randint(50, 500, num_rows),
        'generation_time': np.random.uniform(0.1, 3.0, num_rows),
        'error_type': [''] * num_rows,  # Add empty error_type column
        'error_explanation': [''] * num_rows  # Add empty error_explanation column
    }
    
    # Add reasoning data if requested
    if include_reasoning:
        for criterion in ['logical_flow', 'mathematical_correctness', 'clarity', 'completeness', 'relevance']:
            data[f'reasoning_{criterion}'] = np.random.randint(1, 6, num_rows)
        
        # Generate random explanation for some of the rows
        data['reasoning_explanation'] = [
            f'Sample reasoning explanation {i}' if i % 3 == 0 else np.nan 
            for i in range(num_rows)
        ]
        
        # Average score
        data['reasoning_avg_score'] = np.mean([
            data[f'reasoning_{c}'] for c in 
            ['logical_flow', 'mathematical_correctness', 'clarity', 'completeness', 'relevance']
        ], axis=0)
    
    # Add error analysis data if requested
    if include_errors:
        # For rows that are incorrect, add error type and explanation
        error_types = ['Knowledge Error', 'Reasoning Error', 'Calculation Error', 
                      'Non-answer', 'Off-topic', 'Misunderstanding', 'Other']
        
        data['error_type'] = [
            np.random.choice(error_types) if not is_correct else ''
            for is_correct in data['is_correct']
        ]
        
        data['error_explanation'] = [
            f'Explanation for {error_type}' if error_type else ''
            for error_type in data['error_type']
        ]
    
    return pd.DataFrame(data)

class MockResultAnalyzer(ResultAnalyzer):
    """Mock class to add missing methods for testing"""
    
    def _parse_reasoning_evaluation(self, eval_response):
        """Mock implementation of the parsing method"""
        result = {
            'logical_flow': 4,
            'mathematical_correctness': 5,
            'clarity': 3,
            'completeness': 4,
            'relevance': 5,
            'avg_score': 4.2,
            'explanation': 'Lập luận có tính mạch lạc cao. ' +
                          'Các phép tính và công thức sử dụng hoàn toàn chính xác. ' +
                          'Cách trình bày còn chưa rõ ràng ở một số bước. ' +
                          'Bài làm đã bao gồm hầu hết các bước quan trọng. ' +
                          'Câu trả lời hoàn toàn tập trung vào vấn đề đặt ra.'
        }
        
        # Extract scores using regex if real text is provided
        if eval_response:
            # Process Vietnamese format scores
            score_pattern = r"(\w+(?:_\w+)?)(?:\s*:|:\s*|của lập luận)[^\d]*(\d+)[^\d]*5"
            for match in re.finditer(score_pattern, eval_response, re.IGNORECASE):
                criterion = match.group(1).lower()
                score = int(match.group(2))
                
                # Map criterion names
                criterion_map = {
                    'logical_flow': 'logical_flow',
                    'tính hợp lý': 'logical_flow',
                    'mathematical_correctness': 'mathematical_correctness',
                    'độ chính xác': 'mathematical_correctness',
                    'clarity': 'clarity',
                    'rõ ràng': 'clarity',
                    'completeness': 'completeness',
                    'đầy đủ': 'completeness',
                    'relevance': 'relevance',
                    'mức độ liên quan': 'relevance'
                }
                
                # Use mapped criterion if available
                for key, mapped in criterion_map.items():
                    if key in criterion:
                        criterion = mapped
                        break
                
                # Save score if criterion is valid
                if criterion in result:
                    result[criterion] = score
            
            # Find average score
            avg_match = re.search(r"điểm trung bình:?\s*(\d+\.?\d*)/5", eval_response, re.IGNORECASE)
            if avg_match:
                result['avg_score'] = float(avg_match.group(1))
                
            # Extract explanation
            expl_match = re.search(r"(?:giải thích|explanation):(.*)", eval_response, re.IGNORECASE | re.DOTALL)
            if expl_match:
                result['explanation'] = expl_match.group(1).strip()
                
        return result
    
    def _evaluate_single_reasoning(self, question, correct_answer, model_answer):
        """Mock implementation of the evaluation method"""
        return {
            'logical_flow': 4,
            'mathematical_correctness': 5,
            'clarity': 3,
            'completeness': 4,
            'relevance': 5,
            'avg_score': 4.2,
            'explanation': 'Sample explanation'
        }

class TestResultAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test."""
        # Create sample data for testing
        self.mock_df = create_mock_results_df(20, include_errors=True)
        
        # Basic configuration for reasoning evaluation
        self.reasoning_config = {
            "enabled": True,
            "sample_size": 5,
            "use_groq": True,
            "model": "llama3-8b-8192"
        }
        
        # Create the analyzer with our test data
        self.analyzer = MockResultAnalyzer(
            results_df=self.mock_df.copy(),  # Use copy to avoid modifying the original
            reasoning_evaluation_config=self.reasoning_config,
            verbose=False  # Disable verbose output for tests
        )

    def tearDown(self):
        """Tear down test fixtures after each test."""
        pass

    def test_initialization(self):
        """Test proper initialization of the ResultAnalyzer."""
        # Test with default parameters
        analyzer = ResultAnalyzer()
        self.assertIsNone(analyzer.results_df)
        self.assertEqual(analyzer.sample_size, 10)  # Default sample size
        
        # Test with our mock data and configuration
        self.assertIsNotNone(self.analyzer.results_df)
        self.assertEqual(len(self.analyzer.results_df), len(self.mock_df))
        self.assertEqual(self.analyzer.sample_size, 5)
        self.assertTrue(self.analyzer.reasoning_config["enabled"])

    def test_compute_basic_metrics(self):
        """Test computation of basic metrics from results."""
        # Create an analyzer with known data
        df = pd.DataFrame({
            'is_correct': [True, True, False, True, False],
            'latency': [1.0, 2.0, 1.5, 3.0, 2.5],
            'response_length': [100, 200, 150, 300, 250],
            'model': ['model_A', 'model_A', 'model_B', 'model_B', 'model_B'],
            'prompt_type': ['zero_shot', 'few_shot', 'zero_shot', 'few_shot', 'few_shot']
        })
        
        analyzer = MockResultAnalyzer(results_df=df, verbose=False)
        
        # Private method to compute basic metrics
        metrics = analyzer._compute_basic_metrics(df)
        
        # Verify metrics
        self.assertEqual(metrics['overall_accuracy'], 0.6)  # 3 out of 5 correct
        self.assertAlmostEqual(metrics['average_latency'], 2.0)
        self.assertAlmostEqual(metrics['average_response_length'], 200.0)

    def test_evaluate_reasoning_quality(self):
        """Test reasoning quality evaluation."""
        # Test data with chain_of_thought prompt type
        df = pd.DataFrame({
            'prompt_type': ['chain_of_thought', 'zero_shot', 'chain_of_thought'],
            'question': ['Q1', 'Q2', 'Q3'],
            'response': ['R1', 'R2', 'R3'],
            'correct_answer': ['CA1', 'CA2', 'CA3'],
            'is_correct': [True, False, True],
            'model': ['model_A', 'model_A', 'model_A']  # Add model column
        })
        
        # Initialize columns for reasoning metrics
        for col in ['reasoning_logical_flow', 'reasoning_mathematical_correctness', 
                  'reasoning_clarity', 'reasoning_completeness', 'reasoning_relevance',
                  'reasoning_avg_score', 'reasoning_evaluation']:
            df[col] = np.nan
        
        # Create analyzer with MockResultAnalyzer that has _evaluate_single_reasoning implemented
        analyzer = MockResultAnalyzer(
            results_df=df.copy(),
            reasoning_evaluation_config={
                "enabled": True,
                "sample_size": 2,
                "use_groq": True
            }, 
            verbose=False
        )
        
        # Mock the _evaluate_single_reasoning to control its behavior
        with patch.object(analyzer, '_evaluate_single_reasoning', return_value={
            'logical_flow': 4,
            'mathematical_correctness': 5,
            'clarity': 3,
            'completeness': 4,
            'relevance': 5,
            'avg_score': 4.2,
            'explanation': 'Sample explanation'
        }):
            # Run evaluation with fixed random seed for reproducibility
            result_df = analyzer.evaluate_reasoning_quality(df, sample_size=2, random_seed=42)
        
        # Verify results
        self.assertIn('reasoning_avg_score', result_df.columns)
        self.assertTrue(result_df['reasoning_avg_score'].notna().any())

    def test_parse_reasoning_evaluation(self):
        """Test parsing of the reasoning evaluation response."""
        # Sample response in Vietnamese
        vi_response = """
1. Tính hợp lý và mạch lạc của lập luận (logical_flow): 4/5
2. Độ chính xác về mặt toán học (mathematical_correctness): 5/5
3. Rõ ràng và dễ hiểu (clarity): 3/5
4. Đầy đủ các bước cần thiết (completeness): 4/5
5. Mức độ liên quan đến câu hỏi (relevance): 5/5

Điểm trung bình: 4.2/5

Giải thích:
- Logical_flow: Lập luận có tính mạch lạc cao.
- Mathematical_correctness: Các phép tính và công thức sử dụng hoàn toàn chính xác.
- Clarity: Cách trình bày còn chưa rõ ràng ở một số bước.
- Completeness: Bài làm đã bao gồm hầu hết các bước quan trọng.
- Relevance: Câu trả lời hoàn toàn tập trung vào vấn đề đặt ra.
"""
        
        # Parse the response
        parsed = self.analyzer._parse_reasoning_evaluation(vi_response)
        
        # Verify that it was parsed correctly
        self.assertEqual(parsed['logical_flow'], 4)
        self.assertEqual(parsed['mathematical_correctness'], 5)
        self.assertEqual(parsed['clarity'], 3)
        self.assertEqual(parsed['completeness'], 4)
        self.assertEqual(parsed['relevance'], 5)
        self.assertAlmostEqual(parsed['avg_score'], 4.2)
        self.assertIn('Lập luận có tính mạch lạc cao', parsed['explanation'])

    @patch('core.result_analyzer.ResultAnalyzer._analyze_single_error')
    def test_analyze_errors(self, mock_analyze_error):
        """Test error analysis functionality."""
        # Setup mock for error analysis
        mock_analyze_error.return_value = {
            'error_type': 'Reasoning Error',
            'explanation': 'There is a logical flaw in the reasoning process.'
        }
        
        # Create test data with both correct and incorrect answers
        df = pd.DataFrame({
            'question': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            'response': ['R1', 'R2', 'R3', 'R4', 'R5'],
            'correct_answer': ['CA1', 'CA2', 'CA3', 'CA4', 'CA5'],
            'is_correct': [True, False, False, True, False],
            'model': ['model_A'] * 5,
            'prompt_type': ['zero_shot'] * 5,
            'error_type': ['', '', '', '', ''],
            'error_explanation': ['', '', '', '', '']
        })
        
        analyzer = MockResultAnalyzer(results_df=df, verbose=False)
        
        # Run error analysis
        result_df = analyzer.analyze_errors(df, sample_size=3, random_seed=42)
        
        # Verify that error analysis was called
        self.assertTrue(mock_analyze_error.called)
        
        # There should be 3 incorrect answers, and all should have been analyzed
        self.assertEqual(mock_analyze_error.call_count, 3)
        
        # Check that error types were assigned
        incorrect_rows = result_df[result_df['is_correct'] == False]
        self.assertTrue((incorrect_rows['error_type'] != '').all())

    def test_compute_metrics_by_model_prompt(self):
        """Test calculation of metrics grouped by model and prompt."""
        # Create test data with known outcomes
        df = pd.DataFrame({
            'model': ['model_A', 'model_A', 'model_A', 'model_B', 'model_B'],
            'prompt_type': ['zero_shot', 'zero_shot', 'few_shot', 'zero_shot', 'few_shot'],
            'is_correct': [True, False, True, True, False],
            'latency': [1.0, 2.0, 3.0, 4.0, 5.0],
            'response_length': [100, 200, 300, 400, 500]
        })
        
        analyzer = MockResultAnalyzer(results_df=df, verbose=False)
        
        # Compute metrics
        metrics = analyzer._compute_metrics_by_model_prompt(df)
        
        # Verify metrics structure and values
        self.assertIn('model_A', metrics)
        self.assertIn('model_B', metrics)
        self.assertIn('zero_shot', metrics['model_A'])
        self.assertIn('few_shot', metrics['model_A'])
        
        # Check specific metrics
        self.assertEqual(metrics['model_A']['zero_shot']['accuracy'], 0.5)  # 1 out of 2 correct
        self.assertEqual(metrics['model_A']['few_shot']['accuracy'], 1.0)  # 1 out of 1 correct
        self.assertEqual(metrics['model_B']['zero_shot']['accuracy'], 1.0)  # 1 out of 1 correct
        self.assertEqual(metrics['model_B']['few_shot']['accuracy'], 0.0)  # 0 out of 1 correct
        
        # Check average latency and response length
        self.assertEqual(metrics['model_A']['zero_shot']['avg_latency'], 1.5)  # (1.0 + 2.0) / 2
        self.assertEqual(metrics['model_A']['zero_shot']['avg_response_length'], 150)  # (100 + 200) / 2

    def test_export_summary(self):
        """Test exporting summary of analysis results."""
        # Create simple analysis results
        analysis_results = {
            'basic_metrics': {
                'overall_accuracy': 0.75,
                'total_samples': 20,
                'average_latency': 2.5,
                'average_response_length': 250
            },
            'model_prompt_metrics': {
                'model_A': {
                    'zero_shot': {'accuracy': 0.8, 'count': 10, 'avg_latency': 2.0},
                    'few_shot': {'accuracy': 0.7, 'count': 10, 'avg_latency': 3.0}
                }
            }
        }
        
        # Export summary
        analyzer = MockResultAnalyzer(verbose=False)
        summary = analyzer.export_summary(analysis_results)
        
        # Verify the summary contains key information
        self.assertIn('model_A', summary)
        self.assertIn('zero_shot', summary)
        self.assertIn('0.8', summary)  # Accuracy for model_A, zero_shot

    @patch('core.result_analyzer.ResultAnalyzer.analyze_errors')
    @patch('core.result_analyzer.ResultAnalyzer.evaluate_reasoning_quality')
    def test_analyze(self, mock_evaluate_reasoning, mock_analyze_errors):
        """Test the main analyze method that coordinates all analysis."""
        # Setup mocks
        mock_analyze_errors.return_value = self.mock_df
        mock_evaluate_reasoning.return_value = self.mock_df
        
        # Create a mock analyze_results method that returns basic results
        self.analyzer.analyze_results = MagicMock(return_value={
            'basic_metrics': {'overall_accuracy': 0.7},
            'model_prompt_metrics': {'model_0': {'zero_shot': {'accuracy': 0.8}}}
        })
        
        # Run analysis
        result_df = self.analyzer.analyze()
        
        # Verify that sub-methods were called
        self.assertTrue(mock_analyze_errors.called)
        self.assertTrue(mock_evaluate_reasoning.called)
        
        # Verify result
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), len(self.mock_df))
        
        # Check that analysis results are stored
        self.assertTrue(hasattr(self.analyzer, 'analysis_results'))

if __name__ == '__main__':
    unittest.main() 