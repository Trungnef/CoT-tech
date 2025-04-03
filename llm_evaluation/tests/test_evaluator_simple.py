"""
Simple unit tests for the Evaluator class.
"""

import unittest
import os
import sys
import json
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path to allow importing modules
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Create mock config module
mock_config = MagicMock()
mock_config.MAX_CHECKPOINTS = 5
mock_config.MODEL_CONFIGS = {
    "model1": {"temperature": 0.7},
    "model2": {"temperature": 0.5}
}
mock_config.RESULTS_DIR = "results"
mock_config.REASONING_EVALUATION_CONFIG = {
    "enabled": False
}

# Mock other required modules
sys.modules['config'] = mock_config
sys.modules['core.prompt_builder'] = MagicMock()
sys.modules['core.prompt_builder'].create_prompt = MagicMock(return_value="Mock prompt")
sys.modules['core.checkpoint_manager'] = MagicMock()
sys.modules['core.result_analyzer'] = MagicMock()
sys.modules['core.reporting'] = MagicMock()
sys.modules['utils.logging_setup'] = MagicMock()
sys.modules['utils.logging_setup'].get_logger = MagicMock(return_value=MagicMock())
sys.modules['utils.logging_setup'].log_evaluation_start = MagicMock()
sys.modules['utils.logging_setup'].log_evaluation_progress = MagicMock()
sys.modules['utils.logging_setup'].log_evaluation_complete = MagicMock()
sys.modules['utils.logging_setup'].log_api_error = MagicMock()
sys.modules['utils.logging_setup'].log_checkpoint = MagicMock()
sys.modules['utils.logging_setup'].log_checkpoint_resume = MagicMock()
sys.modules['utils.logging_setup'].log_section = MagicMock()

# Now we can import the Evaluator without import errors
from core.evaluator import Evaluator


class TestEvaluatorSimple(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for results
        self.results_dir = Path(__file__).parent / "test_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create some mock data for testing
        self.models = ["model1", "model2"]
        self.prompts = ["zero-shot", "few-shot"]
        self.questions = [
            {"id": "q1", "question": "What is 2+2?", "answer": "4"},
            {"id": "q2", "question": "Who was the first president of the US?", "answer": "George Washington"}
        ]
        
        # Save questions to a temporary file
        self.questions_file = self.results_dir / "test_questions.json"
        with open(self.questions_file, 'w') as f:
            json.dump(self.questions, f)
    
    def tearDown(self):
        """Clean up after tests."""
        # You can uncomment this if you want to clean up after tests
        # import shutil
        # shutil.rmtree(self.results_dir)
        pass

    @patch('core.evaluator.ModelInterface')
    @patch('core.evaluator.ResultAnalyzer')
    @patch('core.evaluator.CheckpointManager')
    def test_basic_initialization(self, mock_checkpoint_manager, mock_result_analyzer, mock_model_interface):
        """Test that Evaluator can be initialized with basic properties."""
        
        # Create evaluator instance with minimal parameters
        evaluator = Evaluator(
            models_to_evaluate=self.models,
            prompts_to_evaluate=self.prompts,
            questions=self.questions,
            results_dir=str(self.results_dir)
        )
        
        # Assert correct initialization of basic properties
        self.assertEqual(evaluator.models, self.models)
        self.assertEqual(evaluator.prompts, self.prompts)
        self.assertEqual(evaluator.questions, self.questions)
        self.assertEqual(evaluator.results_dir, str(self.results_dir))
        
        # Check that model_interface was initialized
        self.assertIsNotNone(evaluator.model_interface)
        
        # Check that checkpoint_manager was initialized
        self.assertIsNotNone(evaluator.checkpoint_manager)
        
        # Check initial state
        self.assertEqual(evaluator.results, [])
        self.assertEqual(evaluator.completed_combinations, set())
    
    @patch('core.evaluator.ModelInterface')
    @patch('core.evaluator.CheckpointManager')
    def test_evaluate_single_combination(self, mock_checkpoint_manager, mock_model_interface):
        """Test evaluating a single model-prompt-question combination."""
        # Set up the mock model_interface
        mock_instance = mock_model_interface.return_value
        mock_instance.generate_text.return_value = ("4", {
            "token_count": 10,
            "tokens_per_second": 5.0,
            "elapsed_time": 2.0
        })
        
        # Create evaluator instance
        evaluator = Evaluator(
            models_to_evaluate=self.models,
            prompts_to_evaluate=self.prompts,
            questions=self.questions,
            results_dir=str(self.results_dir)
        )
        
        # Evaluate a single combination
        question = self.questions[0]  # "What is 2+2?"
        result = evaluator._evaluate_single_combination("model1", "zero-shot", question, 0)
        
        # Check result structure
        self.assertIsNotNone(result)
        self.assertEqual(result.get('model_name'), "model1")
        self.assertEqual(result.get('prompt_type'), "zero-shot")
        self.assertEqual(result.get('question_id'), "q1")
        self.assertEqual(result.get('question_text'), "What is 2+2?")
        self.assertEqual(result.get('prompt'), "Mock prompt")  # From our mock
        self.assertEqual(result.get('response'), "4")  # From our mock
        
        # Verify the mock was called correctly
        mock_instance.generate_text.assert_called_once()
    
    @patch('core.evaluator.ModelInterface')
    @patch('core.evaluator.CheckpointManager')
    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.DataFrame.to_json')
    def test_save_results(self, mock_to_json, mock_to_csv, mock_checkpoint_manager, mock_model_interface):
        """Test saving evaluation results."""
        # Create evaluator instance
        evaluator = Evaluator(
            models_to_evaluate=self.models,
            prompts_to_evaluate=self.prompts,
            questions=self.questions,
            results_dir=str(self.results_dir)
        )
        
        # Add some test results
        evaluator.results = [
            {
                'model_name': 'model1',
                'prompt_type': 'zero-shot',
                'question_id': 'q1',
                'question_text': 'What is 2+2?',
                'prompt': 'Mock prompt',
                'response': '4',
                'expected_answer': '4',
                'is_correct': True
            },
            {
                'model_name': 'model2',
                'prompt_type': 'few-shot',
                'question_id': 'q2',
                'question_text': 'Who was the first president of the US?',
                'prompt': 'Mock prompt',
                'response': 'George Washington',
                'expected_answer': 'George Washington',
                'is_correct': True
            }
        ]
        
        # Save results
        evaluator._save_results()
        
        # Check that to_csv and to_json were called
        mock_to_csv.assert_called_once()
        mock_to_json.assert_called_once()
    
    @patch('core.evaluator.ModelInterface')
    @patch('core.evaluator.CheckpointManager')
    def test_save_checkpoint(self, mock_checkpoint_manager, mock_model_interface):
        """Test saving a checkpoint during evaluation."""
        # Set up the mock checkpoint_manager
        mock_checkpoint_instance = mock_checkpoint_manager.return_value
        mock_checkpoint_instance.save_checkpoint.return_value = "path/to/checkpoint.json"
        
        # Create evaluator instance
        evaluator = Evaluator(
            models_to_evaluate=self.models,
            prompts_to_evaluate=self.prompts,
            questions=self.questions,
            results_dir=str(self.results_dir)
        )
        
        # Add some test state
        evaluator.current_model = "model1"
        evaluator.current_prompt = "zero-shot"
        evaluator.current_question_idx = 1
        evaluator.results = [{'model_name': 'model1', 'prompt_type': 'zero-shot', 'question_id': 'q1'}]
        evaluator.completed_combinations = {('model1', 'zero-shot', 'q1')}
        
        # Save checkpoint
        checkpoint_path = evaluator.save_checkpoint()
        
        # Check checkpoint was saved
        self.assertEqual(checkpoint_path, "path/to/checkpoint.json")
        
        # Verify checkpoint_manager was called correctly
        mock_checkpoint_instance.save_checkpoint.assert_called_once()
        # The call should include the state
        args = mock_checkpoint_instance.save_checkpoint.call_args[0][0]
        self.assertEqual(args['current_model'], "model1")
        self.assertEqual(args['current_prompt'], "zero-shot")
        self.assertEqual(args['current_question_idx'], 1)
        self.assertEqual(len(args['results']), 1)
        self.assertEqual(len(args['completed_combinations']), 1)


if __name__ == '__main__':
    unittest.main() 