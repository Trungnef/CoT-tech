"""
Unit tests for the Evaluator class.
"""

import unittest
import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path to allow importing modules
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Import the Evaluator class
from core.evaluator import Evaluator

class TestEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for results
        self.results_dir = Path(__file__).parent / "test_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create some mock data for testing
        self.models = ["model1", "model2"]
        self.prompt_types = ["zero-shot", "few-shot"]
        self.questions = [
            {"id": "q1", "question": "What is 2+2?", "answer": "4"},
            {"id": "q2", "question": "Who was the first president of the US?", "answer": "George Washington"}
        ]
        
        # Create a mock config for the evaluator
        self.config = {
            "models": self.models,
            "prompt_types": self.prompt_types,
            "results_dir": str(self.results_dir),
            "metrics": ["accuracy"],
            "num_evals": 1,
            "logging_level": "ERROR"
        }
        
        # Save questions to a temporary file
        self.questions_file = self.results_dir / "test_questions.json"
        with open(self.questions_file, 'w') as f:
            json.dump(self.questions, f)
    
    def tearDown(self):
        """Clean up after tests."""
        # Comment out to retain results for inspection
        # import shutil
        # shutil.rmtree(self.results_dir)
        pass

    @patch('core.evaluator.ModelInterface')
    @patch('core.evaluator.ResultAnalyzer')
    @patch('core.evaluator.create_prompt')
    def test_init(self, mock_create_prompt, mock_analyzer, mock_model_interface):
        """Test Evaluator initialization."""
        # Set up mocks
        mock_model_interface.return_value.generate_text.return_value = "4"
        mock_analyzer.return_value.analyze_results.return_value = {"accuracy": 1.0}
        mock_create_prompt.return_value = "What is 2+2?"
        
        # Create evaluator instance
        evaluator = Evaluator(
            models=self.models,
            prompt_types=self.prompt_types,
            questions_file=str(self.questions_file),
            results_dir=str(self.results_dir),
            metrics=["accuracy"],
            num_evals=1
        )
        
        # Assert correct initialization
        self.assertEqual(evaluator.models, self.models)
        self.assertEqual(evaluator.prompt_types, self.prompt_types)
        self.assertEqual(evaluator.results_dir, str(self.results_dir))
        self.assertIsNotNone(evaluator.logger)
        
    @patch('core.evaluator.ModelInterface')
    @patch('core.evaluator.ResultAnalyzer')
    @patch('core.evaluator.create_prompt')
    def test_evaluate_single_combination(self, mock_create_prompt, mock_analyzer, mock_model_interface):
        """Test evaluating a single model-prompt-question combination."""
        # Set up mocks
        mock_instance = mock_model_interface.return_value
        mock_instance.generate_text.return_value = "4"
        mock_create_prompt.return_value = "What is 2+2?"
        
        # Create evaluator instance
        evaluator = Evaluator(
            models=self.models,
            prompt_types=self.prompt_types,
            questions_file=str(self.questions_file),
            results_dir=str(self.results_dir),
            metrics=["accuracy"],
            num_evals=1
        )
        
        # Manually set the questions to avoid loading from file
        evaluator.questions = self.questions
        
        # Call the evaluate method for a single combination
        result = evaluator._evaluate_single_combination(
            model="model1",
            prompt_type="zero-shot",
            question=self.questions[0]
        )
        
        # Check result structure
        self.assertIn("model", result)
        self.assertIn("prompt_type", result)
        self.assertIn("question_id", result)
        self.assertIn("question", result)
        self.assertIn("correct_answer", result)
        self.assertIn("response", result)
        
        # Verify the mocks were called as expected
        mock_create_prompt.assert_called_once()
        mock_instance.generate_text.assert_called_once()

if __name__ == '__main__':
    unittest.main() 