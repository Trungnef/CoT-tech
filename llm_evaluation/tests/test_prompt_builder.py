"""
Unit tests for the PromptBuilder class.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Import the PromptBuilder class
from core.prompt_builder import PromptBuilder


class TestPromptBuilder(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create both English and Vietnamese PromptBuilder instances for testing
        self.builder_en = PromptBuilder(
            system_message="You are a helpful assistant.",
            language="english"
        )
        
        self.builder_vi = PromptBuilder(
            system_message="Bạn là một trợ lý AI giỏi giải toán.",
            language="vietnamese"
        )
        
        # Example data for few-shot prompts
        self.examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"}
        ]
    
    def test_initialization(self):
        """Test the initialization of the PromptBuilder class."""
        # Test default initialization
        builder = PromptBuilder()
        self.assertEqual(builder.system_message, "Bạn là một trợ lý AI giỏi giải toán.")
        self.assertEqual(builder.language, "vietnamese")
        
        # Test custom initialization
        custom_builder = PromptBuilder(
            system_message="Custom system message",
            language="english"
        )
        self.assertEqual(custom_builder.system_message, "Custom system message")
        self.assertEqual(custom_builder.language, "english")
    
    def test_create_zero_shot_prompt(self):
        """Test the creation of a zero-shot prompt."""
        question = "What is the population of New York City?"
        prompt = self.builder_en.create_prompt(question, "zero_shot")
        
        # Check that the prompt contains the system message, question, and suffix
        self.assertIn(self.builder_en.system_message, prompt)
        self.assertIn(question, prompt)
        self.assertIn("Answer:", prompt)
    
    def test_create_few_shot_prompt(self):
        """Test the creation of a few-shot prompt."""
        question = "What is the largest planet in our solar system?"
        prompt = self.builder_en.create_prompt(question, "few_shot_2", self.examples)
        
        # Check that the prompt contains the system message and question
        self.assertIn(self.builder_en.system_message, prompt)
        self.assertIn("Problem: " + question, prompt)
        
        # Check for the expected structure rather than specific examples
        self.assertIn("Here are some examples", prompt)
        self.assertIn("Problem:", prompt)
        self.assertIn("Answer:", prompt)
        
        # Check that we have the right number of examples (2 examples + 1 question)
        self.assertEqual(prompt.count("Problem:"), 3)
        # Check that we have the right number of answers (2 examples)
        self.assertEqual(prompt.count("Answer:"), 3)  # 2 examples + final prompt
    
    def test_create_cot_prompt(self):
        """Test the creation of a chain-of-thought prompt."""
        question = "If John has 5 apples and gives 2 to Mary, how many does he have left?"
        prompt = self.builder_en.create_prompt(question, "cot")
        
        # Check that the prompt contains the system message, question, and CoT instruction
        self.assertIn(self.builder_en.system_message, prompt)
        self.assertIn(question, prompt)
        self.assertIn("step-by-step", prompt.lower())
    
    def test_few_shot_with_invalid_examples(self):
        """Test few-shot with insufficient examples raises a ValueError."""
        question = "What is the capital of Germany?"
        
        # Try to create a few-shot prompt with more examples than provided
        with self.assertRaises(ValueError):
            self.builder_en.create_prompt(question, "few_shot_5", self.examples)
    
    def test_invalid_prompt_type(self):
        """Test that an invalid prompt type raises a ValueError."""
        question = "What is the speed of light?"
        
        # Try to create a prompt with an invalid type
        with self.assertRaises(ValueError):
            self.builder_en.create_prompt(question, "invalid_prompt_type")
    
    def test_extract_final_answer_vietnamese(self):
        """Test extracting the final answer from a Vietnamese response."""
        # For Vietnamese responses with đáp án (answer)
        response_vi = "Tôi cần tính toán cẩn thận.\n\nĐáp án: 42"
        answer = self.builder_vi.extract_final_answer(response_vi, "zero_shot")
        self.assertEqual(answer.strip(), "42")
    
    def test_extract_final_answer_cot_vietnamese(self):
        """Test extracting the final answer from a Vietnamese chain-of-thought response."""
        response_vi = """
        Hãy suy nghĩ từng bước.
        Đầu tiên, John có 5 quả táo.
        Sau đó, John cho Mary 2 quả.
        Vậy John còn lại 5 - 2 = 3 quả.
        Vậy đáp án là: 3 quả táo.
        """
        answer = self.builder_vi.extract_final_answer(response_vi, "cot")
        self.assertIn("3", answer)
    
    def test_english_response_default_return(self):
        """Test that English responses without Vietnamese patterns return the whole response by default."""
        # For English responses, the method should return the whole response
        response_en = "The answer is 42."
        answer = self.builder_vi.extract_final_answer(response_en, "zero_shot")
        self.assertEqual(answer, response_en)


if __name__ == '__main__':
    unittest.main() 