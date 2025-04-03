"""
Direct patching tests for the ModelInterface class.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Import ModelInterface
from core.model_interface import ModelInterface


class TestModelInterfaceDirect(unittest.TestCase):
    
    def test_generate_text_unsupported_model(self):
        """Test that using an unsupported model returns an error."""
        # Create a patch for the config object that will be used inside generate_text
        with patch('core.model_interface.config', autospec=True) as mock_config:
            # Set up mock config attributes
            mock_config.MODEL_CONFIGS = {
                "llama": {"temperature": 0.7},
                "gemini": {"temperature": 0.5}
            }
            
            # Create ModelInterface instance
            interface = ModelInterface()
            
            # Try to generate text with an unsupported model
            response, stats = interface.generate_text("unsupported_model", "Test prompt")
            
            # Check that it returns an error
            self.assertTrue(response.startswith("[Error:"))
            self.assertTrue(stats["has_error"])
            self.assertIn("Model không được hỗ trợ", stats["error_message"])
    
    def test_generate_text_with_local_model(self):
        """Test generate_text with local model by patching _generate_with_local_model."""
        # Create patches for required objects
        with patch('core.model_interface.config', autospec=True) as mock_config, \
             patch.object(ModelInterface, '_generate_with_local_model') as mock_generate_local:
            
            # Set up mock config
            mock_config.MODEL_CONFIGS = {
                "llama": {"temperature": 0.7},
                "gemini": {"temperature": 0.5}
            }
            
            # Set up mock return value
            mock_generate_local.return_value = ("Generated text", {"token_count": 10})
            
            # Create ModelInterface instance
            interface = ModelInterface()
            
            # Call generate_text with a local model
            response, stats = interface.generate_text("llama", "Test prompt")
            
            # Check the response
            self.assertEqual(response, "Generated text")
            self.assertEqual(stats["token_count"], 10)
            
            # Verify the mock was called correctly
            mock_generate_local.assert_called_once()
    
    def test_generate_text_with_gemini(self):
        """Test generate_text with Gemini by patching _generate_with_gemini."""
        # Create patches for required objects
        with patch('core.model_interface.config', autospec=True) as mock_config, \
             patch.object(ModelInterface, '_generate_with_gemini') as mock_generate_gemini:
            
            # Set up mock config
            mock_config.MODEL_CONFIGS = {
                "llama": {"temperature": 0.7},
                "gemini": {"temperature": 0.5}
            }
            
            # Set up mock return value
            mock_generate_gemini.return_value = ("Gemini response", {"token_count": 15})
            
            # Create ModelInterface instance
            interface = ModelInterface()
            
            # Call generate_text with Gemini
            response, stats = interface.generate_text("gemini", "Test prompt")
            
            # Check the response
            self.assertEqual(response, "Gemini response")
            self.assertEqual(stats["token_count"], 15)
            
            # Verify the mock was called correctly
            mock_generate_gemini.assert_called_once()


if __name__ == '__main__':
    unittest.main() 