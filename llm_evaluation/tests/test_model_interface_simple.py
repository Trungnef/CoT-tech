"""
Simple unit tests for the ModelInterface class.
"""

import unittest
import os
import sys
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path to allow importing modules
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Mock the required modules before importing ModelInterface
with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'huggingface_hub': MagicMock(),
    'google.generativeai': MagicMock(),
    'tenacity': MagicMock(),
    'groq': MagicMock()
}):
    # Import after mocking dependencies
    from core.model_interface import ModelInterface


class TestModelInterfaceSimple(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test cache directory
        self.cache_dir = Path(__file__).parent / "test_model_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define mock config values
        self.model_configs = {
            "llama": {"temperature": 0.7, "max_tokens": 512},
            "gemini": {"temperature": 0.5, "max_tokens": 1024},
            "groq": {"temperature": 0.7}
        }
        self.api_configs = {
            "gemini": {"requests_per_minute": 60},
            "groq": {"requests_per_minute": 120}
        }
        self.disk_cache_config = {
            "enabled": False,
            "cache_dir": str(self.cache_dir),
            "max_size_gb": 10,
            "cleanup_on_startup": False
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Uncomment to clean up the cache directory
        # import shutil
        # shutil.rmtree(self.cache_dir)
        pass
    
    @patch('core.model_interface.config')
    def test_initialization(self, mock_config):
        """Test basic initialization of the ModelInterface."""
        # Configure mock
        mock_config.DISK_CACHE_CONFIG = self.disk_cache_config
        
        # Create a ModelInterface instance
        interface = ModelInterface(use_cache=True, use_disk_cache=False)
        
        # Check attributes
        self.assertTrue(interface.use_cache)
        self.assertFalse(interface.use_disk_cache)
    
    @patch('core.model_interface.config')
    @patch('core.model_interface._RATE_LIMITERS', {})
    def test_setup_rate_limiters(self, mock_config):
        """Test setup of rate limiters."""
        # Configure mock
        mock_config.API_CONFIGS = self.api_configs
        mock_config.DISK_CACHE_CONFIG = self.disk_cache_config
        
        # Create a ModelInterface instance, which should set up rate limiters
        interface = ModelInterface()
        
        # Access the private _setup_rate_limiters method directly for testing
        interface._setup_rate_limiters()
        
        # Check that rate limiters were created for each API in config
        from core.model_interface import _RATE_LIMITERS
        self.assertIn("gemini", _RATE_LIMITERS)
        self.assertIn("groq", _RATE_LIMITERS)
        
        # Check structure of rate limiters
        for api_name in ["gemini", "groq"]:
            self.assertIn("min_interval", _RATE_LIMITERS[api_name])
            self.assertIn("last_request_time", _RATE_LIMITERS[api_name])
            self.assertIn("lock", _RATE_LIMITERS[api_name])
            self.assertIsInstance(_RATE_LIMITERS[api_name]["lock"], threading.Lock)
    
    @patch('core.model_interface.config')
    @patch('core.model_interface.ModelInterface._generate_with_local_model')
    def test_generate_text_local_model(self, mock_generate_local, mock_config):
        """Test generate_text with a local model."""
        # Configure mocks
        mock_config.MODEL_CONFIGS = self.model_configs
        mock_config.DISK_CACHE_CONFIG = self.disk_cache_config
        mock_generate_local.return_value = ("Generated text", {"token_count": 10})
        
        # Create a ModelInterface instance
        interface = ModelInterface()
        
        # Call generate_text with a local model
        response, stats = interface.generate_text("llama", "Test prompt")
        
        # Check the response
        self.assertEqual(response, "Generated text")
        self.assertEqual(stats["token_count"], 10)
        
        # Verify the local model generator was called correctly
        mock_generate_local.assert_called_once_with("llama", "Test prompt", self.model_configs["llama"])
    
    @patch('core.model_interface.config')
    @patch('core.model_interface.ModelInterface._generate_with_gemini')
    def test_generate_text_gemini(self, mock_generate_gemini, mock_config):
        """Test generate_text with Gemini API."""
        # Configure mocks
        mock_config.MODEL_CONFIGS = self.model_configs
        mock_config.DISK_CACHE_CONFIG = self.disk_cache_config
        mock_generate_gemini.return_value = ("Gemini response", {"token_count": 20})
        
        # Create a ModelInterface instance
        interface = ModelInterface()
        
        # Call generate_text with Gemini
        response, stats = interface.generate_text("gemini", "Test prompt")
        
        # Check the response
        self.assertEqual(response, "Gemini response")
        self.assertEqual(stats["token_count"], 20)
        
        # Verify the Gemini generator was called correctly
        mock_generate_gemini.assert_called_once_with("Test prompt", self.model_configs["gemini"])
    
    @patch('core.model_interface.config')
    @patch('core.model_interface.ModelInterface._generate_with_groq')
    def test_generate_text_groq(self, mock_generate_groq, mock_config):
        """Test generate_text with Groq API."""
        # Configure mocks
        mock_config.MODEL_CONFIGS = self.model_configs
        mock_config.DISK_CACHE_CONFIG = self.disk_cache_config
        mock_generate_groq.return_value = ("Groq response", {"token_count": 15})
        
        # Create a ModelInterface instance
        interface = ModelInterface()
        
        # Call generate_text with Groq
        response, stats = interface.generate_text("groq", "Test prompt")
        
        # Check the response
        self.assertEqual(response, "Groq response")
        self.assertEqual(stats["token_count"], 15)
        
        # Verify the Groq generator was called correctly
        mock_generate_groq.assert_called_once_with("Test prompt", self.model_configs["groq"])
    
    @patch('core.model_interface.config')
    def test_generate_text_unsupported_model(self, mock_config):
        """Test generate_text with an unsupported model."""
        # Configure mocks
        mock_config.MODEL_CONFIGS = self.model_configs
        mock_config.DISK_CACHE_CONFIG = self.disk_cache_config
        
        # Create a ModelInterface instance
        interface = ModelInterface()
        
        # Call generate_text with an unsupported model
        response, stats = interface.generate_text("unsupported_model", "Test prompt")
        
        # Check that an error is returned
        self.assertTrue(response.startswith("[Error:"))
        self.assertTrue(stats["has_error"])
        self.assertIn("Model không được hỗ trợ", stats["error_message"])


if __name__ == '__main__':
    unittest.main() 