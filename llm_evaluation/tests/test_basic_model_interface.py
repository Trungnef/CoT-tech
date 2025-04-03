"""
Very basic tests for the ModelInterface class.
"""

import unittest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Create a complete mock config module
mock_config = MagicMock()
mock_config.MODEL_CONFIGS = {
    "llama": {"temperature": 0.7, "max_tokens": 512},
    "gemini": {"temperature": 0.5, "max_tokens": 1024},
    "groq": {"temperature": 0.7}
}
mock_config.API_CONFIGS = {
    "gemini": {"requests_per_minute": 60},
    "groq": {"requests_per_minute": 120}
}
mock_config.DISK_CACHE_CONFIG = {
    "enabled": False,
    "cache_dir": "model_cache",
    "max_size_gb": 10,
    "cleanup_on_startup": False
}

# Replace the actual config module
sys.modules['config'] = mock_config

# Now we can import ModelInterface
from core.model_interface import ModelInterface


class TestBasicModelInterface(unittest.TestCase):
    
    def test_unsupported_model_error(self):
        """Test that using an unsupported model returns an error."""
        # Create ModelInterface
        interface = ModelInterface()
        
        # Try to generate text with an unsupported model
        response, stats = interface.generate_text("unsupported_model", "Test prompt")
        
        # Check that it returns an error
        self.assertTrue(response.startswith("[Error:"))
        self.assertTrue(stats["has_error"])
        self.assertIn("Model không được hỗ trợ", stats["error_message"])


if __name__ == '__main__':
    unittest.main() 