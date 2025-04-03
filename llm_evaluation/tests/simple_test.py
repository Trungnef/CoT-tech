"""
Simple debugging test for config import issues.
"""
import unittest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Create a minimal mock config
mock_config = MagicMock()
mock_config.MODEL_CONFIGS = {"test": {"temperature": 0.7}}
mock_config.DISK_CACHE_CONFIG = {"enabled": False}
mock_config.API_CONFIGS = {}

# Check current sys.path
print("System path:")
for path in sys.path:
    print(f"  - {path}")

# Check if config already exists in sys.modules
if 'config' in sys.modules:
    print(f"Config already in sys.modules: {sys.modules['config']}")

# Forcibly replace config module
sys.modules['config'] = mock_config
print(f"Replaced config in sys.modules: {sys.modules['config']}")

# Now try to import from core.model_interface
print("\nAttempting to import ModelInterface...")
try:
    from core.model_interface import ModelInterface
    print("Successfully imported ModelInterface")
    
    # Create a ModelInterface instance
    interface = ModelInterface()
    print("Successfully created ModelInterface instance")
    
    # Test generate_text with unsupported model
    print("\nTesting generate_text with unsupported model...")
    response, stats = interface.generate_text("unsupported_model", "Test prompt")
    print(f"Response: {response}")
    print(f"Stats: {stats}")
    
except Exception as e:
    print(f"Error importing or using ModelInterface: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    print("\nRunning as script...") 