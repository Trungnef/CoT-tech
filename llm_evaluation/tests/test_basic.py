"""
Basic unit tests to verify test setup.
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing modules
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

class TestBasic(unittest.TestCase):
    
    def test_basic_assert(self):
        """Test that basic assertions work."""
        self.assertEqual(1 + 1, 2)
        self.assertTrue(True)
        self.assertFalse(False)
        
    def test_file_paths(self):
        """Test that we can access files in the project."""
        # Get the project root
        project_root = Path(__file__).parent.parent.absolute()
        
        # Check that important directories exist
        self.assertTrue(os.path.exists(project_root / "core"))
        self.assertTrue(os.path.exists(project_root / "utils"))
        self.assertTrue(os.path.exists(project_root / "tests"))
        
        # Print the structure for debugging
        print("Project structure:")
        for item in os.listdir(project_root):
            if os.path.isdir(project_root / item):
                print(f"  - {item}/")
            else:
                print(f"  - {item}")

if __name__ == '__main__':
    unittest.main() 