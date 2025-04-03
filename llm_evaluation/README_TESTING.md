# Testing Guide for LLM Evaluation Framework

This document provides guidance on testing the LLM Evaluation Framework components.

## Test Structure

The testing framework uses Python's built-in `unittest` library. Test files are located in the `tests/` directory and follow the naming convention `test_*.py`.

## Running Tests

To run all tests:

```bash
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests.test_prompt_builder
```

To run a specific test case:

```bash
python -m unittest tests.test_prompt_builder.TestPromptBuilder.test_initialization
```

## Test Files

### Existing Test Files

- `test_basic.py`: Basic tests to verify the test infrastructure is working correctly
- `test_checkpoint_manager.py`: Tests for the CheckpointManager class
- `test_evaluator.py`: Tests for the Evaluator class
- `test_evaluator_simple.py`: Simplified tests for the Evaluator class
- `test_model_interface.py`: Tests for the ModelInterface class
- `test_prompt_builder.py`: Tests for the PromptBuilder class
- `test_result_analyzer.py`: Tests for the ResultAnalyzer class
- `test_reporting.py`: Tests for the reporting functions
- `test_utils.py`: Tests for utility functions

### Testing Approach

#### Mocking Dependencies

Most core components have dependencies on other modules (e.g., `config`, other classes). The recommended approach is to use `unittest.mock.patch` to mock these dependencies:

```python
@patch('core.evaluator.ModelInterface')
@patch('core.evaluator.ResultAnalyzer')
def test_method(self, mock_analyzer, mock_model_interface):
    # Test code here
    pass
```

#### Testing Vietnamese-Specific Code

Some components (like `PromptBuilder`) are designed to work with Vietnamese text patterns. When testing these components:

- Use Vietnamese text patterns in test inputs when testing pattern recognition
- Be aware of language-specific behavior in methods like `extract_final_answer`

## Common Testing Challenges

1. **Import Issues**: Modules often import from the project root, which can cause issues in tests.
   - Solution: Add project root to `sys.path`
   ```python
   sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
   ```

2. **Config Module**: Many components import from a global `config` module.
   - Solution: Mock the config module with `patch('core.module_name.config')`

3. **Random Selection**: Some methods (like `create_few_shot_prompt`) use random selection.
   - Solution: Test for patterns and counts rather than specific content

## Example: Testing the PromptBuilder Class

```python
def test_create_few_shot_prompt(self):
    """Test the creation of a few-shot prompt."""
    question = "What is the largest planet in our solar system?"
    prompt = self.builder_en.create_prompt(question, "few_shot_2", self.examples)
    
    # Check for patterns rather than specific examples
    self.assertIn(self.builder_en.system_message, prompt)
    self.assertIn("Problem: " + question, prompt)
    
    # Check for the expected structure
    self.assertIn("Here are some examples", prompt)
    self.assertEqual(prompt.count("Problem:"), 3)  # 2 examples + 1 question
    self.assertEqual(prompt.count("Answer:"), 3)   # 2 examples + final prompt
```

## Next Steps for Improving Test Coverage

1. Add more tests for the `Evaluator` class, focusing on individual methods
2. Expand testing for the `ModelInterface` class, mocking specific methods
3. Add tests for error handling and edge cases in all components
4. Implement integration tests for key workflows

## Best Practices

1. Keep tests independent - each test should not depend on the state from other tests
2. Use descriptive test names that explain what's being tested
3. Test both success and failure cases
4. Mock external dependencies but test actual logic
5. Use setUp/tearDown methods to prepare and clean up test fixtures 