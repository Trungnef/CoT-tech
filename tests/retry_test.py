import time
import random
import requests
from unittest.mock import MagicMock, patch
from llm_evaluation.utils.logging_setup import setup_logging, get_logger

# Set up logging
logger = setup_logging(log_file="retry_test.log")

# Import functions to test
from llm_evaluation.core.model_interface import should_retry_exception, wait_with_jitter

class MockRetryState:
    """Mock retry state for testing wait_with_jitter"""
    def __init__(self, attempt_number, exception=None):
        self.attempt_number = attempt_number
        self._exception = exception
        
    def outcome(self):
        class Outcome:
            def __init__(self, exception):
                self._exception = exception
                
            def exception(self):
                return self._exception
        
        return Outcome(self._exception)

def test_should_retry_exception():
    """Test the improved should_retry_exception function"""
    logger.info("Testing should_retry_exception...")
    
    # Test case 1: 429 Too Many Requests
    resp = MagicMock()
    resp.status_code = 429
    exception = requests.exceptions.HTTPError("429 Too Many Requests", response=resp)
    assert should_retry_exception(exception) == True
    logger.info("✅ Correctly identified 429 error as retryable")
    
    # Test case 2: 502 Bad Gateway
    resp = MagicMock()
    resp.status_code = 502
    exception = requests.exceptions.HTTPError("502 Bad Gateway", response=resp)
    assert should_retry_exception(exception) == True
    logger.info("✅ Correctly identified 502 error as retryable")
    
    # Test case 3: Connection error
    exception = ConnectionError("Failed to establish a connection")
    assert should_retry_exception(exception) == True
    logger.info("✅ Correctly identified ConnectionError as retryable")
    
    # Test case 4: Timeout error
    exception = TimeoutError("Request timed out")
    assert should_retry_exception(exception) == True
    logger.info("✅ Correctly identified TimeoutError as retryable")
    
    # Test case 5: 400 Bad Request (should not retry)
    resp = MagicMock()
    resp.status_code = 400
    exception = requests.exceptions.HTTPError("400 Bad Request", response=resp)
    assert should_retry_exception(exception) == False
    logger.info("✅ Correctly identified 400 error as non-retryable")
    
    logger.info("All should_retry_exception tests passed!")

def test_wait_with_jitter():
    """Test the improved wait_with_jitter function"""
    logger.info("Testing wait_with_jitter...")
    
    # Test exponential backoff (without Retry-After header)
    for attempt in range(1, 6):
        exception = TimeoutError("Request timed out")
        retry_state = MockRetryState(attempt, exception)
        
        wait_time = wait_with_jitter(retry_state)
        
        # With our implementation, wait_time should increase exponentially
        # Base (2^(attempt-1)) plus some jitter
        expected_base = min(60.0, 1.0 * (2 ** (attempt - 1)))
        
        logger.info(f"Attempt {attempt}: wait_time={wait_time:.2f}s, expected_base={expected_base:.2f}s")
        
        # Since we add random jitter, we can only check that it's in a reasonable range
        assert wait_time >= expected_base
        assert wait_time <= expected_base * 1.5  # Maximum jitter is 50% of base
    
    # Test Retry-After header handling
    resp = MagicMock()
    resp.headers = {'Retry-After': '10'}
    exception = requests.exceptions.HTTPError("429 Too Many Requests", response=resp)
    retry_state = MockRetryState(1, exception)
    
    # Mock the necessary attributes for Retry-After extraction
    with patch('llm_evaluation.core.model_interface.random.uniform', return_value=0.5):
        wait_time = wait_with_jitter(retry_state)
        logger.info(f"Retry-After test: wait_time={wait_time:.2f}s, expected≈10.5s")
        assert 10 <= wait_time <= 11  # Considering jitter
    
    logger.info("All wait_with_jitter tests passed!")

if __name__ == "__main__":
    logger.info("Starting retry mechanism tests...")
    
    try:
        test_should_retry_exception()
        test_wait_with_jitter()
        logger.info("All retry tests completed successfully!")
    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
        raise 