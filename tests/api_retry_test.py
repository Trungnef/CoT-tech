import time
import random
from unittest.mock import MagicMock, patch
from llm_evaluation.utils.logging_setup import setup_logging, get_logger

# Set up logging
logger = setup_logging(log_file="api_retry_test.log")

class MockResponse:
    """Mock HTTP response"""
    def __init__(self, status_code, headers=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.content = text.encode()
        
    def json(self):
        return {"error": {"message": self.text}}
        
    def raise_for_status(self):
        """Simulate raise_for_status behavior"""
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Error: {self.text}",
                response=self
            )

def test_api_retry_simulation():
    """Test API retry mechanism with simulated errors"""
    from llm_evaluation.core.model_interface import create_smart_retry
    
    # Keep track of call count
    call_count = 0
    
    # Create a function that will fail with different errors
    def api_call_with_errors():
        nonlocal call_count
        call_count += 1
        
        # List of possible errors (status_code, headers, message)
        errors = [
            (429, {"Retry-After": "2"}, "Rate limit exceeded"),
            (502, {}, "Bad Gateway"),
            (500, {}, "Internal Server Error"),
            (503, {}, "Service Unavailable"),
            (None, {}, "Connection error")  # Simulate connection error
        ]
        
        # First 3 calls will fail with random errors, then succeed
        if call_count <= 3:
            error = random.choice(errors)
            logger.info(f"Simulating error: {error}")
            
            if error[0] is None:
                # Simulate connection error
                raise ConnectionError("Failed to establish connection")
            else:
                # Create mock response with error
                response = MockResponse(error[0], error[1], error[2])
                response.raise_for_status()
        
        # Otherwise return success
        logger.info(f"API call succeeded on attempt {call_count}")
        return "Success result"
    
    # Create retry decorator
    retry_decorator = create_smart_retry("test_api")
    
    # Apply decorator to our test function
    decorated_function = retry_decorator(api_call_with_errors)
    
    # Execute and verify retries
    start_time = time.time()
    result = decorated_function()
    end_time = time.time()
    
    # Verify results
    assert result == "Success result"
    assert call_count > 1  # Should have retried at least once
    assert end_time - start_time >= 1  # Should have waited due to backoff
    
    logger.info(f"API call succeeded after {call_count} attempts in {end_time - start_time:.2f} seconds")
    return call_count

if __name__ == "__main__":
    logger.info("Starting API retry simulation test...")
    
    # Run multiple simulations
    total_attempts = 0
    num_runs = 5
    
    for i in range(num_runs):
        logger.info(f"Starting simulation run {i+1}/{num_runs}")
        attempts = test_api_retry_simulation()
        total_attempts += attempts
        time.sleep(1)  # Pause between runs
    
    avg_attempts = total_attempts / num_runs
    logger.info(f"Average attempts needed: {avg_attempts:.1f}")
    logger.info("All API retry simulation tests completed successfully!") 