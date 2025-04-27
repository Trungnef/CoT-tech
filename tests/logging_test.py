import threading
import time
from llm_evaluation.utils.logging_setup import setup_logging, get_logger

def test_concurrent_logging():
    """Test concurrent logging initialization from multiple threads"""
    def worker(worker_id):
        # Each worker will try to get a logger
        logger = get_logger(f"worker_{worker_id}")
        logger.info(f"Worker {worker_id} initialized logger")
        
        # Simulate some work
        for i in range(3):
            logger.debug(f"Worker {worker_id} - debug message {i}")
            logger.info(f"Worker {worker_id} - info message {i}")
            time.sleep(0.1)
    
    # Create and start multiple threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    print("Concurrent logging test completed")

def test_reentrant_logger():
    """Test that reentrant logger calls don't cause deadlocks"""
    def setup_from_within_logging():
        # First get a logger
        logger = get_logger("reentrant_test")
        logger.info("First logger call")
        
        # Try to set up logging again (simulating a reentrant call)
        new_logger = setup_logging(log_file="reentrant_test.log")
        new_logger.info("Logging from reentrant setup")
        
        return new_logger
    
    logger = setup_from_within_logging()
    logger.info("Test completed successfully")

if __name__ == "__main__":
    print("Testing logging improvements...")
    
    # First test: concurrent logger initialization
    test_concurrent_logging()
    
    # Second test: reentrant logger calls
    test_reentrant_logger()
    
    print("All tests completed") 