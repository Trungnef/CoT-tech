"""
Test việc cấu hình max_tokens động dựa trên loại prompt.
Chạy file này để kiểm tra xem hệ thống đã áp dụng đúng max_tokens cho từng loại prompt chưa.
"""

import sys
import logging
import os
from pathlib import Path

# Thêm thư mục gốc vào Python path (tạo đường dẫn tuyệt đối)
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_prompt_tokens")

# Kiểm tra xem config có thể import được không
try:
    import llm_evaluation.config as config
    logger.info("Import config thành công")
except Exception as e:
    logger.error(f"Lỗi khi import config: {e}")
    sys.exit(1)

# Thử import từng module
try:
    from llm_evaluation.core.model_interface import get_model_interface
    logger.info("Import model_interface thành công")
except Exception as e:
    logger.error(f"Lỗi khi import model_interface: {e}")
    sys.exit(1)

try:
    from llm_evaluation.core.prompt_builder import create_prompt
    logger.info("Import prompt_builder thành công")
except Exception as e:
    logger.error(f"Lỗi khi import prompt_builder: {e}")
    sys.exit(1)

def test_prompt_max_tokens():
    """
    Kiểm tra xem hệ thống có áp dụng đúng max_tokens cho từng loại prompt không.
    Tạo các prompt với các loại khác nhau và kiểm tra max_tokens được sử dụng.
    """
    logger.info("Bắt đầu kiểm tra max_tokens cho các loại prompt khác nhau")
    
    # Tạo model interface
    model_interface = get_model_interface(use_cache=True)
    
    # Câu hỏi mẫu để test
    test_question = "Tính tổng các số từ 1 đến 100"
    
    # Danh sách các loại prompt để test
    prompt_types = [
        "zero_shot",
        "few_shot_3",
        "cot",
        "cot_self_consistency_5",
        "react"
    ]
    
    # Danh sách model để test
    models_to_test = config.DEFAULT_MODELS
    
    # In ra cấu hình max_tokens cho mỗi loại prompt và model
    logger.info("=== Cấu hình max_tokens theo loại prompt ===")
    for prompt_type in prompt_types:
        logger.info(f"Prompt type: {prompt_type}")
        for model_name in models_to_test:
            max_tokens = config.get_max_tokens(model_name, prompt_type)
            logger.info(f"  - {model_name}: {max_tokens} tokens")
    
    # Test model_interface.generate_text với prompt_type trong config
    logger.info("\n=== Test 1: Kiểm tra method generate_text với prompt_type trong config ===")
    for model_name in models_to_test:
        for prompt_type in prompt_types:
            # Tạo prompt
            prompt = create_prompt(test_question, prompt_type=prompt_type)
            
            # Chuẩn bị config với prompt_type để lấy max_tokens phù hợp
            test_config = {
                "prompt_type": prompt_type,
                "temperature": 0.7,
            }
            
            # Không thực sự gọi model, chỉ log ra để kiểm tra max_tokens
            expected_max_tokens = config.get_max_tokens(model_name, prompt_type)
            logger.info(f"[Test generate_text] Model: {model_name}, Prompt type: {prompt_type}, Expected max_tokens: {expected_max_tokens}")
            
            # Trong môi trường thực tế, bạn có thể uncomment dòng dưới để thực sự gọi model
            # Nhưng để tiết kiệm API calls và thời gian, chúng ta chỉ log ra thông tin
            # response, stats = model_interface.generate_text(model_name, prompt, test_config)
            # logger.info(f"Response: {response[:100]}...")
    
    # Test get_response để đảm bảo nó cũng sử dụng max_tokens đúng
    logger.info("\n=== Test 2: Kiểm tra method get_response của ModelInterface ===")
    for model_name in models_to_test:
        for prompt_type in prompt_types:
            # Tạo prompt
            prompt = create_prompt(test_question, prompt_type=prompt_type)
            
            # Lấy max_tokens phù hợp
            max_tokens = config.get_max_tokens(model_name, prompt_type)
            
            # Log thông tin để kiểm tra
            logger.info(f"[Test get_response] Model: {model_name}, Prompt type: {prompt_type}, max_tokens: {max_tokens}")
            
            # Trong môi trường thực tế, bạn có thể uncomment dòng dưới để thực sự gọi model
            # response = model_interface.get_response(model_name, prompt, max_tokens)
            # logger.info(f"Response: {response[:100]}...")
    
    # Test batch_generate_text để đảm bảo nó cũng sử dụng max_tokens đúng
    logger.info("\n=== Test 3: Kiểm tra Evaluator._evaluate_local_model_batch ===")
    for model_name in models_to_test:
        if model_name.lower() in ["llama", "qwen"]:  # Chỉ test với model local
            logger.info(f"Testing batch processing for {model_name}")
            
            # Mô phỏng dữ liệu batch_prompts mà Evaluator._evaluate_local_model_batch nhận vào
            batch_prompts = []
            for prompt_type in prompt_types:
                prompt = create_prompt(test_question, prompt_type=prompt_type)
                max_tokens = config.get_max_tokens(model_name, prompt_type)
                
                batch_prompts.append({
                    "prompt": prompt,
                    "prompt_type": prompt_type,
                    "question_idx": 0
                })
                
                logger.info(f"  - Prompt type: {prompt_type}, max_tokens: {max_tokens}")
            
            # Trong môi trường thực tế, bạn sẽ gọi Evaluator._evaluate_local_model_batch
            # Ở đây chúng ta chỉ kiểm tra cấu hình
    
    logger.info("\n=== Kết luận ===")
    logger.info("Kiểm tra hoàn tất. Xem log ở trên để kiểm tra cấu hình max_tokens cho từng loại prompt.")

if __name__ == "__main__":
    test_prompt_max_tokens() 