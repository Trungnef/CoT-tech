#!/usr/bin/env python3
"""
Script kiểm tra hệ thống đánh giá LLM với các chức năng đánh giá mới.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path

# Thêm thư mục gốc vào sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_system.log', 'w', 'utf-8'),
    ]
)

logger = logging.getLogger("test_system")

def check_module_imports():
    """Kiểm tra các modules cần thiết có thể import được không."""
    logger.info("Kiểm tra imports...")
    
    try:
        # Core modules
        import llm_evaluation.core.evaluator
        import llm_evaluation.core.model_interface
        import llm_evaluation.core.prompt_builder
        import llm_evaluation.core.result_analyzer
        from llm_evaluation.core.reporting import Reporting
        
        # Utils
        import llm_evaluation.utils.logging_setup
        
        logger.info("✅ Tất cả modules đã được import thành công.")
        return True
    except ImportError as e:
        logger.error(f"❌ Lỗi import module: {str(e)}")
        return False

def check_visualization_methods():
    """Kiểm tra các phương thức tạo biểu đồ mới."""
    logger.info("Kiểm tra các phương thức tạo biểu đồ...")
    
    try:
        from llm_evaluation.core.reporting import Reporting
        
        # Tạo danh sách các phương thức cần kiểm tra
        visualization_methods = [
            '_create_difficulty_performance_plot',
            '_create_criteria_evaluation_plot',
            '_create_criteria_radar_plot',
            '_create_context_adherence_plot',
            '_create_difficulty_comparison_plot',
            '_create_overall_criteria_comparison',
            '_create_fallback_plot',
            '_generate_visualizations'
        ]
        
        # Kiểm tra từng phương thức
        missing_methods = []
        for method_name in visualization_methods:
            if not hasattr(Reporting, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            logger.error(f"❌ Thiếu các phương thức sau trong lớp Reporting: {', '.join(missing_methods)}")
            return False
        else:
            logger.info("✅ Tất cả phương thức tạo biểu đồ đã được triển khai.")
            return True
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra phương thức tạo biểu đồ: {str(e)}")
        return False

def check_evaluation_criteria():
    """Kiểm tra các tiêu chí đánh giá mới."""
    logger.info("Kiểm tra các tiêu chí đánh giá...")
    
    try:
        from llm_evaluation.core.result_analyzer import ResultAnalyzer
        
        # Tạo một đối tượng ResultAnalyzer giả
        analyzer = ResultAnalyzer()
        
        # Kiểm tra các tiêu chí có trong reasoning_criteria
        expected_criteria = [
            'accuracy', 
            'reasoning_consistency', 
            'consistency', 
            'difficulty_performance', 
            'context_adherence'
        ]
        
        missing_criteria = [c for c in expected_criteria if c not in analyzer.reasoning_criteria]
        
        if missing_criteria:
            logger.error(f"❌ Thiếu các tiêu chí đánh giá sau: {', '.join(missing_criteria)}")
            return False
        
        # Kiểm tra các phương thức tính toán tiêu chí
        computation_methods = [
            '_compute_accuracy_metrics',
            '_compute_difficulty_metrics',
            '_compute_context_adherence_metrics'
        ]
        
        missing_methods = [m for m in computation_methods if not hasattr(ResultAnalyzer, m)]
        
        if missing_methods:
            logger.error(f"❌ Thiếu các phương thức tính toán sau: {', '.join(missing_methods)}")
            return False
            
        logger.info("✅ Tất cả tiêu chí đánh giá đã được triển khai.")
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra tiêu chí đánh giá: {str(e)}")
        return False

def check_prompt_builder():
    """Kiểm tra các cải tiến trong PromptBuilder."""
    logger.info("Kiểm tra cải tiến trong PromptBuilder...")
    
    try:
        from llm_evaluation.core.prompt_builder import PromptBuilder
        
        # Tạo đối tượng PromptBuilder
        builder = PromptBuilder()
        
        # Kiểm tra các cải tiến trong extract_final_answer
        test_prompts = [
            ("Đây là câu trả lời của tôi.\nVậy đáp án là 42.", "zero_shot"),
            ("Theo tính toán, tôi thấy rằng kết quả là 15.", "few_shot_3"),
            ("Bước 1: Tính tổng.\nBước 2: Chia cho 2.\nKết luận: Đáp án là 30.", "cot"),
            ("Cách 1: 2+2=4\nĐáp án: 4\nCách 2: 8/2=4\nĐáp án: 4", "self_consistency_3"),
            ("Suy nghĩ: Tôi cần tính tổng.\nHành động: Cộng các số lại.\nKết quả: 25\nĐáp án cuối cùng: 25", "react")
        ]
        
        success = True
        for response, prompt_type in test_prompts:
            answer = builder.extract_final_answer(response, prompt_type)
            if not answer:
                logger.error(f"❌ extract_final_answer không trích xuất được từ '{response}' với prompt_type='{prompt_type}'")
                success = False
        
        if success:
            logger.info("✅ PromptBuilder.extract_final_answer hoạt động tốt.")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra PromptBuilder: {str(e)}")
        return False
        
def check_model_interface():
    """Kiểm tra phương thức get_response trong ModelInterface."""
    logger.info("Kiểm tra ModelInterface.get_response...")
    
    try:
        from llm_evaluation.core.model_interface import ModelInterface
        
        # Kiểm tra sự tồn tại của phương thức
        if not hasattr(ModelInterface, 'get_response'):
            logger.error("❌ Phương thức get_response không tồn tại trong ModelInterface")
            return False
            
        logger.info("✅ Phương thức get_response tồn tại trong ModelInterface.")
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi khi kiểm tra ModelInterface: {str(e)}")
        return False

def main():
    """Hàm chính chạy tất cả các bài kiểm tra."""
    logger.info("Bắt đầu kiểm tra hệ thống...")
    
    tests = [
        check_module_imports,
        check_visualization_methods,
        check_evaluation_criteria,
        check_prompt_builder,
        check_model_interface
    ]
    
    results = {}
    all_passed = True
    
    for test_func in tests:
        test_name = test_func.__name__
        result = test_func()
        results[test_name] = "PASS" if result else "FAIL"
        all_passed = all_passed and result
    
    # In tổng kết
    logger.info("=== KẾT QUẢ KIỂM TRA ===")
    for test_name, result in results.items():
        icon = "✅" if result == "PASS" else "❌"
        logger.info(f"{icon} {test_name}: {result}")
    
    logger.info(f"Tổng kết: {'✅ Tất cả bài kiểm tra đã đạt.' if all_passed else '❌ Một số bài kiểm tra thất bại.'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 