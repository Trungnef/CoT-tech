#!/usr/bin/env python
"""
Script kiểm tra chức năng chuyển đổi API key khi hết quota.
"""

import os
import sys
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_api_keys")

# Thêm đường dẫn hiện tại vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import config để xác nhận thay đổi
    from llm_evaluation import config
    
    # Kiểm tra danh sách API keys
    logger.info("=== DANH SÁCH API KEYS ===")
    logger.info(f"Gemini API Keys: {config.GEMINI_API_KEYS}")
    logger.info(f"OpenAI API Keys: {config.OPENAI_API_KEYS}")
    logger.info(f"Groq API Keys: {config.GROQ_API_KEYS}")
    
    # Kiểm tra API key đầu tiên
    logger.info("\n=== API KEY HIỆN TẠI ===")
    logger.info(f"Gemini API Key: {config.GEMINI_API_KEY}")
    logger.info(f"OpenAI API Key: {config.OPENAI_API_KEY}")
    logger.info(f"Groq API Key: {config.GROQ_API_KEY}")
    
    # Kiểm tra chức năng lấy API key
    logger.info("\n=== GET API KEY ===")
    gemini_key = config.get_api_key("gemini")
    openai_key = config.get_api_key("openai")
    groq_key = config.get_api_key("groq")
    
    logger.info(f"Gemini Key: {gemini_key[:10]}... (index: {config.CURRENT_API_KEY_INDEX['gemini']})")
    logger.info(f"OpenAI Key: {openai_key[:10] if openai_key else 'None'} (index: {config.CURRENT_API_KEY_INDEX['openai']})")
    logger.info(f"Groq Key: {groq_key[:10] if groq_key else 'None'} (index: {config.CURRENT_API_KEY_INDEX['groq']})")
    
    # Thử chức năng chuyển đổi key
    logger.info("\n=== THỬ CHUYỂN ĐỔI KEY ===")
    if config.GEMINI_API_KEYS and len(config.GEMINI_API_KEYS) > 1:
        # Thử chuyển đổi key Gemini
        old_key = gemini_key
        new_key, success = config.switch_to_next_api_key("gemini", old_key)
        
        logger.info(f"Chuyển từ {old_key[:10]}... (index: {config.CURRENT_API_KEY_INDEX['gemini']-1}) sang {new_key[:10]}... (index: {config.CURRENT_API_KEY_INDEX['gemini']})")
        logger.info(f"Thành công: {success}")
        logger.info(f"Key hiện tại trong biến môi trường: {config.GEMINI_API_KEY[:10]}...")
        
        # Đặt lại danh sách key đã hết quota
        logger.info("\n=== RESET QUOTA EXCEEDED KEYS ===")
        logger.info(f"Trước khi reset: {config.QUOTA_EXCEEDED_KEYS['gemini']}")
        config.reset_quota_exceeded_keys("gemini")
        logger.info(f"Sau khi reset: {config.QUOTA_EXCEEDED_KEYS['gemini']}")
    else:
        logger.warning("Không có đủ Gemini API keys để kiểm tra chức năng chuyển đổi")
    
    logger.info("\n=== KIỂM TRA HOÀN TẤT ===")
    
except Exception as e:
    logger.error(f"Lỗi khi kiểm tra: {str(e)}")
    import traceback
    logger.error(traceback.format_exc()) 