#!/usr/bin/env python
"""
Script kiểm tra trạng thái các API keys và tìm key còn khả dụng.
"""

import os
import sys
import logging
import argparse

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("check_api_keys")

# Thêm đường dẫn hiện tại vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_keys():
    try:
        # Import config để xác nhận thay đổi
        from llm_evaluation import config
        
        # Kiểm tra trạng thái quota
        logger.info("=== TRẠNG THÁI QUOTA ===")
        for api_name in ["gemini", "openai", "groq"]:
            exceeded_keys = config.QUOTA_EXCEEDED_KEYS.get(api_name, set())
            all_keys = getattr(config, f"{api_name.upper()}_API_KEYS", [])
            if not all_keys:
                logger.warning(f"Không có API keys nào cho {api_name}")
                continue
                
            logger.info(f"{api_name.upper()}: {len(exceeded_keys)}/{len(all_keys)} keys đã hết quota")
            
            # Hiển thị chi tiết các key đã hết quota
            if exceeded_keys:
                logger.info("Keys đã hết quota:")
                for key in exceeded_keys:
                    logger.info(f"  - {key[:10]}...")
            
            # Hiển thị các key còn khả dụng
            available_keys = config.find_available_api_keys(api_name)
            if available_keys:
                logger.info("Keys còn khả dụng:")
                for key in available_keys:
                    logger.info(f"  - {key[:10]}...")
            
            # Hiển thị key đang sử dụng
            current_key = config.get_api_key(api_name)
            if current_key:
                logger.info(f"Key đang dùng: {current_key[:10]}... (index: {config.CURRENT_API_KEY_INDEX[api_name]})")
            
            logger.info("---")
        
        # Hiển thị vị trí lưu trạng thái
        logger.info(f"Trạng thái quota được lưu tại: {config.QUOTA_STATE_FILE}")
        
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    return True

def reset_quota_state(api_name=None):
    """Reset trạng thái quota cho API cụ thể hoặc tất cả."""
    try:
        from llm_evaluation import config
        
        if api_name:
            if api_name not in ["gemini", "openai", "groq"]:
                logger.error(f"API không hợp lệ: {api_name}")
                return False
                
            config.reset_quota_exceeded_keys(api_name)
            logger.info(f"Đã reset trạng thái quota cho {api_name}")
        else:
            config.reset_quota_exceeded_keys()
            logger.info("Đã reset trạng thái quota cho tất cả API")
            
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi reset trạng thái quota: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra trạng thái API keys")
    parser.add_argument("--reset", choices=["gemini", "openai", "groq", "all"], 
                      help="Reset trạng thái quota cho API cụ thể hoặc tất cả")
    
    args = parser.parse_args()
    
    if args.reset:
        api_name = None if args.reset == "all" else args.reset
        success = reset_quota_state(api_name)
        if not success:
            sys.exit(1)
    
    success = check_keys()
    if not success:
        sys.exit(1) 