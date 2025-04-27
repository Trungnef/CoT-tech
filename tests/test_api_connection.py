#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script kiểm tra kết nối tới API của Groq và Gemini.
"""

import os
import sys
import time
import logging
from pathlib import Path
import json
from dotenv import load_dotenv

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_api_connection")

# Thêm đường dẫn hiện tại vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Tải biến môi trường từ file .env trong thư mục llm_evaluation
env_path = os.path.join(current_dir, "llm_evaluation", ".env")
if os.path.exists(env_path):
    logger.info(f"Tìm thấy file .env tại: {env_path}")
    load_dotenv(env_path)
else:
    logger.warning(f"Không tìm thấy file .env tại: {env_path}")
    load_dotenv()  # Thử tải từ thư mục hiện tại

def test_groq_api(api_key):
    """Kiểm tra kết nối tới Groq API với một key cụ thể."""
    masked_key = api_key[:10] + "..." if len(api_key) > 10 else api_key
    logger.info(f"Kiểm tra Groq API key: {masked_key}")
    
    try:
        # Import Groq client
        try:
            from groq import Groq
        except ImportError:
            logger.error("Không thể import thư viện Groq. Vui lòng cài đặt: pip install groq")
            return False, "Lỗi import thư viện"
        
        # Khởi tạo client
        client = Groq(api_key=api_key)
        
        # Thực hiện request đơn giản
        start_time = time.time()
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": "Xin chào, bạn có thể giúp tôi không?"}
            ],
            max_tokens=50
        )
        elapsed_time = time.time() - start_time
        
        # Kiểm tra phản hồi
        content = response.choices[0].message.content
        logger.info(f"  - Đã nhận phản hồi sau {elapsed_time:.2f}s: {content[:50]}...")
        
        return True, f"OK ({elapsed_time:.2f}s)"
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"  - Lỗi: {error_msg}")
        return False, error_msg

def test_gemini_api(api_key):
    """Kiểm tra kết nối tới Gemini API với một key cụ thể."""
    masked_key = api_key[:10] + "..." if len(api_key) > 10 else api_key
    logger.info(f"Kiểm tra Gemini API key: {masked_key}")
    
    try:
        # Import Gemini client
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error("Không thể import thư viện Google Generative AI. Vui lòng cài đặt: pip install google-generativeai")
            return False, "Lỗi import thư viện"
        
        # Khởi tạo client
        genai.configure(api_key=api_key)
        
        # Thực hiện request đơn giản
        start_time = time.time()
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Xin chào, bạn có thể giúp tôi không?")
        elapsed_time = time.time() - start_time
        
        # Kiểm tra phản hồi
        content = response.text
        logger.info(f"  - Đã nhận phản hồi sau {elapsed_time:.2f}s: {content[:50]}...")
        
        return True, f"OK ({elapsed_time:.2f}s)"
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"  - Lỗi: {error_msg}")
        return False, error_msg

if __name__ == "__main__":
    logger.info("Bắt đầu kiểm tra kết nối API...")
    
    # Kiểm tra thư viện
    missing_libraries = []
    try:
        import google.generativeai
    except ImportError:
        missing_libraries.append("google-generativeai")
    
    try:
        import groq
    except ImportError:
        missing_libraries.append("groq")
    
    if missing_libraries:
        logger.warning(f"Các thư viện sau chưa được cài đặt: {', '.join(missing_libraries)}")
        logger.info("Vui lòng cài đặt các thư viện thiếu bằng lệnh:")
        logger.info(f"pip install {' '.join(missing_libraries)}")
        sys.exit(1)
    
    # Lấy danh sách API keys
    groq_api_keys = [key.strip() for key in os.getenv("GROQ_API_KEYS", "").split(",") if key.strip()]
    if not groq_api_keys:
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if groq_api_key:
            groq_api_keys = [groq_api_key]
    
    gemini_api_keys = [key.strip() for key in os.getenv("GEMINI_API_KEYS", "").split(",") if key.strip()]
    if not gemini_api_keys:
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_api_key:
            gemini_api_keys = [gemini_api_key]
    
    # Kiểm tra số lượng keys
    if not groq_api_keys:
        logger.error("Không tìm thấy Groq API key trong biến môi trường")
    else:
        logger.info(f"Đã tìm thấy {len(groq_api_keys)} Groq API key(s)")
        
    if not gemini_api_keys:
        logger.error("Không tìm thấy Gemini API key trong biến môi trường")
    else:
        logger.info(f"Đã tìm thấy {len(gemini_api_keys)} Gemini API key(s)")
    
    # Kiểm tra từng Groq API key
    logger.info("\n=== KIỂM TRA GROQ API KEYS ===")
    groq_results = {}
    for idx, key in enumerate(groq_api_keys):
        logger.info(f"Kiểm tra key {idx+1}/{len(groq_api_keys)}")
        success, message = test_groq_api(key)
        groq_results[key] = {"success": success, "message": message}
        time.sleep(1)  # Đợi giữa các request để tránh rate limit
    
    # Kiểm tra từng Gemini API key
    logger.info("\n=== KIỂM TRA GEMINI API KEYS ===")
    gemini_results = {}
    for idx, key in enumerate(gemini_api_keys):
        logger.info(f"Kiểm tra key {idx+1}/{len(gemini_api_keys)}")
        success, message = test_gemini_api(key)
        gemini_results[key] = {"success": success, "message": message}
        time.sleep(1)  # Đợi giữa các request để tránh rate limit
    
    # Tổng kết kết quả
    logger.info("\n=== TỔNG KẾT KẾT QUẢ ===")
    
    # Tổng kết Groq API
    logger.info("\nGroq API Keys:")
    for idx, (key, result) in enumerate(groq_results.items()):
        masked_key = key[:10] + "..." if len(key) > 10 else key
        status = "✓" if result["success"] else "✗"
        logger.info(f"{idx+1}. {masked_key}: {status} - {result['message']}")
    
    groq_working_count = sum(1 for result in groq_results.values() if result["success"])
    logger.info(f"Tổng cộng: {groq_working_count}/{len(groq_results)} Groq key hoạt động")
    
    # Tổng kết Gemini API
    logger.info("\nGemini API Keys:")
    for idx, (key, result) in enumerate(gemini_results.items()):
        masked_key = key[:10] + "..." if len(key) > 10 else key
        status = "✓" if result["success"] else "✗"
        logger.info(f"{idx+1}. {masked_key}: {status} - {result['message']}")
    
    gemini_working_count = sum(1 for result in gemini_results.values() if result["success"])
    logger.info(f"Tổng cộng: {gemini_working_count}/{len(gemini_results)} Gemini key hoạt động")
    
    # Kết luận chung
    all_working = (groq_working_count > 0 and gemini_working_count > 0)
    logger.info("\n=== KẾT LUẬN ===")
    if all_working:
        logger.info("✓ Có ít nhất một key hoạt động cho mỗi API")
        sys.exit(0)
    else:
        if groq_working_count == 0 and gemini_working_count == 0:
            logger.error("✗ Tất cả các API keys đều không hoạt động")
        elif groq_working_count == 0:
            logger.error("✗ Tất cả Groq API keys đều không hoạt động")
        elif gemini_working_count == 0:
            logger.error("✗ Tất cả Gemini API keys đều không hoạt động")
        sys.exit(1) 