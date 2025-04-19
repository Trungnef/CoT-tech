#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để kiểm tra chức năng key rotation trong ModelInterface.
"""

import logging
import time
import datetime
import json
import traceback
from core.model_interface import get_model_interface
import config

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_key_rotation")

def test_key_rotation():
    """Kiểm tra chức năng key rotation."""
    try:
        # Hiển thị thông tin keys
        print(f"=== THÔNG TIN API KEYS ===")
        print(f"Gemini API keys: {len(config.GEMINI_API_KEYS)}")
        for i, key in enumerate(config.GEMINI_API_KEYS):
            masked_key = key[:10] + "..." if len(key) > 10 else key
            print(f"  Key {i+1}: {masked_key}")
            
        print(f"Groq API keys: {len(config.GROQ_API_KEYS)}")
        for i, key in enumerate(config.GROQ_API_KEYS):
            masked_key = key[:10] + "..." if len(key) > 10 else key
            print(f"  Key {i+1}: {masked_key}")
        
        # Khởi tạo ModelInterface
        interface = get_model_interface()
        
        # Hiển thị trạng thái ban đầu
        print("\n=== TRẠNG THÁI BAN ĐẦU ===")
        print(f"Gemini key index: {interface.current_gemini_key_index}")
        print(f"Groq key index: {interface.current_groq_key_index}")
        print(f"Gemini exhausted keys: {len(interface.exhausted_gemini_keys)} keys")
        print(f"Groq exhausted keys: {len(interface.exhausted_groq_keys)} keys")
        
        # Kiểm tra định dạng exhausted keys
        print("\nCấu trúc lưu trữ keys đã hết hạn:")
        print("interface.exhausted_gemini_keys:", type(interface.exhausted_gemini_keys))
        print("interface.exhausted_groq_keys:", type(interface.exhausted_groq_keys))
        
        # Test 1: Mô phỏng lỗi rate limit tạm thời cho Gemini
        print("\n=== TEST 1: MÔ PHỎNG HẾT QUOTA GEMINI KEY (TEMPORARY RATE LIMIT) ===")
        if config.GEMINI_API_KEYS:
            try:
                current_key = config.GEMINI_API_KEYS[0]
                print(f"Đánh dấu key {current_key[:10]}... là hết hạn tạm thời")
                
                interface.exhausted_gemini_keys[current_key] = {
                    'timestamp': time.time(),
                    'reason': 'rate_limit_exceeded',
                    'error': 'Simulated rate limit error'
                }
                
                print(f"Trước khi chuyển key: {interface.current_gemini_key_index}")
                new_client = interface._get_gemini_client()  # Gọi để tự động chuyển key
                print(f"Sau khi chuyển key: {interface.current_gemini_key_index}")
                print(f"Keys đã hết hạn: {len(interface.exhausted_gemini_keys)}")
                
                # In chi tiết về key đã hết hạn
                for k, v in interface.exhausted_gemini_keys.items():
                    print(f"  Key {k[:10]}...: {v['reason']}")
            except Exception as e:
                print(f"Lỗi: {e}")
                traceback.print_exc()
        
        # Test 2: Mô phỏng sang ngày mới
        print("\n=== TEST 2: MÔ PHỎNG SANG NGÀY MỚI ===")
        if config.GEMINI_API_KEYS and len(config.GEMINI_API_KEYS) >= 1:
            try:
                # Thay đổi timestamp của key đầu tiên thành ngày hôm qua
                yesterday = time.time() - 24*60*60  # 24 giờ trước
                current_key = config.GEMINI_API_KEYS[0]
                
                # Đảm bảo key có trong exhausted_keys
                if current_key not in interface.exhausted_gemini_keys:
                    interface.exhausted_gemini_keys[current_key] = {
                        'timestamp': yesterday,
                        'reason': 'daily_quota_exceeded',
                        'error': 'Simulated daily quota exceeded'
                    }
                else:
                    # Cập nhật timestamp thành ngày hôm qua
                    interface.exhausted_gemini_keys[current_key]['timestamp'] = yesterday
                    interface.exhausted_gemini_keys[current_key]['reason'] = 'daily_quota_exceeded'
                
                print(f"Đã đặt timestamp của key {current_key[:10]}... thành ngày hôm qua")
                print(f"Chi tiết key trước khi làm mới:")
                print(json.dumps(interface.exhausted_gemini_keys.get(current_key, {}), indent=2))
                
                # Force refresh keys
                interface.last_key_refresh_time = 0  # Reset thời gian refresh
                interface._refresh_exhausted_keys()
                
                # Kiểm tra xem key có được làm mới không
                is_refreshed = current_key not in interface.exhausted_gemini_keys
                print(f"Key đã được làm mới: {is_refreshed}")
                print(f"Số lượng keys còn trong danh sách hết hạn: {len(interface.exhausted_gemini_keys)}")
            except Exception as e:
                print(f"Lỗi: {e}")
                traceback.print_exc()
        
        print("\n=== KIỂM TRA HOÀN TẤT ===")
    except Exception as e:
        print(f"Lỗi chung: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_key_rotation() 