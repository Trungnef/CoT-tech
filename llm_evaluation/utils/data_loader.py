"""
Module tải dữ liệu câu hỏi cho framework đánh giá LLM.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import traceback

# Sử dụng get_logger từ setup để nhất quán
try:
    # Thử import từ cấu trúc package chuẩn
    from .logging_setup import get_logger
except ImportError:
    # Fallback nếu chạy trực tiếp hoặc import từ nơi khác
    import sys
    sys.path.append(str(Path(__file__).parent.parent.absolute())) # Thêm thư mục gốc vào path
    from utils.logging_setup import get_logger

logger = get_logger(__name__)

def load_questions(file_path: Union[str, Path]) -> Optional[List[Dict[str, Any]]]:
    """
    Đọc danh sách câu hỏi từ file JSON.

    Args:
        file_path (Union[str, Path]): Đường dẫn đến file JSON chứa câu hỏi.
                                      Mỗi câu hỏi nên là một object JSON.

    Returns:
        Optional[List[Dict[str, Any]]]: Danh sách các dictionary câu hỏi,
                                         hoặc None nếu có lỗi xảy ra.
    """
    path = Path(file_path)
    logger.info(f"Đang tải câu hỏi từ: {path}")

    if not path.exists():
        logger.error(f"File câu hỏi không tồn tại: {path}")
        return None
    if not path.is_file():
        logger.error(f"Đường dẫn câu hỏi không phải là file: {path}")
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Đọc từng dòng nếu file là JSON Lines
            if file_path.lower().endswith(".jsonl"):
                questions = [json.loads(line) for line in f if line.strip()]
            # Đọc toàn bộ file nếu là JSON array
            else:
                questions_data = json.load(f)
                if isinstance(questions_data, list):
                    questions = questions_data
                # Xử lý trường hợp file JSON chứa object với key chứa list câu hỏi
                elif isinstance(questions_data, dict):
                    # Tìm key chứa list (ví dụ: 'questions', 'data', 'problems')
                    list_key = next((k for k, v in questions_data.items() if isinstance(v, list)), None)
                    if list_key:
                        logger.warning(f"File JSON chứa object, đang lấy dữ liệu từ key '{list_key}'")
                        questions = questions_data[list_key]
                    else:
                        logger.error("File JSON là object nhưng không tìm thấy key chứa danh sách câu hỏi.")
                        return None
                else:
                     logger.error(f"Định dạng JSON không được hỗ trợ trong file: {path}. Phải là list hoặc object chứa list.")
                     return None

        # Kiểm tra định dạng cơ bản (mỗi item là dict và có 'id', 'question')
        validated_questions = []
        required_keys = {'id', 'question'}
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                logger.warning(f"Mục {i} trong file câu hỏi không phải là dictionary, bỏ qua.")
                continue
            if not required_keys.issubset(q.keys()):
                 # Cố gắng sử dụng 'text' nếu 'question' không tồn tại
                 if 'id' in q and 'text' in q and 'question' not in q:
                     q['question'] = q['text'] # Alias 'text' to 'question'
                     logger.debug(f"Câu hỏi ID {q.get('id', i)}: Sử dụng trường 'text' thay cho 'question'")
                 else:
                    logger.warning(f"Câu hỏi ID {q.get('id', i)} thiếu trường bắt buộc ('id' hoặc 'question'), bỏ qua.")
                    continue
            validated_questions.append(q)

        if not validated_questions:
             logger.error("Không tìm thấy câu hỏi hợp lệ nào trong file.")
             return None

        logger.info(f"Đã tải thành công {len(validated_questions)} câu hỏi hợp lệ từ {path}")
        return validated_questions

    except json.JSONDecodeError as e:
        logger.error(f"Lỗi giải mã JSON trong file {path}: {e}")
        logger.debug(traceback.format_exc())
        return None
    except Exception as e:
        logger.error(f"Lỗi không xác định khi tải câu hỏi từ {path}: {e}")
        logger.debug(traceback.format_exc())
        return None

# Ví dụ sử dụng (có thể bỏ đi khi tích hợp)
if __name__ == "__main__":
    # Tạo file JSON mẫu để test
    sample_questions = [
        {"id": 1, "question": "Câu hỏi 1?", "correct_answer": "Đáp án 1"},
        {"id": 2, "question": "Câu hỏi 2?"},
        {"id": 3, "text": "Câu hỏi 3 từ text?", "category": "test"}, # Sử dụng 'text'
        {"id": 4}, # Thiếu 'question'
        "not a dict", # Không phải dict
        {"question": "Câu hỏi 5?"} # Thiếu 'id'
    ]
    sample_file = "sample_questions.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_questions, f, indent=2)

    # Test loading
    loaded_q = load_questions(sample_file)
    if loaded_q:
        print(f"Loaded {len(loaded_q)} questions:")
        for q in loaded_q:
            print(q)
    else:
        print("Failed to load questions.")

    # Xóa file mẫu
    os.remove(sample_file)
