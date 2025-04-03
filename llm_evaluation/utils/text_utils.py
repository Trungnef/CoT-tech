"""
Tiện ích xử lý văn bản cho framework đánh giá LLM.
Cung cấp các hàm xử lý text, chuẩn hóa văn bản và hỗ trợ tiếng Việt.
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Union, Optional, Set, Tuple
import logging

from .logging_utils import get_logger

logger = get_logger(__name__)

# Ánh xạ accent tiếng Việt
VIETNAMESE_ACCENTS = {
    'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
    'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    'đ': 'd',
    'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
    'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
}

# Mở rộng các ký tự đặc biệt tiếng Việt
VIETNAMESE_SPECIAL_CHARS = 'àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ'

def clean_text(text: str, 
               lower: bool = True, 
               remove_punctuation: bool = False, 
               remove_numbers: bool = False,
               remove_whitespace: bool = False,
               normalize_unicode: bool = True) -> str:
    """
    Làm sạch văn bản, loại bỏ các ký tự không mong muốn.
    
    Args:
        text: Văn bản cần xử lý
        lower: Chuyển thành chữ thường
        remove_punctuation: Xóa dấu câu
        remove_numbers: Xóa chữ số
        remove_whitespace: Xóa khoảng trắng thừa
        normalize_unicode: Chuẩn hóa Unicode (NFKC)
        
    Returns:
        Văn bản đã làm sạch
    """
    if not text:
        return ""
    
    # Chuẩn hóa Unicode
    if normalize_unicode:
        text = unicodedata.normalize('NFKC', text)
    
    # Chuyển thành chữ thường
    if lower:
        text = text.lower()
    
    # Xóa dấu câu
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Xóa chữ số
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Xóa khoảng trắng thừa
    if remove_whitespace:
        text = ' '.join(text.split())
    
    return text

def remove_diacritics(text: str) -> str:
    """
    Loại bỏ dấu thanh, dấu phụ khỏi văn bản tiếng Việt.
    
    Args:
        text: Văn bản tiếng Việt
        
    Returns:
        Văn bản không dấu
    """
    text = unicodedata.normalize('NFD', text)
    result = ''.join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', result)

def simple_vietnamese_tokenize(text: str) -> List[str]:
    """
    Tokenize đơn giản cho văn bản tiếng Việt.
    
    Args:
        text: Văn bản cần tokenize
        
    Returns:
        Danh sách các token
    """
    # Thêm khoảng trắng trước và sau dấu câu
    text = re.sub(r'([.,!?();:"\'])', r' \1 ', text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    # Tách thành tokens
    return text.split()

def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt: sửa lỗi dấu câu, khoảng trắng.
    
    Args:
        text: Văn bản tiếng Việt cần chuẩn hóa
        
    Returns:
        Văn bản đã chuẩn hóa
    """
    if not text:
        return ""
    
    # Chuẩn hóa Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Sửa lỗi khoảng trắng trước dấu câu
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # Sửa lỗi không có khoảng trắng sau dấu câu
    text = re.sub(r'([,.!?;:])([^\s\d])', r'\1 \2', text)
    
    # Sửa lỗi dấu ngoặc
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    # Sửa lỗi dấu nháy
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)
    text = re.sub(r"'\s+", "'", text)
    text = re.sub(r"\s+'", "'", text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_numbers_from_text(text: str) -> List[float]:
    """
    Trích xuất các số từ văn bản.
    
    Args:
        text: Văn bản cần trích xuất
        
    Returns:
        Danh sách các số tìm thấy
    """
    # Tìm các số thực (có dấu phẩy hoặc dấu chấm)
    number_pattern = r'[-+]?\d+[.,]?\d*'
    matches = re.findall(number_pattern, text)
    
    # Chuyển đổi sang kiểu float
    numbers = []
    for match in matches:
        # Chuẩn hóa dấu phân cách
        match = match.replace(',', '.')
        try:
            numbers.append(float(match))
        except ValueError:
            logger.debug(f"Không thể chuyển đổi '{match}' thành số")
    
    return numbers

def is_vietnamese_text(text: str, threshold: float = 0.3) -> bool:
    """
    Kiểm tra xem văn bản có phải tiếng Việt không.
    
    Args:
        text: Văn bản cần kiểm tra
        threshold: Ngưỡng tỷ lệ ký tự tiếng Việt
        
    Returns:
        True nếu văn bản là tiếng Việt, False nếu không phải
    """
    if not text:
        return False
    
    # Đếm số ký tự tiếng Việt
    vietnamese_char_count = sum(1 for c in text if c in VIETNAMESE_SPECIAL_CHARS)
    
    # Tính tỷ lệ
    total_chars = len(text)
    ratio = vietnamese_char_count / total_chars if total_chars > 0 else 0
    
    return ratio >= threshold

def calculate_text_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
    """
    Tính độ tương đồng giữa hai văn bản.
    
    Args:
        text1: Văn bản thứ nhất
        text2: Văn bản thứ hai
        method: Phương pháp tính ('jaccard', 'cosine', 'levenshtein')
        
    Returns:
        Độ tương đồng (0.0 - 1.0)
    """
    if method not in ['jaccard', 'cosine', 'levenshtein']:
        raise ValueError(f"Phương pháp {method} không được hỗ trợ")
    
    if not text1 or not text2:
        return 0.0
    
    # Chuẩn hóa văn bản
    text1 = clean_text(text1, lower=True, remove_whitespace=True)
    text2 = clean_text(text2, lower=True, remove_whitespace=True)
    
    if method == 'jaccard':
        # Jaccard similarity
        tokens1 = set(simple_vietnamese_tokenize(text1))
        tokens2 = set(simple_vietnamese_tokenize(text2))
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union
    
    elif method == 'cosine':
        # Cosine similarity (đơn giản hóa)
        tokens1 = simple_vietnamese_tokenize(text1)
        tokens2 = simple_vietnamese_tokenize(text2)
        
        # Tạo từ điển tần suất
        freq1 = {}
        freq2 = {}
        
        for token in tokens1:
            freq1[token] = freq1.get(token, 0) + 1
        
        for token in tokens2:
            freq2[token] = freq2.get(token, 0) + 1
        
        # Tính tích vô hướng
        dot_product = sum(freq1.get(token, 0) * freq2.get(token, 0) for token in set(freq1.keys()).union(freq2.keys()))
        
        # Tính độ dài vector
        mag1 = sum(freq ** 2 for freq in freq1.values()) ** 0.5
        mag2 = sum(freq ** 2 for freq in freq2.values()) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    elif method == 'levenshtein':
        # Khoảng cách Levenshtein
        n = len(text1)
        m = len(text2)
        
        if n == 0:
            return 0.0 if m > 0 else 1.0
        if m == 0:
            return 0.0
        
        # Tạo ma trận khoảng cách
        d = [[0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if text1[i-1] == text2[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,  # xóa
                    d[i][j-1] + 1,  # chèn
                    d[i-1][j-1] + cost  # thay thế
                )
        
        distance = d[n][m]
        max_len = max(n, m)
        
        return 1.0 - (distance / max_len)

def extract_vietnamese_keywords(text: str, 
                                stop_words: Optional[Set[str]] = None, 
                                min_length: int = 2) -> List[str]:
    """
    Trích xuất từ khóa từ văn bản tiếng Việt.
    
    Args:
        text: Văn bản tiếng Việt
        stop_words: Tập các stop words, nếu None sẽ dùng danh sách mặc định
        min_length: Độ dài tối thiểu của từ khóa
        
    Returns:
        Danh sách các từ khóa tìm thấy
    """
    # Stop words tiếng Việt cơ bản
    default_stop_words = {
        'và', 'hoặc', 'của', 'là', 'có', 'trong', 'cho', 'được', 'với',
        'các', 'những', 'một', 'này', 'đó', 'không', 'đến', 'từ', 'về',
        'mà', 'tôi', 'bạn', 'họ', 'chúng', 'nó', 'lại', 'rồi', 'đã',
        'thì', 'sẽ', 'còn', 'nên', 'cần', 'phải', 'trên', 'dưới', 'theo',
        'nếu', 'khi', 'tại', 'đây', 'đấy', 'bởi', 'vì', 'cũng'
    }
    
    stop_words = stop_words or default_stop_words
    
    # Chuẩn hóa văn bản
    text = clean_text(text, lower=True, remove_punctuation=True, remove_whitespace=True)
    
    # Tokenize
    tokens = simple_vietnamese_tokenize(text)
    
    # Lọc từ khóa
    keywords = [
        token for token in tokens
        if token not in stop_words and len(token) >= min_length
    ]
    
    return keywords

def format_vietnamese_number(number: Union[int, float], 
                             decimal_places: int = 2, 
                             use_comma: bool = True,
                             currency: Optional[str] = None) -> str:
    """
    Định dạng số theo chuẩn tiếng Việt.
    
    Args:
        number: Số cần định dạng
        decimal_places: Số chữ số thập phân
        use_comma: Dùng dấu phẩy làm dấu phân cách thập phân
        currency: Đơn vị tiền tệ (VND, USD, ...)
        
    Returns:
        Chuỗi số đã định dạng
    """
    # Định dạng số
    if isinstance(number, int):
        formatted_number = f"{number:,}"
    else:
        formatted_number = f"{number:,.{decimal_places}f}"
    
    # Thay dấu phân cách
    if use_comma:
        formatted_number = formatted_number.replace('.', '#').replace(',', '.').replace('#', ',')
    
    # Thêm đơn vị tiền tệ
    if currency:
        if currency.upper() == 'VND':
            formatted_number = f"{formatted_number} đ"
        else:
            formatted_number = f"{formatted_number} {currency}"
    
    return formatted_number

def extract_vietnamese_sentences(text: str) -> List[str]:
    """
    Tách văn bản tiếng Việt thành các câu.
    
    Args:
        text: Văn bản tiếng Việt
        
    Returns:
        Danh sách các câu
    """
    # Chuẩn hóa văn bản
    text = normalize_vietnamese_text(text)
    
    # Mẫu tách câu, xử lý cẩn thận với viết tắt
    sentence_pattern = r'(?<!\d)(?<!\w\.\w\.)(?<=[\.\?\!])\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Làm sạch và lọc câu rỗng
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    
    return sentences 