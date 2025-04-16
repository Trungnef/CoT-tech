"""
Tiện ích tính toán các metrics đánh giá mô hình LLM.
Cung cấp các hàm tính precision, recall, F1, accuracy và các metrics khác.
"""

import logging
import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json

from .logging_utils import get_logger
from .text_utils import clean_text

# Thêm import mới cho METEOR và BERTScore
try:
    import nltk
    
    # Đảm bảo tải các resources NLTK cần thiết
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('wordnet', 'corpora/wordnet')
    ]
    
    for resource_name, resource_path in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                logging.warning(f"Không thể tải NLTK resource {resource_name}: {str(e)}")
    
    # Thử tạo punkt_tab nếu cần
    try:
        nltk.data.find('tokenizers/punkt_tab/english/')
    except LookupError:
        try:
            # Tạo thư mục và sao chép dữ liệu punkt nếu cần
            import os
            nltk_data_paths = nltk.data.path
            if nltk_data_paths:
                nltk_data_dir = nltk_data_paths[0]
                punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
                os.makedirs(punkt_tab_dir, exist_ok=True)
                
                # Sao chép dữ liệu từ punkt sang punkt_tab
                punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
                if os.path.exists(punkt_dir):
                    import shutil
                    for file in os.listdir(punkt_dir):
                        if file.endswith('.pickle'):
                            src = os.path.join(punkt_dir, file)
                            dst = os.path.join(punkt_tab_dir, file)
                            shutil.copy2(src, dst)
                            logging.info(f"Đã sao chép {file} từ punkt sang punkt_tab")
        except Exception as e:
            logging.warning(f"Không thể tạo punkt_tab resource: {str(e)}")
    
    # Import meteor_score sau khi đã tải resources
    from nltk.translate.meteor_score import meteor_score
except ImportError as e:
    meteor_score = None
    logging.warning(f"NLTK không được cài đặt hoặc không đầy đủ: {str(e)}. Metrics METEOR sẽ không khả dụng.")

try:
    from bert_score import BERTScorer
    bert_scorer = None  # Khởi tạo lazy khi cần
except ImportError as e:
    BERTScorer = None
    logging.warning(f"bert-score không được cài đặt: {str(e)}. Metrics BERTScore sẽ không khả dụng.")

logger = get_logger(__name__)

def calculate_binary_metrics(y_true: List[Union[int, bool]], 
                             y_pred: List[Union[int, bool]],
                             y_score: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Tính toán các metrics cho bài toán phân loại nhị phân.
    
    Args:
        y_true: Nhãn thực tế (0/1 hoặc False/True)
        y_pred: Nhãn dự đoán (0/1 hoặc False/True)
        y_score: Điểm dự đoán (xác suất) nếu có
        
    Returns:
        Dict chứa các metrics: accuracy, precision, recall, f1, auc (nếu có y_score)
    """
    # Chuẩn hóa dữ liệu
    y_true = [1 if y else 0 for y in y_true]
    y_pred = [1 if y else 0 for y in y_pred]
    
    # Kiểm tra dữ liệu
    if len(y_true) != len(y_pred):
        raise ValueError(f"Độ dài y_true ({len(y_true)}) và y_pred ({len(y_pred)}) không khớp")
    
    if len(y_true) == 0:
        logger.warning("Dữ liệu rỗng, không thể tính metrics")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    try:
        # Tính toán metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Tính AUC nếu có y_score
        if y_score is not None:
            # Kiểm tra xem có đủ classes để tính AUC không
            unique_classes = len(set(y_true))
            if unique_classes >= 2:
                metrics["auc"] = roc_auc_score(y_true, y_score)
            else:
                logger.warning("Cần ít nhất 2 classes khác nhau để tính AUC")
                metrics["auc"] = 0.0
        
        # Tính confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        })
        
        return metrics
    
    except Exception as e:
        logger.error(f"Lỗi khi tính binary metrics: {str(e)}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": str(e)
        }

def calculate_multiclass_metrics(y_true: List[int], 
                                y_pred: List[int],
                                labels: Optional[List[int]] = None,
                                average: str = 'weighted') -> Dict[str, float]:
    """
    Tính toán các metrics cho bài toán phân loại đa lớp.
    
    Args:
        y_true: Nhãn thực tế
        y_pred: Nhãn dự đoán
        labels: Danh sách các nhãn
        average: Cách tính trung bình (None, 'micro', 'macro', 'weighted')
        
    Returns:
        Dict chứa các metrics: accuracy, precision, recall, f1
    """
    # Kiểm tra dữ liệu
    if len(y_true) != len(y_pred):
        raise ValueError(f"Độ dài y_true ({len(y_true)}) và y_pred ({len(y_pred)}) không khớp")
    
    if len(y_true) == 0:
        logger.warning("Dữ liệu rỗng, không thể tính metrics")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    try:
        # Tính toán metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0)
        }
        
        # Tạo báo cáo chi tiết
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        
        # Thêm metrics cho từng class nếu không lấy average
        if average is None and labels is not None:
            for label in labels:
                label_str = str(label)
                if label_str in report:
                    for metric in ['precision', 'recall', 'f1-score']:
                        metrics[f"{label_str}_{metric}"] = report[label_str][metric]
        
        return metrics
    
    except Exception as e:
        logger.error(f"Lỗi khi tính multiclass metrics: {str(e)}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": str(e)
        }

def calculate_regression_metrics(y_true: List[float], 
                                y_pred: List[float]) -> Dict[str, float]:
    """
    Tính toán các metrics cho bài toán regression.
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        Dict chứa các metrics: mse, rmse, mae, r2, mape
    """
    # Kiểm tra dữ liệu
    if len(y_true) != len(y_pred):
        raise ValueError(f"Độ dài y_true ({len(y_true)}) và y_pred ({len(y_pred)}) không khớp")
    
    if len(y_true) == 0:
        logger.warning("Dữ liệu rỗng, không thể tính metrics")
        return {
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "mape": 0.0
        }
    
    try:
        # Chuyển đổi sang numpy array
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        
        # Tính MSE (Mean Squared Error)
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Tính RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        
        # Tính MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Tính R² (Coefficient of Determination)
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        # Tính MAPE (Mean Absolute Percentage Error)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            non_zero_mask = (y_true != 0)
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                mape = 0.0
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape)
        }
    
    except Exception as e:
        logger.error(f"Lỗi khi tính regression metrics: {str(e)}")
        return {
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "mape": 0.0,
            "error": str(e)
        }

def calculate_exact_match_accuracy(predictions: List[str], 
                                 references: List[str],
                                 normalize: bool = True,
                                 case_sensitive: bool = False,
                                 remove_punctuation: bool = True,
                                 remove_whitespace: bool = True) -> float:
    """
    Tính tỉ lệ các dự đoán khớp chính xác với tham chiếu.
    
    Args:
        predictions: Danh sách các dự đoán
        references: Danh sách các tham chiếu
        normalize: Nếu True, trả về tỉ lệ; nếu False, trả về số lượng
        case_sensitive: Nếu True, phân biệt hoa thường; nếu False, không phân biệt
        remove_punctuation: Nếu True, loại bỏ dấu câu khi so sánh
        remove_whitespace: Nếu True, chuẩn hóa khoảng trắng khi so sánh
        
    Returns:
        Tỉ lệ hoặc số lượng các dự đoán khớp chính xác
    """
    if len(predictions) != len(references):
        raise ValueError(f"Số lượng dự đoán ({len(predictions)}) khác với số lượng tham chiếu ({len(references)})")
    
    # Tiền xử lý các dự đoán và tham chiếu
    normalized_predictions = []
    normalized_references = []
    
    for p, r in zip(predictions, references):
        if not isinstance(p, str):
            p = str(p) if p is not None else ""
        if not isinstance(r, str):
            r = str(r) if r is not None else ""
        
        # Chuẩn hóa văn bản dựa trên các tham số
        p_clean = clean_text(p, 
                            lower=not case_sensitive,
                            remove_punctuation=remove_punctuation,
                            remove_whitespace=remove_whitespace)
        
        r_clean = clean_text(r, 
                            lower=not case_sensitive,
                            remove_punctuation=remove_punctuation,
                            remove_whitespace=remove_whitespace)
        
        normalized_predictions.append(p_clean)
        normalized_references.append(r_clean)
    
    # Tính số lượng các dự đoán khớp chính xác
    matches = sum(1 for p, r in zip(normalized_predictions, normalized_references) if p == r)
    
    if normalize:
        return matches / len(predictions) if len(predictions) > 0 else 0.0
    else:
        return matches

def calculate_token_overlap(predictions: List[str], 
                           references: List[str],
                           tokenizer: Callable[[str], List[str]] = None,
                           case_sensitive: bool = False,
                           remove_punctuation: bool = True,
                           remove_stopwords: bool = False) -> Dict[str, float]:
    """
    Tính độ chồng lặp token giữa dự đoán và tham chiếu.
    
    Args:
        predictions: Danh sách các dự đoán
        references: Danh sách các tham chiếu
        tokenizer: Hàm tokenize văn bản, mặc định là split
        case_sensitive: Nếu True, phân biệt hoa thường; nếu False, không phân biệt
        remove_punctuation: Nếu True, loại bỏ dấu câu khi so sánh
        remove_stopwords: Nếu True, loại bỏ stopwords khi so sánh
        
    Returns:
        Dict chứa các metrics: precision, recall, f1
    """
    if len(predictions) != len(references):
        raise ValueError(f"Số lượng dự đoán ({len(predictions)}) khác với số lượng tham chiếu ({len(references)})")
    
    # Hàm tokenize mặc định nâng cao
    def default_tokenizer(text):
        # Chuẩn hóa text trước khi tokenize
        text = clean_text(text, 
                         lower=not case_sensitive, 
                         remove_punctuation=remove_punctuation,
                         remove_whitespace=True)
        
        # Thử sử dụng word_tokenize nếu có NLTK
        try:
            import nltk
            try:
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
        except (ImportError, AttributeError):
            # Tokenize đơn giản bằng split nếu không có NLTK
            tokens = text.split()
        
        # Loại bỏ stopwords nếu cần
        if remove_stopwords:
            try:
                # Sử dụng danh sách stopwords tiếng Việt hoặc tiếng Anh
                # nếu có thư viện nltk
                import nltk
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                
                from nltk.corpus import stopwords
                
                # Ưu tiên stopwords tiếng Việt nếu có, nếu không dùng tiếng Anh
                try:
                    stop_words = set(stopwords.words('vietnamese'))
                except:
                    stop_words = set(stopwords.words('english'))
                
                tokens = [t for t in tokens if t not in stop_words]
            except ImportError:
                # Bỏ qua nếu không có nltk
                pass
        
        return tokens
    
    # Sử dụng tokenizer được cung cấp hoặc default_tokenizer
    tokenizer = tokenizer or default_tokenizer
    
    precisions = []
    recalls = []
    f1s = []
    
    for pred, ref in zip(predictions, references):
        # Xử lý các trường hợp không phải string
        if not isinstance(pred, str):
            pred = str(pred) if pred is not None else ""
        if not isinstance(ref, str):
            ref = str(ref) if ref is not None else ""
            
        # Bỏ qua các cặp rỗng
        if not pred or not ref:
            continue
            
        # Phương pháp 1: Sử dụng tập hợp (set)
        # Tokenize
        pred_tokens = set(tokenizer(pred))
        ref_tokens = set(tokenizer(ref))
        
        if not pred_tokens or not ref_tokens:
            continue
            
        # Tính intersection
        intersection = pred_tokens.intersection(ref_tokens)
        
        # Tính precision, recall, f1
        precision_set = len(intersection) / len(pred_tokens) if pred_tokens else 0
        recall_set = len(intersection) / len(ref_tokens) if ref_tokens else 0
        
        # Phương pháp 2: Sử dụng đếm tần suất (Counter) - thường chính xác hơn
        try:
            from collections import Counter
            pred_tokens_list = tokenizer(pred)
            ref_tokens_list = tokenizer(ref)
            
            pred_counter = Counter(pred_tokens_list)
            ref_counter = Counter(ref_tokens_list)
            
            # Đếm token trùng dựa trên tần suất xuất hiện
            common_counter = pred_counter & ref_counter
            
            precision_counter = sum(common_counter.values()) / sum(pred_counter.values()) if sum(pred_counter.values()) > 0 else 0
            recall_counter = sum(common_counter.values()) / sum(ref_counter.values()) if sum(ref_counter.values()) > 0 else 0
            
            # Sử dụng phương pháp 2 nếu có thể
            precision = precision_counter
            recall = recall_counter
            
        except Exception:
            # Fallback sang phương pháp 1
            precision = precision_set
            recall = recall_set
        
        # Tính F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Tính trung bình
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1s) if f1s else 0
    
    return {
        "token_precision": float(avg_precision),
        "token_recall": float(avg_recall),
        "token_f1": float(avg_f1)
    }

def calculate_llm_reasoning_metrics(scores, criteria_weights=None):
    """
    Tính toán các chỉ số cho việc đánh giá reasoning.
    
    Args:
        scores (list): Danh sách các bản ghi điểm đánh giá reasoning. 
                      Mỗi bản ghi có thể là dict hoặc là hàng DataFrame có các cột reasoning_*.
        criteria_weights (dict, optional): Trọng số cho từng tiêu chí. 
                                          Mặc định là None (cân bằng).
    
    Returns:
        dict: Dictionary chứa các chỉ số đánh giá.
    """
    if not scores or len(scores) == 0:
        return {
            "average_score": 0,
            "min_score": 0,
            "max_score": 0,
            "std_dev": 0,
            "criterion_scores": {},
            "samples_evaluated": 0
        }
    
    # Chuẩn bị DataFrame từ scores
    try:
        data = []
        for s in scores:
            # Xử lý các trường hợp khác nhau của dữ liệu đầu vào
            if isinstance(s, dict):
                # Kiểm tra xem đây là dict reasoning_scores hay row dataframe
                if "reasoning_accuracy" in s:
                    # Row dataframe với các cột flattened
                    row_data = {
                        "accuracy": s.get("reasoning_accuracy", 0),
                        "reasoning": s.get("reasoning_reasoning", 0),
                        "completeness": s.get("reasoning_completeness", 0),
                        "explanation": s.get("reasoning_explanation", 0),
                        "cultural_context": s.get("reasoning_cultural_context", 0),
                        "average": s.get("reasoning_average", 0)
                    }
                elif isinstance(s.get("reasoning_scores"), dict):
                    # Row dataframe với cột reasoning_scores là dict
                    reasoning_scores = s.get("reasoning_scores", {})
                    row_data = {
                        "accuracy": reasoning_scores.get("accuracy", 0),
                        "reasoning": reasoning_scores.get("reasoning", 0),
                        "completeness": reasoning_scores.get("completeness", 0),
                        "explanation": reasoning_scores.get("explanation", 0),
                        "cultural_context": reasoning_scores.get("cultural_context", 0),
                        "average": reasoning_scores.get("average", 0)
                    }
                elif isinstance(s.get("reasoning_scores_str"), str):
                    # Row dataframe với cột reasoning_scores_str (đã chuyển đổi để tránh lỗi)
                    # Thử phân tích JSON để lấy dữ liệu
                    try:
                        # Kiểm tra chuỗi JSON có hợp lệ không
                        reasoning_str = s.get("reasoning_scores_str", "{}")
                        
                        # Xử lý trường hợp có nhiều JSON objects bị nối với nhau
                        if reasoning_str.count('{') > 1:
                            # Tìm vị trí kết thúc của object JSON đầu tiên
                            end_pos = reasoning_str.find('}') + 1
                            # Chỉ sử dụng object đầu tiên
                            reasoning_str = reasoning_str[:end_pos]
                        
                        reasoning_scores = json.loads(reasoning_str)
                        
                        # Nếu parse JSON thành công, sử dụng dữ liệu từ JSON
                        row_data = {
                            "accuracy": reasoning_scores.get("accuracy", s.get("reasoning_accuracy", 0)),
                            "reasoning": reasoning_scores.get("reasoning", s.get("reasoning_reasoning", 0)),
                            "completeness": reasoning_scores.get("completeness", s.get("reasoning_completeness", 0)),
                            "explanation": reasoning_scores.get("explanation", s.get("reasoning_explanation", 0)),
                            "cultural_context": reasoning_scores.get("cultural_context", s.get("reasoning_cultural_context", 0)),
                            "average": reasoning_scores.get("average", s.get("reasoning_average", 0))
                        }
                    except json.JSONDecodeError:
                        # Nếu không phân tích được JSON, sử dụng các cột flattened
                        row_data = {
                            "accuracy": s.get("reasoning_accuracy", 0),
                            "reasoning": s.get("reasoning_reasoning", 0),
                            "completeness": s.get("reasoning_completeness", 0),
                            "explanation": s.get("reasoning_explanation", 0),
                            "cultural_context": s.get("reasoning_cultural_context", 0),
                            "average": s.get("reasoning_average", 0)
                        }
                    except Exception as e:
                        # Log lỗi và dùng giá trị mặc định
                        if hasattr(logging, 'getLogger'):
                            logger = logging.getLogger("metrics_utils")
                            logger.error(f"Lỗi khi phân tích reasoning_scores_str: {e}")
                        
                        row_data = {
                            "accuracy": s.get("reasoning_accuracy", 0),
                            "reasoning": s.get("reasoning_reasoning", 0),
                            "completeness": s.get("reasoning_completeness", 0),
                            "explanation": s.get("reasoning_explanation", 0),
                            "cultural_context": s.get("reasoning_cultural_context", 0),
                            "average": s.get("reasoning_average", 0)
                        }
                else:
                    # Dict trực tiếp là reasoning_scores
                    row_data = {
                        "accuracy": s.get("accuracy", 0),
                        "reasoning": s.get("reasoning", 0),
                        "completeness": s.get("completeness", 0),
                        "explanation": s.get("explanation", 0),
                        "cultural_context": s.get("cultural_context", 0),
                        "average": s.get("average", 0)
                    }
            else:
                # Xử lý trường hợp không phải dict (có thể là pandas Series)
                row_data = {
                    "accuracy": getattr(s, "reasoning_accuracy", 0),
                    "reasoning": getattr(s, "reasoning_reasoning", 0),
                    "completeness": getattr(s, "reasoning_completeness", 0),
                    "explanation": getattr(s, "reasoning_explanation", 0),
                    "cultural_context": getattr(s, "reasoning_cultural_context", 0),
                    "average": getattr(s, "reasoning_average", 0)
                }
            
            # Đảm bảo tất cả các giá trị đều là số
            for key, value in row_data.items():
                if not isinstance(value, (int, float)) or pd.isna(value):
                    row_data[key] = 0
            
            data.append(row_data)
            
        # Tạo DataFrame từ dữ liệu đã xử lý
        df = pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo DataFrame từ scores: {e}")
        return {
            "average_score": 0,
            "min_score": 0,
            "max_score": 0,
            "std_dev": 0,
            "criterion_scores": {},
            "samples_evaluated": 0,
            "error": str(e)
        }
    
    # Kiểm tra nếu không có tiêu chí nào
    if df.empty or len(df.columns) == 0:
        return {
            "average_score": 0,
            "min_score": 0,
            "max_score": 0, 
            "std_dev": 0,
            "criterion_scores": {},
            "samples_evaluated": 0
        }
    
    # Chuẩn hóa trọng số nếu có
    all_criteria = ["accuracy", "reasoning", "completeness", "explanation", "cultural_context"]
    
    if criteria_weights:
        # Đảm bảo tất cả các tiêu chí đều có trọng số
        for criterion in all_criteria:
            if criterion not in criteria_weights:
                criteria_weights[criterion] = 1.0
        
        # Chuẩn hóa trọng số
        total_weight = sum(criteria_weights.values())
        normalized_weights = {k: v/total_weight for k, v in criteria_weights.items()}
    else:
        # Nếu không có trọng số, sử dụng trọng số bằng nhau
        normalized_weights = {criterion: 1.0/len(all_criteria) for criterion in all_criteria}
    
    # Tính toán chỉ số tổng thể
    try:
        # Lấy trung bình của cột 'average' nếu có, nếu không thì tính trung bình có trọng số
        if "average" in df.columns and not df["average"].isna().all() and df["average"].sum() > 0:
            overall_scores = df["average"]
        else:
            # Tính điểm trung bình có trọng số
            overall_scores = pd.Series([0.0] * len(df), index=df.index)
            for criterion in all_criteria:
                if criterion in df.columns:
                    weight = normalized_weights.get(criterion, 0)
                    overall_scores = overall_scores.add(df[criterion] * weight)
        
        metrics = {
            "average_score": overall_scores.mean(),
            "min_score": overall_scores.min(),
            "max_score": overall_scores.max(),
            "std_dev": overall_scores.std(),
            "samples_evaluated": len(scores)
        }
        
        # Tính điểm cho từng tiêu chí
        criterion_scores = {}
        for criterion in all_criteria:
            if criterion in df.columns:
                criterion_scores[criterion] = {
                    "average": df[criterion].mean(),
                    "min": df[criterion].min(),
                    "max": df[criterion].max(),
                    "std_dev": df[criterion].std()
                }
        
        metrics["criterion_scores"] = criterion_scores
        return metrics
    
    except Exception as e:
        logger.error(f"Lỗi khi tính toán metrics: {e}")
        return {
            "average_score": 0,
            "min_score": 0,
            "max_score": 0,
            "std_dev": 0, 
            "criterion_scores": {},
            "samples_evaluated": 0,
            "error": str(e)
        }

def calculate_answer_correctness(
    predicted_answers: List[Any],
    reference_answers: List[Any],
    scoring_fn: Optional[Callable] = None,
    partial_credit: bool = False
) -> Dict[str, float]:
    """
    Đánh giá độ chính xác của các câu trả lời.
    
    Args:
        predicted_answers: Danh sách các câu trả lời dự đoán
        reference_answers: Danh sách các câu trả lời tham chiếu
        scoring_fn: Hàm tính điểm tùy chỉnh, nhận vào (pred, ref) và trả về điểm (0-1)
        partial_credit: Cho phép tính điểm một phần
        
    Returns:
        Dict chứa các metrics: accuracy, partial_score (nếu partial_credit=True)
    """
    if len(predicted_answers) != len(reference_answers):
        raise ValueError("Số lượng dự đoán và tham chiếu không khớp")
    
    # Hàm tính điểm mặc định
    if scoring_fn is None:
        scoring_fn = lambda pred, ref: 1.0 if pred == ref else 0.0
    
    # Tính điểm cho từng cặp
    scores = []
    for pred, ref in zip(predicted_answers, reference_answers):
        score = scoring_fn(pred, ref)
        scores.append(score)
    
    # Tính metrics
    accuracy = sum(1 for s in scores if s >= 0.999) / len(scores) if scores else 0
    
    metrics = {
        "accuracy": float(accuracy)
    }
    
    # Nếu cho phép điểm một phần, tính thêm partial_score
    if partial_credit:
        avg_score = sum(scores) / len(scores) if scores else 0
        metrics["partial_score"] = float(avg_score)
    
    return metrics

def calculate_latency_metrics(
    latencies: List[float],
    percentiles: List[float] = [50, 90, 95, 99]
) -> Dict[str, float]:
    """
    Tính toán các metrics về độ trễ.
    
    Args:
        latencies: Danh sách các giá trị độ trễ (ms hoặc s)
        percentiles: Danh sách các mức phân vị cần tính
        
    Returns:
        Dict chứa các metrics: mean, min, max, std và các percentiles
    """
    if not latencies:
        logger.warning("Không có dữ liệu độ trễ")
        return {
            "latency_mean": 0.0,
            "latency_min": 0.0,
            "latency_max": 0.0,
            "latency_std": 0.0
        }
    
    # Chuyển sang numpy array
    latencies_array = np.array(latencies)
    
    # Tính các metrics cơ bản
    metrics = {
        "latency_mean": float(np.mean(latencies_array)),
        "latency_min": float(np.min(latencies_array)),
        "latency_max": float(np.max(latencies_array)),
        "latency_std": float(np.std(latencies_array))
    }
    
    # Tính các percentiles
    for p in percentiles:
        metrics[f"latency_p{p}"] = float(np.percentile(latencies_array, p))
    
    return metrics

def calculate_rouge_scores(predictions: List[str], 
                         references: List[str],
                         rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']) -> Dict[str, float]:
    """
    Tính các chỉ số ROUGE (Recall-Oriented Understudy for Gisting Evaluation) cho đánh giá tóm tắt văn bản.
    
    Args:
        predictions: Danh sách các dự đoán
        references: Danh sách các tham chiếu
        rouge_types: Các loại chỉ số ROUGE cần tính
        
    Returns:
        Dict chứa các chỉ số ROUGE
    """
    if len(predictions) != len(references):
        raise ValueError(f"Số lượng dự đoán ({len(predictions)}) khác với số lượng tham chiếu ({len(references)})")
    
    try:
        from rouge import Rouge
    except ImportError:
        logger.warning("Không thể tính ROUGE score: Thiếu thư viện 'rouge'. Cài đặt với 'pip install rouge'")
        return {
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "import_error": "Thiếu thư viện 'rouge'"
        }
    
    rouge = Rouge(metrics=rouge_types)
    all_scores = {
        "rouge1_f": [],
        "rouge1_p": [],
        "rouge1_r": [],
        "rouge2_f": [],
        "rouge2_p": [],
        "rouge2_r": [],
        "rougeL_f": [],
        "rougeL_p": [],
        "rougeL_r": []
    }
    
    # Tính điểm ROUGE cho từng cặp dự đoán/tham chiếu
    for pred, ref in zip(predictions, references):
        # Xử lý các trường hợp không phải string
        if not isinstance(pred, str):
            pred = str(pred) if pred is not None else ""
        if not isinstance(ref, str):
            ref = str(ref) if ref is not None else ""
            
        # Bỏ qua các cặp rỗng hoặc quá ngắn
        if not pred or not ref or len(pred.split()) < 1 or len(ref.split()) < 1:
            continue
            
        try:
            # Tính điểm ROUGE
            scores = rouge.get_scores(pred, ref)[0]
            
            # Lưu các điểm vào danh sách để tính trung bình
            for rouge_type in scores:
                for metric in ['f', 'p', 'r']:
                    key = f"{rouge_type}_{metric}"
                    if key in all_scores:
                        all_scores[key].append(scores[rouge_type][metric])
        except Exception as e:
            logger.debug(f"Lỗi khi tính ROUGE cho một cặp: {str(e)}")
            continue
    
    # Tính trung bình cho tất cả các điểm
    result = {}
    for key, values in all_scores.items():
        if values:
            result[key] = float(np.mean(values))
        else:
            result[key] = 0.0
    
    return result

def calculate_bleu_scores(predictions: List[str], 
                        references: List[str],
                        max_ngram: int = 4,
                        lowercase: bool = True) -> Dict[str, float]:
    """
    Tính điểm BLEU (Bilingual Evaluation Understudy) cho đánh giá chất lượng dịch máy.
    
    Args:
        predictions: Danh sách các dự đoán
        references: Danh sách các tham chiếu
        max_ngram: Độ dài n-gram tối đa cần xét (1 đến 4)
        lowercase: Có chuyển văn bản về chữ thường không
        
    Returns:
        Dict chứa các chỉ số BLEU
    """
    if len(predictions) != len(references):
        raise ValueError(f"Số lượng dự đoán ({len(predictions)}) khác với số lượng tham chiếu ({len(references)})")
    
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Tải các resource cần thiết nếu chưa có
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    except ImportError:
        logger.warning("Không thể tính BLEU score: Thiếu thư viện 'nltk'. Cài đặt với 'pip install nltk'")
        return {
            "bleu": 0.0, 
            "bleu1": 0.0, 
            "bleu2": 0.0, 
            "bleu3": 0.0, 
            "bleu4": 0.0,
            "import_error": "Thiếu thư viện 'nltk'"
        }
    
    # Khởi tạo smoothing function để tránh điểm 0 khi không có n-gram match
    smoothing = SmoothingFunction().method1
    
    # Danh sách điểm BLEU cho từng n-gram (1-4)
    bleu_scores = {f"bleu{i}": [] for i in range(1, max_ngram + 1)}
    bleu_scores["bleu"] = []  # Điểm BLEU tổng hợp
    
    # Tính điểm BLEU cho từng cặp dự đoán/tham chiếu
    for pred, ref in zip(predictions, references):
        # Xử lý các trường hợp không phải string
        if not isinstance(pred, str):
            pred = str(pred) if pred is not None else ""
        if not isinstance(ref, str):
            ref = str(ref) if ref is not None else ""
            
        # Bỏ qua các cặp rỗng
        if not pred or not ref:
            continue
            
        # Hạ chữ hoa nếu cần
        if lowercase:
            pred = pred.lower()
            ref = ref.lower()
        
        try:
            # Tokenize
            pred_tokens = nltk.word_tokenize(pred)
            ref_tokens = [nltk.word_tokenize(ref)]  # BLEU cần list của các references
            
            # Tính điểm BLEU tổng hợp (trung bình các n-gram)
            weights = [1.0/max_ngram] * max_ngram
            bleu = sentence_bleu(ref_tokens, pred_tokens, 
                               weights=weights,
                               smoothing_function=smoothing)
            bleu_scores["bleu"].append(bleu)
            
            # Tính điểm BLEU cho từng n-gram
            for i in range(1, max_ngram + 1):
                weights = [0.0] * max_ngram
                weights[i-1] = 1.0
                bleu_i = sentence_bleu(ref_tokens, pred_tokens, 
                                      weights=weights,
                                      smoothing_function=smoothing)
                bleu_scores[f"bleu{i}"].append(bleu_i)
        except Exception as e:
            logger.debug(f"Lỗi khi tính BLEU cho một cặp: {str(e)}")
            continue
    
    # Tính trung bình cho tất cả các điểm
    result = {}
    for key, values in bleu_scores.items():
        if values:
            result[key] = float(np.mean(values))
        else:
            result[key] = 0.0
    
    return result

def calculate_meteor_score(predictions: List[str], 
                          references: List[str],
                          alpha: float = 0.9,
                          beta: float = 3.0,
                          gamma: float = 0.5) -> Dict[str, float]:
    """
    Tính điểm METEOR (Metric for Evaluation of Translation with Explicit ORdering).
    
    Args:
        predictions: Danh sách các dự đoán
        references: Danh sách các tham chiếu
        alpha, beta, gamma: Tham số của METEOR
        
    Returns:
        Dict chứa điểm METEOR
    """
    # Check if NLTK's meteor_score is available
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        logger.warning("NLTK không được cài đặt. Không thể tính điểm METEOR.")
        return {
            "meteor": 0.0, 
            "meteor_scores": [0.0] * len(predictions) if predictions else [],
            "error": "NLTK not installed"
        }
    
    if len(predictions) != len(references):
        raise ValueError(f"Số lượng dự đoán ({len(predictions)}) và tham chiếu ({len(references)}) không khớp")
    
    try:
        # Ensure NLTK data is available
        import nltk
        try:
            nltk.data.find('wordnet')
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            
        # Tính METEOR score cho từng cặp
        meteor_scores = []
        
        for i in range(len(predictions)):
            try:
                # Chuẩn hóa prediction và reference
                prediction = str(predictions[i]).strip() if predictions[i] is not None else ""
                reference = str(references[i]).strip() if references[i] is not None else ""
                
                if not prediction or not reference:
                    meteor_scores.append(0.0)
                    continue
                    
                # Thử tokenize bằng word_tokenize
                try:
                    from nltk.tokenize import word_tokenize
                    pred_tokens = word_tokenize(prediction)
                    ref_tokens = word_tokenize(reference)
                except (ImportError, LookupError):
                    # Fallback sang split() nếu word_tokenize không khả dụng
                    pred_tokens = prediction.lower().split()
                    ref_tokens = reference.lower().split()
                    
                if not pred_tokens or not ref_tokens:
                    meteor_scores.append(0.0)
                    continue
                    
                # Tính METEOR score
                try:
                    # Thử sử dụng single_meteor_score nếu có
                    from nltk.translate.meteor_score import single_meteor_score
                    score = single_meteor_score(ref_tokens, pred_tokens)
                except (ImportError, AttributeError):
                    # Fallback sang meteor_score
                    score = meteor_score([[ref_tokens]], pred_tokens)
                    
                meteor_scores.append(score)
                
            except Exception as e:
                logger.debug(f"Lỗi khi tính METEOR cho cặp {i}: {str(e)}")
                # Fallback: Tính tương tự METEOR bằng cách tính F-score
                try:
                    # Không sử dụng word_tokenize để tránh lỗi
                    ref_words = references[i].lower().split()
                    pred_words = predictions[i].lower().split()
                    
                    # Sử dụng Counter để đếm tần suất
                    from collections import Counter
                    ref_counter = Counter(ref_words)
                    pred_counter = Counter(pred_words)
                    
                    # Tìm tokens chung
                    common_counter = ref_counter & pred_counter
                    
                    # Tính precision, recall
                    ref_count = sum(ref_counter.values())
                    pred_count = sum(pred_counter.values())
                    common_count = sum(common_counter.values())
                    
                    if pred_count == 0:
                        precision = 0
                    else:
                        precision = common_count / pred_count
                        
                    if ref_count == 0:
                        recall = 0
                    else:
                        recall = common_count / ref_count
                        
                    # Tính F-score với alpha=0.9 (tương tự METEOR)
                    if precision + recall > 0:
                        meteor_fallback = precision * recall / (alpha * precision + (1 - alpha) * recall)
                        meteor_scores.append(meteor_fallback)
                    else:
                        meteor_scores.append(0.0)
                except Exception:
                    meteor_scores.append(0.0)
        
        # Tính trung bình
        avg_meteor = np.mean(meteor_scores) if meteor_scores else 0.0
        
        return {
            "meteor": float(avg_meteor),
            "meteor_scores": [float(score) for score in meteor_scores]
        }
    
    except Exception as e:
        logger.error(f"Lỗi khi tính điểm METEOR: {str(e)}")
        return {
            "meteor": 0.0,
            "meteor_scores": [0.0] * len(predictions) if predictions else [],
            "error": str(e)
        }

def calculate_text_generation_metrics(predictions: List[str], 
                                    references: List[str],
                                    include_bleu: bool = True,
                                    include_rouge: bool = True,
                                    include_token_overlap: bool = True,
                                    include_meteor: bool = True,
                                    include_bertscore: bool = False,
                                    case_sensitive: bool = False,
                                    remove_punctuation: bool = True) -> Dict[str, float]:
    """
    Tính toán tất cả các metrics liên quan đến sinh văn bản.
    
    Args:
        predictions: Danh sách các dự đoán
        references: Danh sách các tham chiếu
        include_bleu: Có tính BLEU không
        include_rouge: Có tính ROUGE không
        include_token_overlap: Có tính token overlap không
        include_meteor: Có tính METEOR không 
        include_bertscore: Có tính BERTScore không
        case_sensitive: Có phân biệt chữ hoa/thường không
        remove_punctuation: Có loại bỏ dấu câu không
        
    Returns:
        Dict chứa tất cả các metrics
    """
    if not predictions or not references:
        logger.warning("Danh sách dự đoán hoặc tham chiếu rỗng")
        return {}
    
    # Chuẩn hóa độ dài danh sách nếu cần
    if len(predictions) != len(references):
        min_len = min(len(predictions), len(references))
        logger.warning(f"Độ dài không khớp: {len(predictions)} dự đoán vs {len(references)} tham chiếu. Cắt xuống {min_len}")
        predictions = predictions[:min_len]
        references = references[:min_len]
    
    all_metrics = {}
    
    # Tính BLEU
    if include_bleu:
        try:
            bleu_metrics = calculate_bleu_scores(
                predictions, references, 
                lowercase=(not case_sensitive)
            )
            all_metrics.update(bleu_metrics)
        except Exception as e:
            logger.error(f"Lỗi khi tính BLEU: {str(e)}")
            all_metrics.update({"bleu_error": str(e)})
    
    # Tính ROUGE
    if include_rouge:
        try:
            rouge_metrics = calculate_rouge_scores(predictions, references)
            all_metrics.update(rouge_metrics)
        except Exception as e:
            logger.error(f"Lỗi khi tính ROUGE: {str(e)}")
            all_metrics.update({"rouge_error": str(e)})
    
    # Tính token overlap
    if include_token_overlap:
        try:
            overlap_metrics = calculate_token_overlap(
                predictions, references,
                case_sensitive=case_sensitive,
                remove_punctuation=remove_punctuation
            )
            all_metrics.update(overlap_metrics)
        except Exception as e:
            logger.error(f"Lỗi khi tính token overlap: {str(e)}")
            all_metrics.update({"overlap_error": str(e)})
    
    # Tính METEOR
    if include_meteor and meteor_score is not None:
        try:
            meteor_metrics = calculate_meteor_score(predictions, references)
            all_metrics.update(meteor_metrics)
        except Exception as e:
            logger.error(f"Lỗi khi tính METEOR: {str(e)}")
            all_metrics.update({"meteor_error": str(e)})
    
    # Tính BERTScore
    if include_bertscore and BERTScorer is not None:
        try:
            bertscore_metrics = calculate_bertscore(predictions, references)
            all_metrics.update(bertscore_metrics)
        except Exception as e:
            logger.error(f"Lỗi khi tính BERTScore: {str(e)}")
            all_metrics.update({"bertscore_error": str(e)})
    
    return all_metrics 

def calculate_bertscore(predictions: List[str], 
                       references: List[str],
                       lang: str = "vi",
                       model_type: str = "microsoft/mdeberta-v3-base",
                       batch_size: int = 8,
                       rescale_with_baseline: bool = True) -> Dict[str, float]:
    """
    Tính điểm BERTScore sử dụng bert-score package.
    
    Args:
        predictions: Danh sách các dự đoán
        references: Danh sách các tham chiếu
        lang: Ngôn ngữ của văn bản
        model_type: Mô hình BERT để sử dụng
        batch_size: Kích thước batch
        rescale_with_baseline: Có áp dụng rescaling không
        
    Returns:
        Dict chứa điểm BERTScore (precision, recall, F1)
    """
    # Try to import bert_score directly
    try:
        from bert_score import score as bert_score_func
    except ImportError:
        logger.warning("bert-score không được cài đặt. Không thể tính BERTScore.")
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0, 
            "bertscore_f1": 0.0,
            "error": "bert-score not installed"
        }
    
    if len(predictions) != len(references):
        raise ValueError(f"Số lượng dự đoán ({len(predictions)}) và tham chiếu ({len(references)}) không khớp")
    
    try:
        # Check if torch is available and set device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Adjust batch size for CPU to avoid OOM
        if device == "cpu":
            batch_size = min(batch_size, 4)
            logger.info(f"Sử dụng CPU cho BERTScore với batch_size={batch_size}")
        
        # Filter out empty predictions/references
        valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                      if p and r and isinstance(p, str) and isinstance(r, str)]
        
        if not valid_pairs:
            logger.warning("Không có cặp dự đoán/tham chiếu hợp lệ cho BERTScore")
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "error": "No valid prediction/reference pairs"
            }
        
        valid_preds, valid_refs = zip(*valid_pairs)
        
        # Calculate BERTScore
        try:
            P, R, F1 = bert_score_func(
                valid_preds, 
                valid_refs, 
                lang=lang,
                model_type=model_type,
                batch_size=batch_size,
                device=device,
                rescale_with_baseline=rescale_with_baseline
            )
            
            # Convert to Python objects
            avg_precision = P.mean().item()
            avg_recall = R.mean().item() 
            avg_f1 = F1.mean().item()
            
            precision_scores = P.tolist()
            recall_scores = R.tolist()
            f1_scores = F1.tolist()
            
            # Fill in scores for all original predictions
            all_precision = []
            all_recall = []
            all_f1 = []
            
            score_idx = 0
            for p, r in zip(predictions, references):
                if p and r and isinstance(p, str) and isinstance(r, str):
                    all_precision.append(precision_scores[score_idx])
                    all_recall.append(recall_scores[score_idx])
                    all_f1.append(f1_scores[score_idx])
                    score_idx += 1
                else:
                    all_precision.append(0.0)
                    all_recall.append(0.0)
                    all_f1.append(0.0)
            
            return {
                "bertscore_precision": avg_precision,
                "bertscore_recall": avg_recall,
                "bertscore_f1": avg_f1,
                "bertscore_precision_scores": all_precision,
                "bertscore_recall_scores": all_recall,
                "bertscore_f1_scores": all_f1
            }
        except Exception as e:
            logger.error(f"Lỗi khi tính BERTScore: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "error": str(e)
            }
    
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo để tính BERTScore: {str(e)}")
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "error": str(e)
        } 