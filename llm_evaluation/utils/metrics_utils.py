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
                                 case_sensitive: bool = False) -> float:
    """
    Tính tỉ lệ các dự đoán khớp chính xác với tham chiếu.
    
    Args:
        predictions: Danh sách các dự đoán
        references: Danh sách các tham chiếu
        normalize: Nếu True, trả về tỉ lệ; nếu False, trả về số lượng
        case_sensitive: Nếu True, phân biệt hoa thường; nếu False, không phân biệt
        
    Returns:
        Tỉ lệ hoặc số lượng các dự đoán khớp chính xác
    """
    if len(predictions) != len(references):
        raise ValueError(f"Số lượng dự đoán ({len(predictions)}) khác với số lượng tham chiếu ({len(references)})")
    
    if not case_sensitive:
        predictions = [p.lower() if isinstance(p, str) else p for p in predictions]
        references = [r.lower() if isinstance(r, str) else r for r in references]
    
    matches = sum(1 for p, r in zip(predictions, references) if p == r)
    
    if normalize:
        return matches / len(predictions) if len(predictions) > 0 else 0.0
    else:
        return matches

def calculate_token_overlap(predictions: List[str], 
                           references: List[str],
                           tokenizer: Callable[[str], List[str]] = None) -> Dict[str, float]:
    """
    Tính độ chồng lặp token giữa dự đoán và tham chiếu.
    
    Args:
        predictions: Danh sách các dự đoán
        references: Danh sách các tham chiếu
        tokenizer: Hàm tokenize văn bản, mặc định là split
        
    Returns:
        Dict chứa các metrics: precision, recall, f1
    """
    if len(predictions) != len(references):
        raise ValueError(f"Số lượng dự đoán ({len(predictions)}) khác với số lượng tham chiếu ({len(references)})")
    
    if tokenizer is None:
        tokenizer = lambda x: x.lower().split()
    
    precisions = []
    recalls = []
    f1s = []
    
    for pred, ref in zip(predictions, references):
        # Bỏ qua các cặp rỗng
        if not pred or not ref:
            continue
            
        # Tokenize
        pred_tokens = set(tokenizer(pred))
        ref_tokens = set(tokenizer(ref))
        
        # Tính intersection
        intersection = pred_tokens.intersection(ref_tokens)
        
        # Tính precision, recall, f1
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
        recall = len(intersection) / len(ref_tokens) if ref_tokens else 0
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