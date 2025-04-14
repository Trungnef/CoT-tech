"""
Result Analyzer cho hệ thống đánh giá LLM.
Phân tích các kết quả từ quá trình đánh giá và tính toán các metrics.
Hỗ trợ đánh giá chất lượng suy luận qua API Groq và các metrics khác.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
import traceback
import re
import random

# Import các module cần thiết
try:
    from ..core.model_interface import generate_text
except ImportError:
    # Fallback khi chạy module trực tiếp
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.model_interface import generate_text

# Thiết lập logging
logger = logging.getLogger(__name__)

class ResultAnalyzer:
    """
    Phân tích kết quả đánh giá mô hình ngôn ngữ.
    Tính toán các metrics và phân tích chất lượng suy luận.
    """
    
    def __init__(self, 
                 results_df: Optional[pd.DataFrame] = None,
                 reasoning_evaluation_config: Optional[Dict[str, Any]] = None,
                 reasoning_model: str = "groq/llama3-70b-8192",
                 language: str = "vietnamese",
                 similarity_model: Optional[str] = None,
                 verbose: bool = True):
        """
        Khởi tạo ResultAnalyzer.
        
        Args:
            results_df (pd.DataFrame, optional): DataFrame kết quả để phân tích
            reasoning_evaluation_config (Dict, optional): Cấu hình đánh giá suy luận
            reasoning_model (str): Mô hình dùng để đánh giá chất lượng suy luận
            language (str): Ngôn ngữ sử dụng trong đánh giá
            similarity_model (str, optional): Mô hình để tính toán semantic similarity
            verbose (bool): Hiển thị thông tin chi tiết trong quá trình phân tích
        """
        self.results_df = results_df
        self.reasoning_config = reasoning_evaluation_config or {}
        self.reasoning_model = self.reasoning_config.get("model", reasoning_model)
        self.language = language.lower()
        self.similarity_model = similarity_model
        self.verbose = verbose
        self.sample_size = self.reasoning_config.get("sample_size", 10)
        
        # Các tiêu chí đánh giá mới
        self.reasoning_criteria = {
            "accuracy": "Độ chính xác (Accuracy)",
            "reasoning_consistency": "Độ suy luận hợp lý (Reasoning Consistency)",
            "consistency": "Tính nhất quán (Consistency)",
            "difficulty_performance": "Hiệu suất trên độ khó (Difficulty Performance)",
            "context_adherence": "Độ phù hợp ngữ cảnh (Context Adherence)"
        }

        # Cấu trúc prompt đánh giá
        self.reasoning_eval_template = """
# HƯỚNG DẪN ĐÁNH GIÁ CHẤT LƯỢNG ĐẦU RA CỦA MÔ HÌNH LLM

Bạn là một chuyên gia đánh giá chất lượng đầu ra của các mô hình ngôn ngữ lớn (LLMs). Nhiệm vụ của bạn là đánh giá câu trả lời của một mô hình LLM cho một bài toán cụ thể dựa trên các tiêu chí khách quan và rõ ràng.

## TIÊU CHÍ ĐÁNH GIÁ

1. **Độ chính xác (Accuracy)**
   - Câu trả lời có đúng về mặt nội dung và kết quả so với đáp án chuẩn không?
   - Với bài toán số học: kết quả cuối cùng có đúng không?
   - Với bài toán lý luận: kết luận có chính xác không?
   - Điểm 5: Hoàn toàn chính xác
   - Điểm 1: Hoàn toàn sai

2. **Độ suy luận hợp lý (Reasoning Consistency)**
   - Quá trình lập luận có logic và có cấu trúc rõ ràng không?
   - Các bước suy luận có thể theo dõi và kiểm chứng được không?
   - Có sai sót logic trong các bước lập luận không?
   - Điểm 5: Lập luận hoàn hảo, rõ ràng, và đầy đủ
   - Điểm 1: Lập luận rời rạc, mâu thuẫn hoặc sai cơ bản

3. **Tính nhất quán (Consistency)**
   - Câu trả lời có nhất quán từ đầu đến cuối không?
   - Không có mâu thuẫn giữa các phần trong câu trả lời?
   - Các định nghĩa và ký hiệu được sử dụng nhất quán?
   - Điểm 5: Hoàn toàn nhất quán
   - Điểm 1: Nhiều mâu thuẫn nội bộ

4. **Hiệu suất phù hợp với độ khó (Difficulty Performance)**
   - Câu trả lời có phù hợp với độ khó của bài toán không?
   - Mô hình có xử lý đầy đủ độ phức tạp của bài toán không?
   - Điểm 5: Xử lý xuất sắc bài toán theo đúng độ khó
   - Điểm 1: Không đáp ứng được yêu cầu cơ bản của bài toán

5. **Độ phù hợp ngữ cảnh (Context Adherence)**
   - Câu trả lời có tận dụng tốt ngữ cảnh/ví dụ được cung cấp không?
   - Áp dụng đúng các mẫu/cấu trúc từ ngữ cảnh vào bài giải?
   - Điểm 5: Tận dụng tối đa ngữ cảnh một cách hiệu quả
   - Điểm 1: Hoàn toàn không sử dụng ngữ cảnh được cung cấp

## BÀI TOÁN CẦN GIẢI QUYẾT

{question}

## ĐÁP ÁN CHUẨN

{correct_answer}

## CÂU TRẢ LỜI CỦA MÔ HÌNH CẦN ĐÁNH GIÁ

{model_answer}

## ĐÁNH GIÁ THEO THANG ĐIỂM 5

Hãy đánh giá và cho điểm từ 1-5 cho từng tiêu chí, trong đó 1 là kém nhất và 5 là tốt nhất:

1. Độ chính xác (accuracy): ?/5
2. Độ suy luận hợp lý (reasoning): ?/5
3. Tính nhất quán (completeness): ?/5
4. Hiệu suất phù hợp với độ khó (explanation): ?/5
5. Độ phù hợp ngữ cảnh (cultural_context): ?/5

Điểm trung bình (average): ?/5

## GIẢI THÍCH CHI TIẾT

- Độ chính xác: [giải thích chi tiết]
- Độ suy luận hợp lý: [giải thích chi tiết]
- Tính nhất quán: [giải thích chi tiết]
- Hiệu suất phù hợp với độ khó: [giải thích chi tiết]
- Độ phù hợp ngữ cảnh: [giải thích chi tiết]

## KẾT LUẬN TỔNG THỂ

[nhận xét tổng quan về chất lượng câu trả lời]
"""

    def analyze(self) -> pd.DataFrame:
        """
        Phân tích kết quả và thực hiện đánh giá.
        Phương thức này gọi khi ResultAnalyzer được sử dụng từ Evaluator.
        
        Returns:
            pd.DataFrame: DataFrame với kết quả phân tích
        """
        if self.results_df is None or len(self.results_df) == 0:
            logger.warning("Không có dữ liệu để phân tích")
            return pd.DataFrame()
        
        if self.verbose:
            logger.info(f"🔍 Bắt đầu phân tích {len(self.results_df)} kết quả đánh giá")
        
        # Kiểm tra các cột cần thiết
        required_cols = ['model_name', 'prompt_type', 'question_text']
        missing_cols = [col for col in required_cols if col not in self.results_df.columns]
        if missing_cols:
            logger.warning(f"Thiếu các cột cần thiết cho phân tích: {missing_cols}")
            logger.info(f"Các cột hiện có: {list(self.results_df.columns)}")
            return self.results_df
        
        # Khởi tạo dictionary metrics chính với các khóa chính là dict rỗng
        analysis_results = {
            'basic_metrics': {},
            'model_prompt_metrics': {},
            'question_type_metrics': {},
            'accuracy_metrics': {},
            'reasoning_metrics': {},
            'consistency_metrics': {},
            'difficulty_metrics': {},
            'context_metrics': {}
        }
        
        # Tính toán metrics cơ bản
        analysis_results['basic_metrics'] = self._compute_basic_metrics(self.results_df)
        
        # Tính toán metrics theo model và prompt type
        analysis_results['model_prompt_metrics'] = self._compute_metrics_by_model_prompt(self.results_df)
        
        # Tính toán metrics theo loại câu hỏi (nếu có thông tin)
        if 'question_type' in self.results_df.columns:
            analysis_results['question_type_metrics'] = self._compute_metrics_by_question_type(self.results_df)
        
        # Đánh giá theo các tiêu chí mới
        # 1. Accuracy
        if 'is_correct' in self.results_df.columns:
            analysis_results['accuracy_metrics'] = self._compute_accuracy_metrics(self.results_df)
        
        # 2. Reasoning Consistency
        if any(col.startswith('reasoning_') and col != 'reasoning_scores_str' for col in self.results_df.columns):
            analysis_results['reasoning_metrics'] = self._compute_reasoning_metrics(self.results_df)
        
        # 3. Consistency
        if self.results_df['prompt_type'].str.contains('consistency|cot_self_consistency', case=False).any():
            if self.verbose:
                logger.info("Đánh giá tính nhất quán trong các self-consistency runs")
            
            try:
                self.results_df = self.evaluate_consistency(self.results_df)
                analysis_results["consistency_metrics"] = self._compute_consistency_metrics(self.results_df)
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá tính nhất quán: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # 4. Performance on different difficulty levels
        analysis_results['difficulty_metrics'] = self._compute_difficulty_metrics(self.results_df)
        
        # 5. Context Adherence
        analysis_results['context_metrics'] = self._compute_context_adherence_metrics(self.results_df)
        
        # Lưu kết quả phân tích vào thuộc tính
        self.analysis_results = analysis_results
        
        return self.results_df
    
    def analyze_errors(self, 
                   results_df: pd.DataFrame, 
                   sample_size: int = 50,
                   random_seed: int = 42) -> pd.DataFrame:
        """
        Phân tích và phân loại các lỗi trong câu trả lời của model.
        
        Các loại lỗi được phân loại bao gồm:
        - Lỗi kiến thức (Knowledge Error): Thiếu thông tin hoặc kiến thức không chính xác
        - Lỗi suy luận (Reasoning Error): Lỗi trong quá trình suy luận, sai logic
        - Lỗi tính toán (Calculation Error): Sai số học hoặc tính toán
        - Lỗi không trả lời (Non-answer): Từ chối hoặc không trả lời
        - Lỗi lạc đề (Off-topic): Trả lời không liên quan đến câu hỏi
        - Lỗi hiểu nhầm (Misunderstanding): Hiểu sai câu hỏi
        - Lỗi khác (Other): Các lỗi không thuộc các loại trên
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            sample_size (int): Số lượng mẫu cần phân tích
            random_seed (int): Seed ngẫu nhiên cho việc lấy mẫu
            
        Returns:
            pd.DataFrame: DataFrame đã bổ sung phân loại lỗi
        """
        # Kiểm tra xem có các cột cần thiết không
        required_cols = ['question_text', 'response', 'is_correct']
        for col in required_cols:
            if col not in results_df.columns:
                logger.warning(f"Không thể phân tích lỗi: thiếu cột '{col}'")
                return results_df
                
        # Đảm bảo có các cột cần thiết
        if 'error_type' not in results_df.columns:
            results_df['error_type'] = ''
            
        if 'error_explanation' not in results_df.columns:
            results_df['error_explanation'] = ''
        
        # Lọc các câu trả lời sai
        if 'is_correct' not in results_df.columns:
            logger.warning("Không có cột 'is_correct' để phân tích lỗi")
            return results_df
        
        # Lọc các câu trả lời sai chưa được phân tích lỗi
        error_rows = (results_df['is_correct'] == False) & (results_df['error_type'] == '')
        
        # Kiểm tra xem có câu trả lời sai để phân tích không
        if not error_rows.any():
            logger.info("Không tìm thấy câu trả lời sai chưa được phân tích")
            return results_df
        
        error_indices = results_df[error_rows].index.tolist()
        
        # Lấy mẫu ngẫu nhiên nếu cần
        np.random.seed(random_seed)
        if len(error_indices) > sample_size:
            sample_indices = np.random.choice(error_indices, size=sample_size, replace=False)
        else:
            sample_indices = error_indices
        
        logger.info(f"Phân tích lỗi cho {len(sample_indices)}/{len(error_indices)} câu trả lời sai")
        
        # Phân tích lỗi cho từng mẫu
        for i, idx in enumerate(sample_indices):
            row = results_df.loc[idx]
            
            question = row['question_text']
            model_answer = row['response']
            correct_answer = row['correct_answer'] if 'correct_answer' in row else None
            
            if self.verbose:
                logger.info(f"Phân tích lỗi mẫu {i+1}/{len(sample_indices)}: model={row['model_name']}, prompt={row['prompt_type']}")
            
            try:
                # Phân tích lỗi
                error_result = self._analyze_single_error(question, model_answer, correct_answer)
                
                # Cập nhật DataFrame
                results_df.at[idx, 'error_type'] = error_result.get('error_type', 'Unknown')
                results_df.at[idx, 'error_explanation'] = error_result.get('explanation', '')
            
            except Exception as e:
                logger.error(f"Lỗi khi phân tích lỗi cho mẫu {idx}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Gán giá trị mặc định
                results_df.at[idx, 'error_type'] = 'Analysis Error'
                results_df.at[idx, 'error_explanation'] = f"Lỗi khi phân tích: {str(e)}"
        
        return results_df
    
    def _analyze_single_error(self, 
                            question: str, 
                            model_answer: str, 
                            correct_answer: str = None) -> Dict[str, Any]:
        """
        Phân tích lỗi cho một mẫu câu hỏi/câu trả lời.
        
        Args:
            question (str): Câu hỏi
            model_answer (str): Câu trả lời của model
            correct_answer (str, optional): Đáp án đúng nếu có
            
        Returns:
            Dict: Kết quả phân tích lỗi
        """
        try:
            # Cắt bớt độ dài nếu quá dài
            max_length = 4000  # Giới hạn độ dài để tránh vượt quá context window
            if len(model_answer) > max_length:
                logger.warning(f"Cắt bớt câu trả lời ({len(model_answer)} -> {max_length} ký tự)")
                model_answer = model_answer[:max_length] + "..."
            
            # Tạo prompt phân tích lỗi
            if self.language.lower() == "vietnamese":
                correct_answer_part = f"\nĐÁP ÁN ĐÚNG:\n{correct_answer}" if correct_answer else ""
                
                error_analysis_prompt = f"""
Bạn là một chuyên gia phân tích lỗi trong câu trả lời của mô hình ngôn ngữ. Hãy phân tích và phân loại lỗi trong câu trả lời dưới đây.

CÂU HỎI:
{question}
{correct_answer_part}

CÂU TRẢ LỜI CỦA MÔ HÌNH:
{model_answer}

Phân loại lỗi vào MỘT trong các danh mục sau:
1. Lỗi kiến thức (Knowledge Error): Thiếu thông tin hoặc kiến thức không chính xác
2. Lỗi suy luận (Reasoning Error): Lỗi trong quá trình suy luận, sai logic
3. Lỗi tính toán (Calculation Error): Sai số học hoặc tính toán
4. Lỗi không trả lời (Non-answer): Từ chối hoặc không trả lời
5. Lỗi lạc đề (Off-topic): Trả lời không liên quan đến câu hỏi
6. Lỗi hiểu nhầm (Misunderstanding): Hiểu sai câu hỏi
7. Lỗi khác (Other): Các lỗi không thuộc các loại trên

Loại lỗi: [Chọn MỘT loại lỗi từ danh sách trên]

Giải thích ngắn gọn:
[Giải thích tại sao câu trả lời bị coi là sai và thuộc loại lỗi đã chọn]
"""
            else:
                correct_answer_part = f"\nCORRECT ANSWER:\n{correct_answer}" if correct_answer else ""
                
                error_analysis_prompt = f"""
You are an expert analyzing errors in language model responses. Analyze and categorize the error in the response below.

QUESTION:
{question}
{correct_answer_part}

MODEL RESPONSE:
{model_answer}

Categorize the error into ONE of the following categories:
1. Knowledge Error: Missing information or incorrect knowledge
2. Reasoning Error: Errors in the reasoning process, logical fallacies
3. Calculation Error: Mathematical or computational mistakes
4. Non-answer: Refusing or failing to provide an answer
5. Off-topic: Response unrelated to the question
6. Misunderstanding: Misinterpreting the question
7. Other: Errors that don't fall into the above categories

Error Type: [Select ONE error type from the list above]

Brief Explanation:
[Explain why the answer is incorrect and why it belongs to the selected error type]
"""
            
            # Sử dụng model API để phân tích lỗi
            use_groq = self.reasoning_config.get("use_groq", True)
            if use_groq:
                # Sử dụng Groq API
                from core.model_interface import generate_text
                
                # Lấy tên model Groq
                model_name = "groq"
                config = {
                    "model": self.reasoning_config.get("models", {}).get(
                        "error_analysis", "llama3-70b-8192"
                    ),
                    "temperature": 0.1,  # Thấp để đảm bảo phân loại nhất quán
                    "max_tokens": 1024
                }
                
                # Gọi API
                logger.debug("Phân tích lỗi bằng Groq API")
                response_text, stats = generate_text(model_name, error_analysis_prompt, config)
                
                if stats.get("has_error", False):
                    logger.error(f"Lỗi khi gọi Groq API: {stats.get('error_message')}")
                    # Fallback về giá trị mặc định
                    return {
                        "error_type": "Unknown",
                        "explanation": f"[Lỗi phân tích: {stats.get('error_message')}]"
                    }
            else:
                # TODO: Sử dụng model khác nếu cần
                logger.warning("Chỉ hỗ trợ Groq API để phân tích lỗi")
                response_text = ""
            
            # Phân tích kết quả phân loại lỗi
            result = self._parse_error_analysis(response_text)
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích lỗi: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Trả về giá trị mặc định
            return {
                "error_type": "Analysis Error",
                "explanation": f"Lỗi khi phân tích: {str(e)}"
            }
    
    def _parse_error_analysis(self, analysis_response: str) -> Dict[str, Any]:
        """
        Phân tích kết quả phân loại lỗi từ API.
        
        Args:
            analysis_response (str): Phản hồi từ API
            
        Returns:
            Dict: Dictionary chứa loại lỗi và giải thích
        """
        result = {
            "error_type": "Unknown",
            "explanation": analysis_response
        }
        
        if not analysis_response:
            return result
        
        # Xác định loại lỗi từ phản hồi
        error_type_patterns = [
            r"(?:Loại lỗi|Error Type)[:]\s*(.*?)(?:\n|$)",
            r"(?:lỗi|error)[:]\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in error_type_patterns:
            match = re.search(pattern, analysis_response, re.IGNORECASE)
            if match:
                error_type = match.group(1).strip()
                
                # Chuẩn hóa loại lỗi
                error_type = self._normalize_error_type(error_type)
                result["error_type"] = error_type
                break
        
        # Trích xuất phần giải thích
        explanation_patterns = [
            r"(?:Giải thích|Brief Explanation)[:]\s*([\s\S]*)",
            r"(?:giải thích|explanation)[:]\s*([\s\S]*)"
        ]
        
        for pattern in explanation_patterns:
            match = re.search(pattern, analysis_response, re.IGNORECASE)
            if match:
                explanation = match.group(1).strip()
                result["explanation"] = explanation
                break
        
        return result
    
    def _normalize_error_type(self, error_type: str) -> str:
        """
        Chuẩn hóa loại lỗi về các loại chuẩn.
        
        Args:
            error_type (str): Loại lỗi được trích xuất từ phản hồi
            
        Returns:
            str: Loại lỗi đã chuẩn hóa
        """
        # Các từ khóa để phân loại lỗi
        error_types = {
            "Knowledge Error": ["knowledge", "kiến thức", "thiếu thông tin", "incorrect knowledge"],
            "Reasoning Error": ["reasoning", "suy luận", "logic", "logical", "reasoning process"],
            "Calculation Error": ["calculation", "tính toán", "mathematical", "computational", "số học"],
            "Non-answer": ["non-answer", "không trả lời", "refusing", "failing", "từ chối"],
            "Off-topic": ["off-topic", "lạc đề", "unrelated", "không liên quan"],
            "Misunderstanding": ["misunderstanding", "hiểu nhầm", "misinterpreting", "misinterpretation"],
            "Other": ["other", "khác"]
        }
        
        error_type_lower = error_type.lower()
        
        # Tìm loại lỗi phù hợp nhất
        for standard_type, keywords in error_types.items():
            if any(keyword.lower() in error_type_lower for keyword in keywords):
                return standard_type
        
        # Kiểm tra chỉ số 1-7 nếu có
        if re.match(r"^[1-7][\.:]?", error_type_lower):
            index = int(error_type_lower[0]) - 1
            standard_types = list(error_types.keys())
            if 0 <= index < len(standard_types):
                return standard_types[index]
        
        return "Other"
    
    def _compute_error_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán các metrics liên quan đến phân tích lỗi.
        
        Args:
            df (pd.DataFrame): DataFrame đã có phân loại lỗi
            
        Returns:
            Dict: Metrics liên quan đến phân tích lỗi
        """
        metrics = {}
        
        # Lọc các dòng có phân loại lỗi
        error_df = df[df['error_type'] != '']
        
        if len(error_df) == 0:
            return metrics
        
        # 1. Tỷ lệ các loại lỗi tổng thể
        error_counts = error_df['error_type'].value_counts()
        error_percentages = error_counts / len(error_df) * 100
        
        metrics["overall"] = {
            "error_counts": error_counts.to_dict(),
            "error_percentages": error_percentages.to_dict()
        }
        
        # 2. Tỷ lệ lỗi theo model
        metrics["by_model"] = {}
        if 'model_name' in error_df.columns:
            for model in error_df['model_name'].unique():
                model_df = error_df[error_df['model_name'] == model]
                
                model_error_counts = model_df['error_type'].value_counts()
                model_error_percentages = model_error_counts / len(model_df) * 100
                
                metrics["by_model"][model] = {
                    "error_counts": model_error_counts.to_dict(),
                    "error_percentages": model_error_percentages.to_dict()
                }
        elif 'model' in error_df.columns:
            for model in error_df['model'].unique():
                model_df = error_df[error_df['model'] == model]
                
                model_error_counts = model_df['error_type'].value_counts()
                model_error_percentages = model_error_counts / len(model_df) * 100
                
                metrics["by_model"][model] = {
                    "error_counts": model_error_counts.to_dict(),
                    "error_percentages": model_error_percentages.to_dict()
                }
        
        # 3. Tỷ lệ lỗi theo prompt type
        metrics["by_prompt_type"] = {}
        for prompt in error_df['prompt_type'].unique():
            prompt_df = error_df[error_df['prompt_type'] == prompt]
            
            prompt_error_counts = prompt_df['error_type'].value_counts()
            prompt_error_percentages = prompt_error_counts / len(prompt_df) * 100
            
            metrics["by_prompt_type"][prompt] = {
                "error_counts": prompt_error_counts.to_dict(),
                "error_percentages": prompt_error_percentages.to_dict()
            }
        
        # 4. Tỷ lệ lỗi theo model và prompt type
        metrics["by_model_prompt"] = {}
        if 'model_name' in error_df.columns:
            for model in error_df['model_name'].unique():
                metrics["by_model_prompt"][model] = {}
                model_df = error_df[error_df['model_name'] == model]
                
                for prompt in model_df['prompt_type'].unique():
                    prompt_df = model_df[model_df['prompt_type'] == prompt]
                    
                    mp_error_counts = prompt_df['error_type'].value_counts()
                    mp_error_percentages = mp_error_counts / len(prompt_df) * 100
                    
                    metrics["by_model_prompt"][model][prompt] = {
                        "error_counts": mp_error_counts.to_dict(),
                        "error_percentages": mp_error_percentages.to_dict()
                    }
        elif 'model' in error_df.columns:
            for model in error_df['model'].unique():
                metrics["by_model_prompt"][model] = {}
                model_df = error_df[error_df['model'] == model]
                
                for prompt in model_df['prompt_type'].unique():
                    prompt_df = model_df[model_df['prompt_type'] == prompt]
                    
                    mp_error_counts = prompt_df['error_type'].value_counts()
                    mp_error_percentages = mp_error_counts / len(prompt_df) * 100
                    
                    metrics["by_model_prompt"][model][prompt] = {
                        "error_counts": mp_error_counts.to_dict(),
                        "error_percentages": mp_error_percentages.to_dict()
                    }
        
        return metrics

    def evaluate_reasoning_quality(self, 
                               results_df: pd.DataFrame, 
                               sample_size: int = 10,
                               random_seed: int = 42) -> pd.DataFrame:
        """
        Đánh giá chất lượng suy luận của các câu trả lời LLM.
        
        Args:
            results_df (pd.DataFrame): DataFrame kết quả để đánh giá
            sample_size (int): Số lượng mẫu để đánh giá
            random_seed (int): Seed ngẫu nhiên cho việc lấy mẫu
            
        Returns:
            pd.DataFrame: DataFrame đã bổ sung đánh giá suy luận
        """
        # Kiểm tra xem có các cột cần thiết không
        required_cols = ['question_text', 'response', 'correct_answer']
        for col in required_cols:
            if col not in results_df.columns:
                logger.warning(f"Không thể đánh giá suy luận: thiếu cột '{col}'")
                return results_df
                
        # Đảm bảo có các cột cần thiết cho đánh giá suy luận
        if 'reasoning_avg_score' not in results_df.columns:
            results_df['reasoning_avg_score'] = np.nan
            
        if 'reasoning_evaluation' not in results_df.columns:
            results_df['reasoning_evaluation'] = ''
        
        # Đảm bảo có cột đánh giá suy luận
        result_cols = [
            'reasoning_logical_flow', 
            'reasoning_mathematical_correctness', 
            'reasoning_clarity', 
            'reasoning_completeness', 
            'reasoning_relevance',
            'reasoning_avg_score',
            'reasoning_evaluation'
        ]
        
        for col in result_cols:
            if col not in results_df.columns:
                results_df[col] = np.nan
                
        # Lọc các mẫu có đáp án đúng và có sử dụng prompt yêu cầu lập luận
        has_reasoning = results_df['prompt_type'].str.contains('thought|cot|reasoning|react', case=False, na=False)
        
        # Nếu không chỉ định mẫu cụ thể, chúng ta chọn ngẫu nhiên
        valid_indices = results_df.index[has_reasoning].tolist()
        
        if not valid_indices:
            logger.warning("Không có câu trả lời nào phù hợp để đánh giá suy luận")
            return results_df
            
        # Lấy mẫu ngẫu nhiên từ các chỉ số hợp lệ
        random.seed(random_seed)
        
        # Giới hạn số lượng mẫu để đánh giá
        sample_size = min(sample_size, len(valid_indices))
        sample_indices = random.sample(valid_indices, sample_size)
        
        if self.verbose:
            logger.info(f"Đánh giá suy luận cho {sample_size} mẫu ngẫu nhiên")
        
        # Đánh giá từng mẫu
        for i, idx in enumerate(sample_indices):
            row = results_df.loc[idx]
            
            question = row['question_text']
            correct_answer = row['correct_answer']
            model_answer = row['response']
            
            if self.verbose:
                logger.info(f"Đánh giá mẫu {i+1}/{len(sample_indices)}: model={row['model_name']}, prompt={row['prompt_type']}")
                
            try:
                # Đánh giá suy luận
                eval_result = self._evaluate_single_reasoning(question, correct_answer, model_answer)
                
                # Cập nhật DataFrame
                for criterion, score in eval_result.items():
                    if criterion != 'explanation':
                        col_name = f'reasoning_{criterion}'
                        if col_name in results_df.columns:
                            results_df.at[idx, col_name] = score
                
                # Tính điểm trung bình
                criteria_scores = [v for k, v in eval_result.items() if k != 'explanation' and isinstance(v, (int, float))]
                avg_score = sum(criteria_scores) / len(criteria_scores) if criteria_scores else 0
                results_df.at[idx, 'reasoning_avg_score'] = avg_score
                
                # Lưu giải thích đánh giá
                if 'explanation' in eval_result:
                    results_df.at[idx, 'reasoning_evaluation'] = eval_result['explanation']
                    
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá suy luận cho mẫu {idx}: {str(e)}")
                logger.error(traceback.format_exc())
        
        return results_df
    
    def _evaluate_single_reasoning(self, question, correct_answer, model_answer):
        """
        Đánh giá chất lượng suy luận cho một cặp câu hỏi-câu trả lời.
        
        Sử dụng LLM (mặc định là Llama 3 qua Groq API) để đánh giá chất lượng suy luận
        dựa trên các tiêu chí như tính logic, tính toán chính xác, rõ ràng, đầy đủ và liên quan.
        
        Args:
            question (str): Câu hỏi/bài toán
            correct_answer (str): Câu trả lời đúng/đáp án
            model_answer (str): Câu trả lời của mô hình cần đánh giá
            
        Returns:
            Dict: Kết quả đánh giá với các tiêu chí và điểm số
        """
        # Import cần thiết chỉ trong hàm này để tránh import cycle
        try:
            from core.model_interface import generate_text
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from core.model_interface import generate_text
            
        # Tạo prompt đánh giá
        evaluation_prompt = self.reasoning_eval_template.format(
            question=question,
            correct_answer=correct_answer,
            model_answer=model_answer
        )
        
        # Lấy phản hồi đánh giá từ LLM
        if self.verbose:
            logger.info(f"Gửi yêu cầu đánh giá reasoning đến model: {self.reasoning_model}")
            
        try:
            eval_response = generate_text(
                model_name=self.reasoning_model,
                prompt=evaluation_prompt,
                generation_config={
                    "temperature": 0.1,  # Giảm temperature để có kết quả ổn định
                    "max_tokens": 1000    # Đủ dài cho đánh giá chi tiết
                }
            )
            
            # Nếu response là tuple (text, stats), lấy text
            if isinstance(eval_response, tuple) and len(eval_response) > 0:
                eval_response = eval_response[0]
                
            # Parse kết quả đánh giá
            return self._parse_reasoning_evaluation(eval_response)
            
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá suy luận với LLM: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Trả về kết quả mặc định nếu có lỗi
            return {
                'logical_flow': 0,
                'mathematical_correctness': 0,
                'clarity': 0,
                'completeness': 0,
                'relevance': 0,
                'avg_score': 0,
                'explanation': f"Lỗi khi đánh giá: {str(e)}"
            }
    
    def _parse_reasoning_evaluation(self, eval_response):
        """
        Phân tích kết quả đánh giá từ LLM để trích xuất điểm số và giải thích.
        
        Args:
            eval_response (str): Phản hồi từ mô hình đánh giá
            
        Returns:
            Dict: Kết quả đánh giá với các tiêu chí và điểm số
        """
        # Kiểm tra xem eval_response có phải là chuỗi JSON hợp lệ không
        import json
        
        # Khởi tạo kết quả mặc định
        result = {
            'logical_flow': 0,
            'mathematical_correctness': 0,
            'clarity': 0,
            'completeness': 0,
            'relevance': 0,
            'avg_score': 0,
            'explanation': ''
        }
        
        # Xử lý khi eval_response là dict (đã được parse trước đó)
        if isinstance(eval_response, dict):
            # Cập nhật kết quả từ dict
            for key in result.keys():
                if key in eval_response:
                    result[key] = eval_response[key]
            return result
        
        # Thử phân tích dưới dạng JSON
        if eval_response and isinstance(eval_response, str):
            try:
                # Xử lý trường hợp nhiều JSON objects bị nối với nhau
                if eval_response.count('{') > 1 and eval_response.count('}') > 1:
                    # Tìm JSON object đầu tiên
                    first_open = eval_response.find('{')
                    first_close = eval_response.find('}', first_open) + 1
                    
                    if first_open >= 0 and first_close > first_open:
                        clean_response = eval_response[first_open:first_close]
                        logger.debug(f"Phát hiện nhiều JSON objects, chỉ sử dụng object đầu tiên: {clean_response}")
                        try:
                            json_data = json.loads(clean_response)
                            logger.debug(f"Đã phân tích chuỗi JSON đầu tiên thành công: {json_data}")
                            
                            # Cập nhật kết quả
                            for key in result.keys():
                                if key in json_data:
                                    result[key] = json_data[key]
                            
                            return result
                        except json.JSONDecodeError:
                            logger.debug(f"Không thể phân tích JSON object đầu tiên, tiếp tục tìm kiếm")
                
                # Thử phân tích toàn bộ chuỗi như JSON
                try:
                    json_data = json.loads(eval_response)
                    logger.debug(f"Đã phân tích chuỗi JSON thành công: {json_data}")
                    
                    # Cập nhật kết quả từ dữ liệu JSON
                    for key in result.keys():
                        if key in json_data:
                            result[key] = json_data[key]
                    
                    # Hoàn tất và trả về kết quả
                    return result
                    
                except json.JSONDecodeError:
                    # Cố gắng làm sạch chuỗi và thử lại
                    # Tìm JSON object hợp lệ trong chuỗi
                    import re
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    match = re.search(json_pattern, eval_response)
                    
                    if match:
                        potential_json = match.group(0)
                        try:
                            json_data = json.loads(potential_json)
                            logger.debug(f"Đã phân tích chuỗi JSON được trích xuất thành công: {json_data}")
                            
                            # Cập nhật kết quả từ dữ liệu JSON
                            for key in result.keys():
                                if key in json_data:
                                    result[key] = json_data[key]
                            
                            return result
                        except:
                            logger.debug("Không thể phân tích JSON sau khi trích xuất, tiếp tục với phương pháp regex")
                    else:
                        logger.debug("Không tìm thấy chuỗi JSON hợp lệ, tiếp tục với phương pháp regex")
                    
            except Exception as e:
                logger.debug(f"Lỗi khi xử lý JSON: {str(e)}")
        
        # Tiếp tục với phương pháp phân tích regex nếu không phải JSON
        # Các tiêu chí cần trích xuất
        criteria = {
            'logical_flow': r'(?:Tính hợp lý|Độ hợp lý|Logical flow|Tính logic).*?(\d+)[/\s]*5',
            'mathematical_correctness': r'(?:Độ chính xác về mặt toán học|Mathematical correctness|Tính toán chính xác).*?(\d+)[/\s]*5',
            'clarity': r'(?:Rõ ràng|Độ rõ ràng|Clarity).*?(\d+)[/\s]*5',
            'completeness': r'(?:Tính đầy đủ|Đầy đủ|Completeness).*?(\d+)[/\s]*5',
            'relevance': r'(?:Mức độ liên quan|Tính liên quan|Relevance).*?(\d+)[/\s]*5'
        }
        
        # Mẫu để trích xuất điểm trung bình
        avg_pattern = r'(?:Điểm trung bình|Average score|Avg score).*?(\d+\.?\d*)[/\s]*5'
        
        # Mẫu để trích xuất phần giải thích
        explanation_pattern = r'(?:Giải thích|Explanation)\s*:(.*?)(?:$|(?=\n\s*\d))'
        
        # Trích xuất điểm số cho từng tiêu chí
        import re
        for criterion, pattern in criteria.items():
            match = re.search(pattern, eval_response, re.IGNORECASE | re.DOTALL)
            if match:
                result[criterion] = int(match.group(1))
        
        # Trích xuất điểm trung bình
        avg_match = re.search(avg_pattern, eval_response, re.IGNORECASE | re.DOTALL)
        if avg_match:
            try:
                result['avg_score'] = float(avg_match.group(1))
            except ValueError:
                # Tính toán lại điểm trung bình nếu không thể trích xuất
                scores = [result[c] for c in criteria.keys()]
                result['avg_score'] = sum(scores) / len(scores) if scores else 0
        else:
            # Tính toán điểm trung bình
            scores = [result[c] for c in criteria.keys()]
            result['avg_score'] = sum(scores) / len(scores) if scores else 0
        
        # Trích xuất phần giải thích
        explanation_match = re.search(explanation_pattern, eval_response, re.IGNORECASE | re.DOTALL)
        if explanation_match:
            result['explanation'] = explanation_match.group(1).strip()
        else:
            # Nếu không tìm thấy phần giải thích theo mẫu,
            # lấy phần cuối của eval_response làm giải thích
            lines = eval_response.strip().split('\n')
            for i, line in enumerate(lines):
                if 'giải thích' in line.lower() or 'explanation' in line.lower():
                    result['explanation'] = '\n'.join(lines[i+1:]).strip()
                    break
        
        return result
    
    def _compute_reasoning_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán các metrics đánh giá suy luận.
        
        Args:
            df (pd.DataFrame): DataFrame đã có kết quả đánh giá suy luận
            
        Returns:
            Dict: Các metrics của đánh giá suy luận
        """
        metrics = {}
        
        # Kiểm tra các cột reasoning có tồn tại không
        reasoning_cols = [col for col in df.columns if col.startswith('reasoning_') 
                        and col not in ['reasoning_evaluation', 'reasoning_scores', 'reasoning_scores_str']]
        
        if not reasoning_cols:
            logger.warning("Không tìm thấy các cột reasoning_ để tính toán metrics")
            return metrics
        
        logger.debug(f"Tính toán metrics từ các cột reasoning: {reasoning_cols}")
        
        # Đảm bảo các cột chứa dữ liệu số
        for col in reasoning_cols:
            try:
                # Kiểm tra xem cột có chứa dữ liệu không phải số không
                if df[col].dtype == 'object':
                    logger.debug(f"Chuyển đổi cột {col} thành số")
                    # Thử chuyển đổi cột thành số
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.error(f"Lỗi khi chuyển đổi cột {col} thành số: {e}")
                # Loại bỏ cột này khỏi danh sách cần tính toán
                reasoning_cols.remove(col)
        
        if not reasoning_cols:
            logger.warning("Không còn cột reasoning_ nào để tính toán sau khi chuyển đổi")
            return metrics
        
        # 1. Metrics tổng thể
        metrics["overall"] = {}
        for col in reasoning_cols:
            criterion = col.replace('reasoning_', '')
            # Sử dụng mean trên dữ liệu số, bỏ qua giá trị NaN
            metrics["overall"][criterion] = df[col].mean(skipna=True)
        
        # 2. Metrics theo model
        metrics["by_model"] = {}
        for model in df['model_name'].unique():
            metrics["by_model"][model] = {}
            model_df = df[df['model_name'] == model]
            
            for col in reasoning_cols:
                criterion = col.replace('reasoning_', '')
                metrics["by_model"][model][criterion] = model_df[col].mean(skipna=True)
        
        # 3. Metrics theo prompt type
        metrics["by_prompt_type"] = {}
        for prompt in df['prompt_type'].unique():
            metrics["by_prompt_type"][prompt] = {}
            prompt_df = df[df['prompt_type'] == prompt]
            
            for col in reasoning_cols:
                criterion = col.replace('reasoning_', '')
                metrics["by_prompt_type"][prompt][criterion] = prompt_df[col].mean(skipna=True)
        
        # 4. Metrics theo model và prompt type
        metrics["by_model_prompt"] = {}
        for model in df['model_name'].unique():
            metrics["by_model_prompt"][model] = {}
            model_df = df[df['model_name'] == model]
            
            for prompt in model_df['prompt_type'].unique():
                metrics["by_model_prompt"][model][prompt] = {}
                prompt_df = model_df[model_df['prompt_type'] == prompt]
                
                for col in reasoning_cols:
                    criterion = col.replace('reasoning_', '')
                    metrics["by_model_prompt"][model][prompt][criterion] = prompt_df[col].mean(skipna=True)
        
        return metrics
    
    def calculate_similarity(self, 
                            df: pd.DataFrame,
                            reference_column: str = 'correct_answer',
                            response_column: str = 'response') -> pd.DataFrame:
        """
        Tính toán semantic similarity giữa câu trả lời mô hình và đáp án chuẩn.
        
        Args:
            df (pd.DataFrame): DataFrame kết quả
            reference_column (str): Tên cột chứa tham chiếu (thường là đáp án chuẩn)
            response_column (str): Tên cột chứa câu trả lời mô hình
            
        Returns:
            pd.DataFrame: DataFrame với cột similarity bổ sung
        """
        # Chỉ tính similarity khi có mô hình được chỉ định
        if not self.similarity_model:
            logger.warning("Không có mô hình similarity được chỉ định")
            return df
        
        if self.verbose:
            logger.info(f"📏 Tính toán semantic similarity cho {len(df)} mục")
        
        # Copy DataFrame để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Chuẩn bị cột similarity
        result_df['similarity_score'] = 0.0
        
        try:
            # Tính similarity cho từng dòng
            for idx, row in result_df.iterrows():
                reference = row[reference_column]
                response = row[response_column]
                
                # Tính similarity score
                similarity = self._calculate_text_similarity(reference, response)
                result_df.at[idx, 'similarity_score'] = similarity
                
        except Exception as e:
            logger.error(f"Lỗi khi tính toán similarity: {str(e)}")
        
        return result_df
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Tính toán similarity giữa hai đoạn văn bản.
        Phương thức này sẽ được triển khai khi có mô hình similarity cụ thể.
        
        Args:
            text1 (str): Văn bản thứ nhất
            text2 (str): Văn bản thứ hai
            
        Returns:
            float: Điểm similarity (0-1)
        """
        # Chỉ dùng khi có mô hình
        if not self.similarity_model:
            return 0.0
        
        # TODO: Triển khai tính similarity khi cần thiết
        # Hiện tại, trả về giá trị mặc định
        return 0.0
    
    def export_summary(self, analysis_results: Dict[str, Any], format: str = 'text') -> str:
        """
        Xuất bản tóm tắt kết quả phân tích theo định dạng chỉ định.
        
        Args:
            analysis_results (Dict): Kết quả phân tích từ hàm analyze_results
            format (str): Định dạng xuất ('text', 'markdown', 'json', 'html')
            
        Returns:
            str: Tóm tắt kết quả phân tích theo định dạng chỉ định
        """
        if format == 'text':
            return self._export_text_summary(analysis_results)
        elif format == 'markdown':
            return self._export_markdown_summary(analysis_results)
        elif format == 'json':
            import json
            return json.dumps(analysis_results, indent=2, ensure_ascii=False)
        elif format == 'html':
            return self._export_html_summary(analysis_results)
        else:
            return self._export_text_summary(analysis_results)
    
    def _export_text_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Xuất bản tóm tắt dạng text.
        
        Args:
            analysis_results (Dict): Kết quả phân tích
            
        Returns:
            str: Tóm tắt dạng text
        """
        summary = []
        
        # Thông tin tổng quan
        summary.append("=== KẾT QUẢ PHÂN TÍCH ===")
        
        # Metrics cơ bản
        basic = analysis_results.get('basic_metrics', {})
        summary.append("\n--- METRICS CƠ BẢN ---")
        
        if 'overall_accuracy' in basic:
            summary.append(f"Accuracy tổng thể: {basic['overall_accuracy']:.4f}")
        
        if 'average_latency' in basic:
            summary.append(f"Thời gian trung bình: {basic['average_latency']:.2f}s")
        
        if 'average_response_length' in basic:
            summary.append(f"Độ dài phản hồi trung bình: {basic['average_response_length']:.2f} tokens")
        
        # Metrics theo model và prompt type
        model_metrics = analysis_results.get('model_prompt_metrics', {})
        if model_metrics:
            summary.append("\n--- METRICS THEO MODEL & PROMPT TYPE ---")
            
            for model, prompts in model_metrics.items():
                summary.append(f"\nModel: {model}")
                
                for prompt, metrics in prompts.items():
                    summary.append(f"  Prompt: {prompt}")
                    
                    if 'accuracy' in metrics:
                        summary.append(f"    - Accuracy: {metrics['accuracy']:.4f}")
                    
                    if 'avg_latency' in metrics:
                        summary.append(f"    - Thời gian TB: {metrics['avg_latency']:.2f}s")
                    
                    if 'avg_response_length' in metrics:
                        summary.append(f"    - Độ dài TB: {metrics['avg_response_length']:.2f} tokens")
        
        # Metrics theo loại câu hỏi
        q_metrics = analysis_results.get('question_type_metrics', {})
        if q_metrics:
            summary.append("\n--- METRICS THEO LOẠI CÂU HỎI ---")
            
            for q_type, metrics in q_metrics.items():
                summary.append(f"\nLoại: {q_type} (số lượng: {metrics.get('count', 0)})")
                
                if 'accuracy' in metrics:
                    summary.append(f"  - Accuracy: {metrics['accuracy']:.4f}")
                
                if 'avg_latency' in metrics:
                    summary.append(f"  - Thời gian TB: {metrics['avg_latency']:.2f}s")
        
        return "\n".join(summary)
    
    def _export_markdown_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Xuất bản tóm tắt dạng markdown.
        
        Args:
            analysis_results (Dict): Kết quả phân tích
            
        Returns:
            str: Tóm tắt dạng markdown
        """
        summary = []
        
        # Thông tin tổng quan
        summary.append("# KẾT QUẢ PHÂN TÍCH")
        
        # Metrics cơ bản
        basic = analysis_results.get('basic_metrics', {})
        summary.append("\n## Metrics Cơ Bản")
        
        if basic:
            summary.append("| Metric | Giá trị |")
            summary.append("|--------|--------|")
            
            if 'overall_accuracy' in basic:
                summary.append(f"| Accuracy tổng thể | {basic['overall_accuracy']:.4f} |")
            
            if 'average_latency' in basic:
                summary.append(f"| Thời gian trung bình | {basic['average_latency']:.2f}s |")
            
            if 'average_response_length' in basic:
                summary.append(f"| Độ dài phản hồi trung bình | {basic['average_response_length']:.2f} tokens |")
        
        # Metrics theo model và prompt type
        model_metrics = analysis_results.get('model_prompt_metrics', {})
        if model_metrics:
            summary.append("\n## Metrics Theo Model & Prompt Type")
            
            for model, prompts in model_metrics.items():
                summary.append(f"\n### Model: {model}")
                
                summary.append("| Prompt Type | Accuracy | Thời gian TB | Độ dài TB |")
                summary.append("|------------|----------|--------------|-----------|")
                
                for prompt, metrics in prompts.items():
                    acc = f"{metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in metrics else 'N/A'
                    lat = f"{metrics.get('avg_latency', 'N/A'):.2f}s" if 'avg_latency' in metrics else 'N/A'
                    len_val = f"{metrics.get('avg_response_length', 'N/A'):.2f}" if 'avg_response_length' in metrics else 'N/A'
                    
                    summary.append(f"| {prompt} | {acc} | {lat} | {len_val} |")
        
        # Metrics theo loại câu hỏi
        q_metrics = analysis_results.get('question_type_metrics', {})
        if q_metrics:
            summary.append("\n## Metrics Theo Loại Câu Hỏi")
            
            summary.append("| Loại | Số lượng | Accuracy | Thời gian TB |")
            summary.append("|------|----------|----------|--------------|")
            
            for q_type, metrics in q_metrics.items():
                count = metrics.get('count', 'N/A')
                acc = f"{metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in metrics else 'N/A'
                lat = f"{metrics.get('avg_latency', 'N/A'):.2f}s" if 'avg_latency' in metrics else 'N/A'
                
                summary.append(f"| {q_type} | {count} | {acc} | {lat} |")
        
        return "\n".join(summary)
    
    def _export_html_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Xuất bản tóm tắt dạng HTML.
        
        Args:
            analysis_results (Dict): Kết quả phân tích
            
        Returns:
            str: Tóm tắt dạng HTML
        """
        # Chuyển đổi từ markdown sang HTML
        try:
            import markdown
            md_summary = self._export_markdown_summary(analysis_results)
            html = markdown.markdown(md_summary, extensions=['tables'])
            
            # Bọc trong template HTML cơ bản
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Kết Quả Phân Tích LLM</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    h1, h2, h3 {{ color: #333; }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """
        except ImportError:
            # Fallback nếu không có thư viện markdown
            return f"<pre>{self._export_text_summary(analysis_results)}</pre>"

    def evaluate_consistency(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Đánh giá tính nhất quán trong các self-consistency runs.
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            
        Returns:
            pd.DataFrame: DataFrame đã bổ sung đánh giá tính nhất quán
        """
        # Kiểm tra xem có cột response không
        if 'response' not in results_df.columns:
            logger.error("Lỗi khi đánh giá tính nhất quán: thiếu cột 'response'")
            return results_df
            
        # Đảm bảo có các cột cần thiết
        if 'consistency_score' not in results_df.columns:
            results_df['consistency_score'] = np.nan
        
        if 'consistency_agreement_rate' not in results_df.columns:
            results_df['consistency_agreement_rate'] = np.nan
        
        if 'consistency_most_common' not in results_df.columns:
            results_df['consistency_most_common'] = ''
        
        # Lọc các prompt có chứa self-consistency
        self_consistency_mask = results_df['prompt_type'].str.contains('consistency|cot_self_consistency', case=False)
        
        if not self_consistency_mask.any():
            logger.warning("Không tìm thấy self-consistency runs để đánh giá tính nhất quán")
            return results_df
        
        # Nhóm theo model, question và prompt type (loại bỏ phần số runs nếu có)
        # Ví dụ: cot_self_consistency_3 và cot_self_consistency_5 sẽ được nhóm chung
        results_df['base_prompt_type'] = results_df['prompt_type'].str.replace(r'_\d+$', '', regex=True)
        
        # Sử dụng model_name thay vì model nếu có
        model_col = 'model_name' if 'model_name' in results_df.columns else 'model'
        
        # Xác định các nhóm chạy self-consistency
        groups = results_df[self_consistency_mask].groupby([model_col, 'question_id', 'base_prompt_type'])
        
        # Xử lý từng nhóm
        for (model, question_id, prompt_type), group in groups:
            # Bỏ qua nếu chỉ có một kết quả
            if len(group) <= 1:
                continue
            
            # Lấy tất cả các câu trả lời trong nhóm
            responses = group['response'].tolist()
            final_answers = group['final_answer'].tolist() if 'final_answer' in group.columns else responses
            
            # Tính toán tỷ lệ nhất quán
            from collections import Counter
            answer_counts = Counter(final_answers)
            
            # Xác định câu trả lời phổ biến nhất
            most_common_answer, most_common_count = answer_counts.most_common(1)[0]
            agreement_rate = most_common_count / len(final_answers)
            
            # Tính điểm nhất quán: 1 nếu hoàn toàn nhất quán, giảm dần khi có nhiều câu trả lời khác nhau
            unique_answers = len(answer_counts)
            consistency_score = 1.0 if unique_answers == 1 else (most_common_count / len(final_answers))
            
            # Cập nhật điểm nhất quán cho từng dòng trong nhóm
            for idx in group.index:
                results_df.at[idx, 'consistency_score'] = consistency_score
                results_df.at[idx, 'consistency_agreement_rate'] = agreement_rate
                results_df.at[idx, 'consistency_most_common'] = most_common_answer
        
        # Xóa cột tạm base_prompt_type
        results_df = results_df.drop('base_prompt_type', axis=1)
        
        return results_df
    
    def _compute_consistency_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán metrics cho đánh giá tính nhất quán.
        
        Args:
            df (pd.DataFrame): DataFrame đã có đánh giá tính nhất quán
            
        Returns:
            Dict: Metrics liên quan đến tính nhất quán
        """
        metrics = {}
        
        # Lọc các dòng có đánh giá tính nhất quán
        consistency_df = df[~df['consistency_score'].isna()]
        
        if len(consistency_df) == 0:
            return metrics
        
        # 1. Metrics tổng thể
        metrics["overall"] = {
            "avg_consistency_score": consistency_df['consistency_score'].mean(),
            "avg_agreement_rate": consistency_df['consistency_agreement_rate'].mean()
        }
        
        # 2. Metrics theo model
        metrics["by_model"] = {}
        
        # Sử dụng model_name thay vì model nếu có
        model_col = 'model_name' if 'model_name' in consistency_df.columns else 'model'
        
        if model_col in consistency_df.columns:
            for model in consistency_df[model_col].unique():
                model_df = consistency_df[consistency_df[model_col] == model]
                metrics["by_model"][model] = {
                    "avg_consistency_score": model_df['consistency_score'].mean(),
                    "avg_agreement_rate": model_df['consistency_agreement_rate'].mean()
                }
        
        # 3. Metrics theo prompt type
        metrics["by_prompt_type"] = {}
        if 'prompt_type' in consistency_df.columns:
            for prompt in consistency_df['prompt_type'].unique():
                prompt_df = consistency_df[consistency_df['prompt_type'] == prompt]
                metrics["by_prompt_type"][prompt] = {
                    "avg_consistency_score": prompt_df['consistency_score'].mean(),
                    "avg_agreement_rate": prompt_df['consistency_agreement_rate'].mean()
                }
        
        # 4. Metrics theo model và prompt type
        metrics["by_model_prompt"] = {}
        if model_col in consistency_df.columns and 'prompt_type' in consistency_df.columns:
            for model in consistency_df[model_col].unique():
                metrics["by_model_prompt"][model] = {}
                model_df = consistency_df[consistency_df[model_col] == model]
                
                for prompt in model_df['prompt_type'].unique():
                    prompt_df = model_df[model_df['prompt_type'] == prompt]
                    metrics["by_model_prompt"][model][prompt] = {
                        "avg_consistency_score": prompt_df['consistency_score'].mean(),
                        "avg_agreement_rate": prompt_df['consistency_agreement_rate'].mean()
                    }
        
        return metrics

    def evaluate_completeness(self, 
                          results_df: pd.DataFrame, 
                          sample_size: int = 50,
                          random_seed: int = 42) -> pd.DataFrame:
        """
        Đánh giá tính đầy đủ của các câu trả lời.
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            sample_size (int): Số lượng mẫu cần đánh giá
            random_seed (int): Seed ngẫu nhiên cho việc lấy mẫu
            
        Returns:
            pd.DataFrame: DataFrame đã bổ sung đánh giá tính đầy đủ
        """
        # Kiểm tra xem có các cột cần thiết không
        required_cols = ['question_text', 'response']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if missing_cols:
            logger.warning(f"Không thể đánh giá tính đầy đủ: thiếu các cột {missing_cols}")
            return results_df
                
        # Đảm bảo có các cột cần thiết
        if 'completeness_score' not in results_df.columns:
            results_df['completeness_score'] = np.nan
            
        if 'completeness_evaluation' not in results_df.columns:
            results_df['completeness_evaluation'] = ''
            
        # Lọc các hàng có câu hỏi và câu trả lời
        valid_rows = (
            ~results_df['question_text'].isna() & 
            ~results_df['response'].isna() &
            (results_df['response'] != '') &  # Thêm điều kiện check chuỗi rỗng
            results_df['completeness_score'].isna()  # Chưa được đánh giá
        )
        
        valid_indices = results_df[valid_rows].index.tolist()
        
        if not valid_indices:
            logger.warning("Không có mẫu phù hợp để đánh giá tính đầy đủ")
            return results_df
            
        # Lấy mẫu ngẫu nhiên nếu cần
        np.random.seed(random_seed)
        if len(valid_indices) > sample_size:
            sample_indices = np.random.choice(valid_indices, size=sample_size, replace=False)
        else:
            sample_indices = valid_indices
            
        logger.info(f"Đánh giá tính đầy đủ cho {len(sample_indices)} mẫu")
        
        # Đánh giá từng mẫu
        for i, idx in enumerate(sample_indices):
            row = results_df.loc[idx]
            
            question = row['question_text']
            model_answer = row['response']
            
            if self.verbose:
                model_name = row['model_name'] if 'model_name' in row else row.get('model', 'unknown')
                prompt_type = row['prompt_type'] if 'prompt_type' in row else 'unknown'
                logger.info(f"Đánh giá tính đầy đủ mẫu {i+1}/{len(sample_indices)}: model={model_name}, prompt={prompt_type}")
                
            try:
                # Đánh giá tính đầy đủ
                eval_result = self._evaluate_single_completeness(question, model_answer)
                
                # Cập nhật DataFrame
                results_df.at[idx, 'completeness_score'] = eval_result.get('score', 0.0)
                results_df.at[idx, 'completeness_evaluation'] = eval_result.get('explanation', '')
                    
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá tính đầy đủ cho mẫu {idx}: {str(e)}")
                logger.error(traceback.format_exc())
        
        return results_df
    
    def _evaluate_single_completeness(self, question: str, model_answer: str) -> Dict[str, Any]:
        """
        Đánh giá tính đầy đủ cho một mẫu câu hỏi/câu trả lời.
        
        Args:
            question (str): Câu hỏi
            model_answer (str): Câu trả lời của model
            
        Returns:
            Dict: Kết quả đánh giá (điểm và giải thích)
        """
        try:
            # Cắt bớt độ dài nếu quá dài
            max_length = 4000  # Giới hạn độ dài để tránh vượt quá context window
            if len(model_answer) > max_length:
                logger.warning(f"Cắt bớt câu trả lời ({len(model_answer)} -> {max_length} ký tự)")
                model_answer = model_answer[:max_length] + "..."
            
            # Tạo prompt đánh giá
            if self.language.lower() == "vietnamese":
                eval_prompt = """
Bạn là một chuyên gia đánh giá tính đầy đủ của câu trả lời. Hãy đánh giá xem câu trả lời có giải quyết tất cả các khía cạnh của câu hỏi hay không.

CÂU HỎI:
{question}

CÂU TRẢ LỜI:
{answer}

Hãy đánh giá tính đầy đủ của câu trả lời theo thang điểm từ 0-10 (10 là hoàn toàn đầy đủ). Xác định các khía cạnh của câu hỏi và kiểm tra xem câu trả lời đã đề cập đến tất cả các khía cạnh đó chưa.

Điểm tính đầy đủ: ?/10

Phân tích chi tiết:
1. Các khía cạnh của câu hỏi:
2. Các khía cạnh được trả lời:
3. Các khía cạnh chưa được trả lời (nếu có):
"""
            else:
                eval_prompt = """
You are an expert evaluating the completeness of answers. Assess whether the answer addresses all aspects of the question.

QUESTION:
{question}

ANSWER:
{answer}

Evaluate the completeness of the answer on a scale of 0-10 (10 being completely comprehensive). Identify the aspects of the question and check if the answer addresses all of them.

Completeness score: ?/10

Detailed analysis:
1. Aspects of the question:
2. Aspects addressed in the answer:
3. Aspects not addressed (if any):
"""
            
            # Format prompt
            eval_prompt = eval_prompt.format(question=question, answer=model_answer)
            
            # Sử dụng model API để đánh giá
            use_groq = self.reasoning_config.get("use_groq", True)
            if use_groq:
                # Sử dụng Groq API
                from core.model_interface import generate_text
                
                # Lấy tên model Groq
                model_name = "groq"
                config = {
                    "model": self.reasoning_config.get("models", {}).get(
                        "completeness_evaluation", "llama3-70b-8192"
                    ),
                    "temperature": 0.2,  # Thấp để đảm bảo tính nhất quán
                    "max_tokens": 1024
                }
                
                # Gọi API
                logger.debug("Đánh giá tính đầy đủ bằng Groq API")
                response_text, stats = generate_text(model_name, eval_prompt, config)
                
                if stats.get("has_error", False):
                    logger.error(f"Lỗi khi gọi Groq API: {stats.get('error_message')}")
                    # Fallback về giá trị mặc định
                    return {
                        "score": 5.0,
                        "explanation": f"[Lỗi đánh giá: {stats.get('error_message')}]"
                    }
            else:
                # TODO: Sử dụng model khác nếu cần
                logger.warning("Chỉ hỗ trợ Groq API để đánh giá tính đầy đủ")
                response_text = ""
                
            # Phân tích kết quả đánh giá
            eval_result = self._parse_completeness_evaluation(response_text)
            
            return eval_result
                
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá tính đầy đủ: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Trả về giá trị mặc định
            return {
                "score": 5.0,
                "explanation": f"[Lỗi đánh giá: {str(e)}]"
            }
    
    def _parse_completeness_evaluation(self, eval_response: str) -> Dict[str, Any]:
        """
        Phân tích kết quả đánh giá tính đầy đủ từ API.
        
        Args:
            eval_response (str): Phản hồi từ API
            
        Returns:
            Dict: Dictionary chứa điểm số và giải thích
        """
        result = {
            "score": 5.0,  # Giá trị mặc định
            "explanation": eval_response
        }
        
        if not eval_response:
            return result
            
        # Tìm điểm đánh giá từ phản hồi
        score_patterns = [
            r"(?:điểm|score)[^\d]*(\d+(?:\.\d+)?)/10",
            r"(?:điểm|score)[^\d]*:?[^\d]*(\d+(?:\.\d+)?)"
        ]
        
        # Tìm điểm số
        for pattern in score_patterns:
            match = re.search(pattern, eval_response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Chuyển về thang điểm 0-1
                    normalized_score = score / 10.0
                    result["score"] = max(0.0, min(1.0, normalized_score))
                    break
                except (ValueError, IndexError):
                    continue
        
        return result
    
    def _compute_completeness_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán metrics cho đánh giá tính đầy đủ.
        
        Args:
            df (pd.DataFrame): DataFrame đã có đánh giá tính đầy đủ
            
        Returns:
            Dict: Metrics liên quan đến tính đầy đủ
        """
        metrics = {}
        
        # Lọc các dòng có đánh giá tính đầy đủ
        completeness_df = df[~df['completeness_score'].isna()]
        
        if len(completeness_df) == 0:
            return metrics
        
        # 1. Metrics tổng thể
        metrics["overall"] = {
            "avg_completeness_score": completeness_df['completeness_score'].mean(),
            "high_completeness_rate": (completeness_df['completeness_score'] >= 0.8).mean()
        }
        
        # 2. Metrics theo model
        metrics["by_model"] = {}
        
        # Xác định cột model (model_name hoặc model)
        model_col = 'model_name' if 'model_name' in completeness_df.columns else 'model'
        
        if model_col in completeness_df.columns:
            for model in completeness_df[model_col].unique():
                model_df = completeness_df[completeness_df[model_col] == model]
                metrics["by_model"][model] = {
                    "avg_completeness_score": model_df['completeness_score'].mean(),
                    "high_completeness_rate": (model_df['completeness_score'] >= 0.8).mean()
                }
        
        # 3. Metrics theo prompt type
        metrics["by_prompt_type"] = {}
        if 'prompt_type' in completeness_df.columns:
            for prompt in completeness_df['prompt_type'].unique():
                prompt_df = completeness_df[completeness_df['prompt_type'] == prompt]
                metrics["by_prompt_type"][prompt] = {
                    "avg_completeness_score": prompt_df['completeness_score'].mean(),
                    "high_completeness_rate": (prompt_df['completeness_score'] >= 0.8).mean()
                }
        
        # 4. Metrics theo model và prompt type
        metrics["by_model_prompt"] = {}
        if model_col in completeness_df.columns and 'prompt_type' in completeness_df.columns:
            for model in completeness_df[model_col].unique():
                metrics["by_model_prompt"][model] = {}
                model_df = completeness_df[completeness_df[model_col] == model]
                
                for prompt in model_df['prompt_type'].unique():
                    prompt_df = model_df[model_df['prompt_type'] == prompt]
                    metrics["by_model_prompt"][model][prompt] = {
                        "avg_completeness_score": prompt_df['completeness_score'].mean(),
                        "high_completeness_rate": (prompt_df['completeness_score'] >= 0.8).mean()
                    }
        
        return metrics

    def evaluate_similarity(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán các metrics đo lường độ tương đồng giữa câu trả lời của model và đáp án chuẩn.
        Sử dụng các phương pháp:
        - ROUGE (đo lường sự trùng lặp n-gram)
        - BLEU (đo lường độ chính xác n-gram)
        - Cosine similarity của embeddings (nếu có similarity_model)
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            
        Returns:
            pd.DataFrame: DataFrame đã bổ sung các metrics đo lường độ tương đồng
        """
        # Đảm bảo có cột đáp án chuẩn và câu trả lời
        if 'correct_answer' not in results_df.columns or 'response' not in results_df.columns:
            logger.warning("Thiếu cột 'correct_answer' hoặc 'response' để tính toán độ tương đồng")
            return results_df
        
        # Thêm các cột cần thiết nếu chưa có
        for col in ['rouge_score', 'bleu_score', 'embedding_similarity']:
            if col not in results_df.columns:
                results_df[col] = np.nan
        
        # Lọc các hàng có đáp án chuẩn và câu trả lời
        valid_rows = ~results_df['correct_answer'].isna() & ~results_df['response'].isna()
        
        if not valid_rows.any():
            logger.warning("Không có mẫu phù hợp để tính toán độ tương đồng")
            return results_df
        
        # Tính toán ROUGE và BLEU scores
        try:
            # Nếu chưa có thư viện ROUGE hoặc BLEU, cần thêm vào requirements.txt
            from rouge import Rouge
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Download NLTK data nếu cần
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            # Khởi tạo ROUGE
            rouge = Rouge()
            smoothing = SmoothingFunction().method1
            
            # Xử lý từng hàng
            for idx in results_df[valid_rows].index:
                reference = results_df.at[idx, 'correct_answer']
                hypothesis = results_df.at[idx, 'response']
                
                # Tính ROUGE score
                try:
                    rouge_scores = rouge.get_scores(hypothesis, reference)
                    # Lấy trung bình rouge-1, rouge-2, rouge-l f1-scores
                    rouge_f1 = (rouge_scores[0]['rouge-1']['f'] + 
                               rouge_scores[0]['rouge-2']['f'] + 
                               rouge_scores[0]['rouge-l']['f']) / 3
                    results_df.at[idx, 'rouge_score'] = rouge_f1
                except Exception as e:
                    logger.debug(f"Lỗi khi tính ROUGE score: {str(e)}")
                
                # Tính BLEU score
                try:
                    # Tokenize câu
                    reference_tokens = nltk.word_tokenize(reference.lower())
                    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
                    
                    # Tính BLEU (sử dụng smoothing để tránh lỗi khi không có n-gram khớp)
                    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, 
                                              smoothing_function=smoothing)
                    results_df.at[idx, 'bleu_score'] = bleu_score
                except Exception as e:
                    logger.debug(f"Lỗi khi tính BLEU score: {str(e)}")
                
        except ImportError as e:
            logger.warning(f"Không thể tính ROUGE/BLEU scores do thiếu thư viện: {str(e)}")
        
        # Tính toán embedding similarity nếu có similarity model
        if self.similarity_model:
            # Thực hiện tính toán embedding similarity
            try:
                for idx in results_df[valid_rows].index:
                    reference = results_df.at[idx, 'correct_answer']
                    hypothesis = results_df.at[idx, 'response']
                    
                    similarity = self._calculate_embedding_similarity(reference, hypothesis)
                    results_df.at[idx, 'embedding_similarity'] = similarity
            except Exception as e:
                logger.warning(f"Lỗi khi tính embedding similarity: {str(e)}")
        
        return results_df
    
    def _calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Tính toán cosine similarity giữa embeddings của hai đoạn văn bản.
        
        Args:
            text1 (str): Văn bản thứ nhất
            text2 (str): Văn bản thứ hai
            
        Returns:
            float: Cosine similarity (0-1)
        """
        if not self.similarity_model:
            return 0.0
            
        try:
            # Đây là phần triển khai tùy thuộc vào model và framework sử dụng
            # Ví dụ với sentence-transformers
            from sentence_transformers import SentenceTransformer
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Kiểm tra xem similarity_model có phải là đường dẫn hoặc tên model không
            if isinstance(self.similarity_model, str):
                # Lazy loading model
                if not hasattr(self, '_embedding_model'):
                    self._embedding_model = SentenceTransformer(self.similarity_model)
                
                # Tính embeddings
                embedding1 = self._embedding_model.encode([text1])[0]
                embedding2 = self._embedding_model.encode([text2])[0]
                
                # Tính cosine similarity
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                return float(similarity)
            else:
                # Nếu similarity_model đã là instance của model
                embedding1 = self.similarity_model.encode([text1])[0]
                embedding2 = self.similarity_model.encode([text2])[0]
                
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                return float(similarity)
                
        except Exception as e:
            logger.error(f"Lỗi khi tính embedding similarity: {str(e)}")
            return 0.0
    
    def _compute_similarity_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán các metrics liên quan đến độ tương đồng với đáp án chuẩn.
        
        Args:
            df (pd.DataFrame): DataFrame đã có các metrics đo lường độ tương đồng
            
        Returns:
            Dict: Metrics liên quan đến độ tương đồng
        """
        metrics = {}
        
        # Danh sách các cột similarity metrics
        similarity_cols = ['rouge_score', 'bleu_score', 'embedding_similarity']
        
        # Lọc các hàng có ít nhất một metric similarity được tính toán
        similarity_df = df[df[similarity_cols].notna().any(axis=1)]
        
        if len(similarity_df) == 0:
            return metrics
        
        # 1. Metrics tổng thể
        metrics["overall"] = {}
        for col in similarity_cols:
            if col in similarity_df.columns and similarity_df[col].notna().any():
                metrics["overall"][col] = similarity_df[col].mean()
        
        # 2. Metrics theo model
        metrics["by_model"] = {}
        for model in similarity_df['model'].unique():
            metrics["by_model"][model] = {}
            model_df = similarity_df[similarity_df['model'] == model]
            
            for col in similarity_cols:
                if col in model_df.columns and model_df[col].notna().any():
                    metrics["by_model"][model][col] = model_df[col].mean()
        
        # 3. Metrics theo prompt type
        metrics["by_prompt_type"] = {}
        for prompt in similarity_df['prompt_type'].unique():
            metrics["by_prompt_type"][prompt] = {}
            prompt_df = similarity_df[similarity_df['prompt_type'] == prompt]
            
            for col in similarity_cols:
                if col in prompt_df.columns and prompt_df[col].notna().any():
                    metrics["by_prompt_type"][prompt][col] = prompt_df[col].mean()
        
        # 4. Metrics theo model và prompt type
        metrics["by_model_prompt"] = {}
        for model in similarity_df['model'].unique():
            metrics["by_model_prompt"][model] = {}
            model_df = similarity_df[similarity_df['model'] == model]
            
            for prompt in model_df['prompt_type'].unique():
                metrics["by_model_prompt"][model][prompt] = {}
                prompt_df = model_df[model_df['prompt_type'] == prompt]
                
                for col in similarity_cols:
                    if col in prompt_df.columns and prompt_df[col].notna().any():
                        metrics["by_model_prompt"][model][prompt][col] = prompt_df[col].mean()
        
        # 5. Tương quan giữa accuracy và các metrics similarity
        if 'is_correct' in similarity_df.columns:
            metrics["correlation"] = {}
            
            for col in similarity_cols:
                if col in similarity_df.columns and similarity_df[col].notna().any():
                    corr = similarity_df[['is_correct', col]].corr().iloc[0, 1]
                    metrics["correlation"][f"{col}_vs_accuracy"] = corr
        
        return metrics

    def _compute_accuracy_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán các metrics liên quan đến độ chính xác (Accuracy).
        
        Args:
            df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            
        Returns:
            Dict[str, Any]: Các metrics liên quan đến accuracy
        """
        metrics = {}
        
        if 'is_correct' not in df.columns:
            logger.warning("Không thể tính accuracy metrics: thiếu cột is_correct")
            return metrics
        
        # Tính overall accuracy
        metrics['overall_accuracy'] = df['is_correct'].mean()
        
        # Xác định cột model (có thể là 'model_name' hoặc 'model')
        model_col = 'model_name' if 'model_name' in df.columns else 'model'
        
        # Tính accuracy theo model và prompt type
        if model_col in df.columns:
            accuracy_by_model = df.groupby(model_col)['is_correct'].mean().to_dict()
            accuracy_by_model_prompt = df.groupby([model_col, 'prompt_type'])['is_correct'].mean().unstack().to_dict('index')
            
            metrics['accuracy_by_model'] = accuracy_by_model
            metrics['accuracy_by_model_prompt'] = accuracy_by_model_prompt
        
        if 'prompt_type' in df.columns:
            accuracy_by_prompt = df.groupby('prompt_type')['is_correct'].mean().to_dict()
            metrics['accuracy_by_prompt'] = accuracy_by_prompt
        
        # Tính F1 score nếu có thể
        try:
            from sklearn.metrics import f1_score
            if 'is_correct' in df.columns and 'expected_answer' in df.columns and 'response' in df.columns:
                # Thực hiện tính toán F1 score cho từng model/prompt
                f1_scores = {}
                for (model, prompt), group in df.groupby(['model_name', 'prompt_type']):
                    if len(group) > 0:
                        f1 = self._calculate_f1_score(group)
                        f1_scores[(model, prompt)] = f1
                
                metrics['f1_scores'] = f1_scores
        except (ImportError, Exception) as e:
            logger.warning(f"Không thể tính F1 score: {str(e)}")
        
        return metrics
    
    def _calculate_f1_score(self, group_df: pd.DataFrame) -> float:
        """
        Tính F1 score cho một nhóm kết quả.
        
        Args:
            group_df (pd.DataFrame): DataFrame chứa một nhóm kết quả
            
        Returns:
            float: F1 score
        """
        # Đơn giản hóa: coi is_correct như true positive/negative
        try:
            from sklearn.metrics import f1_score
            return f1_score([1] * len(group_df), group_df['is_correct'])
        except Exception:
            # Fallback: tính thủ công
            tp = group_df['is_correct'].sum()
            total = len(group_df)
            precision = tp / total if total > 0 else 0
            recall = tp / total if total > 0 else 0
            
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    def _compute_difficulty_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích hiệu suất dựa trên các mức độ khó khác nhau.
        
        Args:
            df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            
        Returns:
            Dict[str, Any]: Các metrics về hiệu suất theo độ khó
        """
        metrics = {}
        
        if 'difficulty' not in df.columns or 'is_correct' not in df.columns:
            logger.warning("Không thể tính difficulty metrics: thiếu cột difficulty hoặc is_correct")
            return metrics
        
        # Đảm bảo cột difficulty có giá trị
        df_valid = df.dropna(subset=['difficulty', 'is_correct'])
        
        if len(df_valid) == 0:
            logger.warning("Không có dữ liệu hợp lệ để tính difficulty metrics")
            return metrics
        
        # Tính accuracy theo độ khó
        accuracy_by_difficulty = df_valid.groupby('difficulty')['is_correct'].mean().to_dict()
        metrics['accuracy_by_difficulty'] = accuracy_by_difficulty
        
        # Tính accuracy theo model và độ khó
        accuracy_by_model_difficulty = df_valid.groupby(['model_name', 'difficulty'])['is_correct'].mean().unstack().to_dict('index')
        metrics['accuracy_by_model_difficulty'] = accuracy_by_model_difficulty
        
        # Tính accuracy theo prompt và độ khó
        accuracy_by_prompt_difficulty = df_valid.groupby(['prompt_type', 'difficulty'])['is_correct'].mean().unstack().to_dict('index')
        metrics['accuracy_by_prompt_difficulty'] = accuracy_by_prompt_difficulty
        
        # Phân tích mức độ cải thiện giữa các độ khó
        difficulty_levels = ['Dễ', 'Trung bình', 'Khó']
        valid_levels = [level for level in difficulty_levels if level in df_valid['difficulty'].unique()]
        
        if len(valid_levels) > 1:
            improvements = {}
            for model in df_valid['model_name'].unique():
                model_improvements = {}
                for i in range(len(valid_levels)-1):
                    easier = valid_levels[i]
                    harder = valid_levels[i+1]
                    
                    easier_acc = df_valid[(df_valid['model_name'] == model) & (df_valid['difficulty'] == easier)]['is_correct'].mean()
                    harder_acc = df_valid[(df_valid['model_name'] == model) & (df_valid['difficulty'] == harder)]['is_correct'].mean()
                    
                    # Tính sự suy giảm hiệu suất
                    if not np.isnan(easier_acc) and not np.isnan(harder_acc):
                        diff = harder_acc - easier_acc
                        model_improvements[f'{easier}_to_{harder}'] = diff
                
                improvements[model] = model_improvements
            
            metrics['difficulty_improvements'] = improvements
        
        return metrics
    
    def _compute_context_adherence_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán metrics cho đánh giá context adherence.
        
        Args:
            df (pd.DataFrame): DataFrame đã có đánh giá context adherence
            
        Returns:
            Dict: Metrics liên quan đến context adherence
        """
        metrics = {}
        
        # Lọc các loại prompt liên quan tới context (few-shot, react)
        context_prompt_mask = df['prompt_type'].str.contains('few-shot|react', case=False, na=False)
        df_context = df[context_prompt_mask]
        
        if len(df_context) == 0:
            return metrics
        
        # 1. Tỷ lệ câu trả lời đúng cho prompts liên quan context
        if 'is_correct' in df_context.columns:
            metrics['context_accuracy'] = df_context['is_correct'].mean()
        
        # 2. So sánh với non-context prompts
        non_context_mask = ~df['prompt_type'].str.contains('few-shot|react', case=False, na=False)
        df_non_context = df[non_context_mask]
        
        if len(df_non_context) > 0 and 'is_correct' in df_non_context.columns:
            metrics['non_context_accuracy'] = df_non_context['is_correct'].mean()
            
            # Tính delta accuracy
            context_acc = metrics.get('context_accuracy', 0)
            non_context_acc = metrics.get('non_context_accuracy', 0)
            metrics['context_accuracy_delta'] = context_acc - non_context_acc
        
        # Phân tích reasoning_cultural_context nếu có
        if 'reasoning_cultural_context' in df_context.columns:
            try:
                context_scores = df_context['reasoning_cultural_context'].dropna().tolist()
                
                if context_scores:
                    metrics['avg_context_adherence_score'] = sum(context_scores) / len(context_scores)
                    metrics['max_context_adherence_score'] = max(context_scores)
                    metrics['min_context_adherence_score'] = min(context_scores)
            except Exception as e:
                logger.error(f"Lỗi khi tính toán context adherence score: {str(e)}")
                logger.error(traceback.format_exc())
        
        return metrics

    def _compute_basic_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Tính toán các metrics cơ bản trên toàn bộ dataset.
        
        Args:
            df (pd.DataFrame): DataFrame kết quả
            
        Returns:
            Dict[str, float]: Các metrics cơ bản
        """
        metrics = {}
        
        # Tính toán accuracy tổng thể
        if 'is_correct' in df.columns:
            metrics['overall_accuracy'] = df['is_correct'].mean()
        
        # Tính toán thời gian trung bình
        if 'latency' in df.columns:
            metrics['average_latency'] = df['latency'].mean()
            metrics['max_latency'] = df['latency'].max()
            metrics['min_latency'] = df['latency'].min()
        
        # Tính toán độ dài phản hồi trung bình
        if 'response_length' in df.columns:
            metrics['average_response_length'] = df['response_length'].mean()
        
        return metrics
    
    def _compute_metrics_by_model_prompt(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán các metrics theo từng cặp model-prompt.
        
        Args:
            df (pd.DataFrame): DataFrame kết quả
            
        Returns:
            Dict[str, Dict[str, Any]]: Các metrics theo từng cặp model-prompt
        """
        metrics = {}
        
        # Xác định cột model (có thể là 'model_name' hoặc 'model')
        model_col = 'model_name' if 'model_name' in df.columns else 'model'
        
        if model_col not in df.columns or 'prompt_type' not in df.columns:
            return metrics
        
        # Lấy danh sách models và prompt types
        models = df[model_col].unique()
        prompt_types = df['prompt_type'].unique()
        
        # Tính metrics cho từng cặp model-prompt
        for model in models:
            metrics[model] = {}
            for prompt_type in prompt_types:
                mp_df = df[(df[model_col] == model) & (df['prompt_type'] == prompt_type)]
                
                if len(mp_df) == 0:
                    continue
                
                mp_metrics = {}
                
                # Accuracy (nếu có)
                if 'is_correct' in mp_df.columns:
                    mp_metrics['accuracy'] = mp_df['is_correct'].mean()
                
                # Latency (nếu có)
                if 'latency' in mp_df.columns:
                    mp_metrics['average_latency'] = mp_df['latency'].mean()
                    mp_metrics['max_latency'] = mp_df['latency'].max()
                    mp_metrics['min_latency'] = mp_df['latency'].min()
                
                # Token count (nếu có)
                if 'token_count' in mp_df.columns:
                    mp_metrics['average_token_count'] = mp_df['token_count'].mean()
                
                # Thêm các metrics khác nếu cần
                
                # Lưu metrics cho cặp model-prompt
                metrics[model][prompt_type] = mp_metrics
        
        return metrics
    
    def _compute_metrics_by_question_type(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Tính toán metrics theo loại câu hỏi.
        
        Args:
            df (pd.DataFrame): DataFrame kết quả
            
        Returns:
            Dict: Metrics theo loại câu hỏi
        """
        metrics = {}
        
        if 'question_type' not in df.columns:
            return metrics
        
        # Lặp qua từng loại câu hỏi
        for q_type in df['question_type'].unique():
            metrics[q_type] = {}
            type_df = df[df['question_type'] == q_type]
            
            # Tính accuracy cho loại câu hỏi này
            if 'is_correct' in df.columns:
                metrics[q_type]['accuracy'] = type_df['is_correct'].mean()
            
            # Tính thời gian trung bình
            if 'latency' in df.columns:
                metrics[q_type]['avg_latency'] = type_df['latency'].mean()
            
            # Tính số lượng câu hỏi
            metrics[q_type]['count'] = len(type_df)
        
        return metrics

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gộp hai dict một cách đệ quy.
        
        Args:
            base: Dict cơ sở
            override: Dict ghi đè
            
        Returns:
            Dict mới sau khi gộp
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
