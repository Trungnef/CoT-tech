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
        
        # Các tiêu chí đánh giá suy luận (điểm từ 1-5)
        self.reasoning_criteria = {
            "logical_flow": "Tính hợp lý và mạch lạc của lập luận (1-5)",
            "mathematical_correctness": "Độ chính xác về mặt toán học (1-5)",
            "clarity": "Rõ ràng và dễ hiểu (1-5)",
            "completeness": "Đầy đủ các bước cần thiết (1-5)",
            "relevance": "Mức độ liên quan đến câu hỏi (1-5)"
        }

        # Cấu trúc prompt đánh giá
        if language.lower() == "vietnamese":
            self.reasoning_eval_template = """
Bạn là một chuyên gia đánh giá chất lượng suy luận. Hãy đánh giá câu trả lời cho bài toán dưới đây theo 5 tiêu chí, đưa ra điểm số từ 1-5 (5 là tốt nhất).

BÀI TOÁN:
{question}

ĐÁP ÁN CHUẨN:
{correct_answer}

CÂU TRẢ LỜI CẦN ĐÁNH GIÁ:
{model_answer}

HÃY ĐÁNH GIÁ THEO CÁC TIÊU CHÍ SAU (điểm từ 1-5):
1. Tính hợp lý và mạch lạc của lập luận (logical_flow): ?/5
2. Độ chính xác về mặt toán học (mathematical_correctness): ?/5
3. Rõ ràng và dễ hiểu (clarity): ?/5
4. Đầy đủ các bước cần thiết (completeness): ?/5
5. Mức độ liên quan đến câu hỏi (relevance): ?/5

Điểm trung bình: ?/5

Giải thích ngắn gọn cho mỗi điểm:
"""
        else:
            self.reasoning_eval_template = """
You are an expert evaluating reasoning quality. Please evaluate the answer to the following problem according to 5 criteria, giving scores from 1-5 (5 being the best).

PROBLEM:
{question}

CORRECT ANSWER:
{correct_answer}

ANSWER TO EVALUATE:
{model_answer}

EVALUATE ACCORDING TO THESE CRITERIA (score 1-5):
1. Logical flow and coherence (logical_flow): ?/5
2. Mathematical correctness (mathematical_correctness): ?/5
3. Clarity and understandability (clarity): ?/5
4. Completeness of necessary steps (completeness): ?/5
5. Relevance to the question (relevance): ?/5

Average score: ?/5

Brief explanation for each score:
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
        
        # Tính toán metrics cơ bản
        analysis_results = self.analyze_results(self.results_df)
        
        # Đánh giá tính nhất quán trong self-consistency runs
        if self.results_df['prompt_type'].str.contains('consistency|cot_self_consistency', case=False).any():
            if self.verbose:
                logger.info("Đánh giá tính nhất quán trong các self-consistency runs")
            
            try:
                self.results_df = self.evaluate_consistency(self.results_df)
                analysis_results["consistency_metrics"] = self._compute_consistency_metrics(self.results_df)
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá tính nhất quán: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Đánh giá tính đầy đủ của câu trả lời
        if self.reasoning_config.get("evaluate_completeness", True):
            if self.verbose:
                logger.info("Đánh giá tính đầy đủ của câu trả lời")
                
            try:
                completeness_sample_size = self.reasoning_config.get("completeness_sample_size", self.sample_size)
                self.results_df = self.evaluate_completeness(
                    self.results_df,
                    sample_size=completeness_sample_size
                )
                
                analysis_results["completeness_metrics"] = self._compute_completeness_metrics(self.results_df)
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá tính đầy đủ: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Tính toán metrics đo lường độ tương đồng với đáp án chuẩn
        if 'correct_answer' in self.results_df.columns and self.reasoning_config.get("evaluate_similarity", True):
            if self.verbose:
                logger.info("Tính toán độ tương đồng với đáp án chuẩn")
                
            try:
                self.results_df = self.evaluate_similarity(self.results_df)
                analysis_results["similarity_metrics"] = self._compute_similarity_metrics(self.results_df)
            except Exception as e:
                logger.error(f"Lỗi khi tính toán độ tương đồng: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Phân tích lỗi nếu có các câu trả lời sai
        if 'is_correct' in self.results_df.columns and self.reasoning_config.get("error_analysis", True):
            try:
                error_sample_size = self.reasoning_config.get("error_sample_size", min(50, self.sample_size))
                
                if self.verbose:
                    logger.info(f"Thực hiện phân tích lỗi với sample_size={error_sample_size}")
                
                self.results_df = self.analyze_errors(
                    self.results_df,
                    sample_size=error_sample_size
                )
                
                analysis_results["error_analysis"] = self._compute_error_metrics(self.results_df)
            except Exception as e:
                logger.error(f"Lỗi khi phân tích lỗi: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Đánh giá khả năng suy luận nếu được bật
        if self.reasoning_config.get("enabled", True) and 'correct_answer' in self.results_df.columns:
            try:
                if self.verbose:
                    logger.info(f"Đánh giá chất lượng suy luận với sample_size={self.sample_size}")
                
                # Thực hiện đánh giá suy luận cho mẫu ngẫu nhiên
                self.results_df = self.evaluate_reasoning_quality(
                    self.results_df, 
                    sample_size=self.sample_size
                )
                
                # Tính metrics cho kết quả đánh giá suy luận
                reasoning_metrics = self._compute_reasoning_metrics(self.results_df)
                analysis_results["reasoning_metrics"] = reasoning_metrics
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá khả năng suy luận: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Lưu kết quả phân tích vào instance để có thể truy cập sau
        self.analysis_results = analysis_results
        
        # Hiển thị tóm tắt nếu ở chế độ verbose
        if self.verbose:
            logger.info("\n" + self.export_summary(analysis_results))
        
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

    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích DataFrame kết quả và tính toán tất cả các metrics.
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            
        Returns:
            Dict[str, Any]: Dictionary chứa tất cả metrics và kết quả phân tích
        """
        if self.verbose:
            logger.info(f"🔍 Phân tích kết quả cho {len(results_df)} mục")
        
        # 1. Tính toán metrics cơ bản
        basic_metrics = self._compute_basic_metrics(results_df)
        
        # 2. Tính các metrics theo model và prompt type
        model_prompt_metrics = self._compute_metrics_by_model_prompt(results_df)
        
        # 3. Tính toán metrics theo loại câu hỏi (nếu có thông tin)
        question_type_metrics = {}
        if 'question_type' in results_df.columns:
            question_type_metrics = self._compute_metrics_by_question_type(results_df)
        
        # 4. Kết quả của phân tích
        analysis_results = {
            "basic_metrics": basic_metrics,
            "model_prompt_metrics": model_prompt_metrics,
            "question_type_metrics": question_type_metrics,
            "raw_results": results_df
        }
        
        return analysis_results
    
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
    
    def _compute_metrics_by_model_prompt(self, df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Tính toán metrics theo từng model và prompt type.
        
        Args:
            df (pd.DataFrame): DataFrame kết quả
            
        Returns:
            Dict: Metrics theo model và prompt type
        """
        metrics = {}
        
        # Kiểm tra các cột cần thiết
        model_col = 'model_name' if 'model_name' in df.columns else ('model' if 'model' in df.columns else None)
        prompt_col = 'prompt_type' if 'prompt_type' in df.columns else None
        
        if not model_col or not prompt_col:
            logger.warning(f"DataFrame thiếu cột cần thiết để tính metrics theo model/prompt. "
                         f"Cần có 'model_name'/'model' và 'prompt_type', hiện có: {list(df.columns)}")
            
            # Tạo một bộ metrics đơn giản với dữ liệu khả dụng
            if len(df) > 0:
                # Tạo các cột giả nếu không có
                if not model_col:
                    df['model_name'] = 'unknown_model'
                    model_col = 'model_name'
                if not prompt_col:
                    df['prompt_type'] = 'unknown_prompt'
                    prompt_col = 'prompt_type'
            else:
                return metrics
        
        # Lặp qua từng model
        for model in df[model_col].unique():
            metrics[model] = {}
            model_df = df[df[model_col] == model]
            
            # Lặp qua từng prompt type
            for prompt_type in model_df[prompt_col].unique():
                metrics[model][prompt_type] = {}
                prompt_df = model_df[model_df[prompt_col] == prompt_type]
                
                # Tính accuracy nếu có cột is_correct
                if 'is_correct' in df.columns:
                    metrics[model][prompt_type]['accuracy'] = prompt_df['is_correct'].mean()
                
                # Tính thời gian trung bình - kiểm tra cả hai tên cột có thể có
                latency_col = None
                for col_name in ['latency', 'elapsed_time']:
                    if col_name in df.columns:
                        latency_col = col_name
                        break
                        
                if latency_col:
                    metrics[model][prompt_type]['avg_latency'] = prompt_df[latency_col].mean()
                
                # Tính độ dài phản hồi trung bình
                if 'response_length' in df.columns:
                    metrics[model][prompt_type]['avg_response_length'] = prompt_df['response_length'].mean()
                elif 'token_count' in df.columns:
                    metrics[model][prompt_type]['avg_token_count'] = prompt_df['token_count'].mean()
                
                # Thêm số lượng mẫu cho mỗi tổ hợp model/prompt
                metrics[model][prompt_type]['sample_count'] = len(prompt_df)
        
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
        valid_rows = (
            has_reasoning & 
            results_df['reasoning_avg_score'].isna() &  # Chưa được đánh giá
            ~results_df['correct_answer'].isna() &  # Có đáp án đúng
            ~results_df['response'].isna()  # Có câu trả lời
        )
        
        valid_indices = results_df[valid_rows].index.tolist()
        
        if not valid_indices:
            logger.warning("Không có mẫu phù hợp để đánh giá suy luận")
            return results_df
            
        # Lấy mẫu ngẫu nhiên nếu cần
        np.random.seed(random_seed)
        if len(valid_indices) > sample_size:
            sample_indices = np.random.choice(valid_indices, size=sample_size, replace=False)
        else:
            sample_indices = valid_indices
            
        logger.info(f"Đánh giá suy luận cho {len(sample_indices)} mẫu")
        
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
                criteria_scores = [v for k, v in eval_result.items() if k in self.reasoning_criteria]
                avg_score = sum(criteria_scores) / len(criteria_scores) if criteria_scores else 0
                results_df.at[idx, 'reasoning_avg_score'] = avg_score
                
                # Lưu giải thích đánh giá
                if 'explanation' in eval_result:
                    results_df.at[idx, 'reasoning_evaluation'] = eval_result['explanation']
                    
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá suy luận cho mẫu {idx}: {str(e)}")
                logger.error(traceback.format_exc())
        
        return results_df
    
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
        reasoning_cols = [col for col in df.columns if col.startswith('reasoning_') and col != 'reasoning_evaluation']
        if not reasoning_cols:
            return metrics
        
        # 1. Metrics tổng thể
        metrics["overall"] = {}
        for col in reasoning_cols:
            criterion = col.replace('reasoning_', '')
            metrics["overall"][criterion] = df[col].mean()
        
        # 2. Metrics theo model
        metrics["by_model"] = {}
        for model in df['model_name'].unique():
            metrics["by_model"][model] = {}
            model_df = df[df['model_name'] == model]
            
            for col in reasoning_cols:
                criterion = col.replace('reasoning_', '')
                metrics["by_model"][model][criterion] = model_df[col].mean()
        
        # 3. Metrics theo prompt type
        metrics["by_prompt_type"] = {}
        for prompt in df['prompt_type'].unique():
            metrics["by_prompt_type"][prompt] = {}
            prompt_df = df[df['prompt_type'] == prompt]
            
            for col in reasoning_cols:
                criterion = col.replace('reasoning_', '')
                metrics["by_prompt_type"][prompt][criterion] = prompt_df[col].mean()
        
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
                    metrics["by_model_prompt"][model][prompt][criterion] = prompt_df[col].mean()
        
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
