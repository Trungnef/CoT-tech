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
 # Kiểm tra các thư viện cần thiết
import torch
import nltk
from nltk.tokenize import word_tokenize

# Import các module cần thiết
try:
    from ..utils.metrics_utils import calculate_bleu_scores, calculate_exact_match_accuracy, calculate_rouge_scores, calculate_consistency_metrics
except ImportError:
    # Fallback khi chạy module trực tiếp
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.metrics_utils import calculate_bleu_scores, calculate_exact_match_accuracy, calculate_rouge_scores, calculate_consistency_metrics

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
# HƯỚNG DẪN ĐÁNH GIÁ CHẤT LƯỢNG ĐẦU RA CỦA MÔ HÌNH LLM CHO BÀI TOÁN TIẾNG VIỆT

Bạn là một chuyên gia đánh giá chất lượng đầu ra của các mô hình ngôn ngữ lớn (LLMs) chuyên về tiếng Việt. 
Nhiệm vụ của bạn là đánh giá câu trả lời của một mô hình LLM cho một bài toán cụ thể dựa trên các tiêu chí khách quan và phù hợp với ngữ cảnh tiếng Việt.

## TIÊU CHÍ ĐÁNH GIÁ

1. **Độ chính xác (Accuracy)**
   - Câu trả lời có đúng về mặt nội dung và kết quả so với đáp án chuẩn không?
   - Với bài toán số học: kết quả cuối cùng có đúng không? Chấp nhận các cách diễn đạt khác nhau nếu kết quả là chính xác.
   - Với bài toán lý luận: kết luận có chính xác không? Xem xét bối cảnh văn hóa và ngôn ngữ tiếng Việt.
   - Điểm 5: Hoàn toàn chính xác (kết quả đúng, cách giải phù hợp)
   - Điểm 4: Phần lớn chính xác (kết quả đúng, có lỗi nhỏ trong quá trình)
   - Điểm 3: Chính xác một phần (kết quả đúng nhưng cách giải có vấn đề)
   - Điểm 2: Phần lớn không chính xác (kết quả sai nhưng cách tiếp cận hợp lý)
   - Điểm 1: Hoàn toàn sai (kết quả sai và cách giải không phù hợp)

2. **Độ suy luận hợp lý (Reasoning Consistency)**
   - Quá trình lập luận có logic và có cấu trúc rõ ràng không?
   - Các bước suy luận có thể theo dõi và kiểm chứng được không?
   - Có sử dụng đúng các khái niệm, thuật ngữ tiếng Việt và phù hợp với bối cảnh văn hóa không?
   - Có sai sót logic trong các bước lập luận không?
   - Điểm 5: Lập luận xuất sắc, rõ ràng và đầy đủ
   - Điểm 4: Lập luận tốt, có thể theo dõi được, có ít lỗi nhỏ
   - Điểm 3: Lập luận trung bình, có thể hiểu được, có một số lỗi
   - Điểm 2: Lập luận kém, khó theo dõi, nhiều lỗi
   - Điểm 1: Lập luận rất kém, mâu thuẫn hoặc sai cơ bản

3. **Tính nhất quán (Consistency)**
   - Câu trả lời có nhất quán từ đầu đến cuối không?
   - Không có mâu thuẫn giữa các phần trong câu trả lời?
   - Các định nghĩa và ký hiệu được sử dụng nhất quán?
   - Có sự thống nhất trong cách dùng thuật ngữ tiếng Việt không?
   - Điểm 5: Hoàn toàn nhất quán, không có mâu thuẫn
   - Điểm 4: Khá nhất quán, có ít mâu thuẫn nhỏ nhưng không ảnh hưởng kết quả
   - Điểm 3: Tương đối nhất quán, có một số mâu thuẫn nhỏ
   - Điểm 2: Thiếu nhất quán, có nhiều mâu thuẫn
   - Điểm 1: Rất mâu thuẫn, khó hiểu logic tổng thể

4. **Hiệu suất phù hợp với độ khó (Difficulty Performance)**
   - Câu trả lời có phù hợp với độ khó của bài toán không?
   - Mô hình có xử lý đầy đủ độ phức tạp của bài toán dành cho học sinh Việt Nam không?
   - Có đi đúng trọng tâm của bài toán không?
   - Điểm 5: Xử lý xuất sắc bài toán theo đúng độ khó
   - Điểm 4: Xử lý tốt bài toán, phù hợp với độ khó
   - Điểm 3: Xử lý trung bình, có thể chưa đáp ứng đầy đủ yêu cầu của độ khó
   - Điểm 2: Xử lý kém, không đáp ứng được độ khó của bài toán
   - Điểm 1: Không thể xử lý được bài toán ở mức độ cơ bản nhất

5. **Độ phù hợp ngữ cảnh (Context Adherence)**
   - Câu trả lời có tận dụng tốt ngữ cảnh/ví dụ được cung cấp không?
   - Áp dụng đúng các mẫu/cấu trúc từ ngữ cảnh vào bài giải?
   - Có sử dụng ngôn ngữ phù hợp với văn hóa và cách diễn đạt tiếng Việt không?
   - Điểm 5: Tận dụng tối đa ngữ cảnh một cách hiệu quả và phù hợp văn hóa
   - Điểm 4: Tận dụng tốt ngữ cảnh và phù hợp văn hóa
   - Điểm 3: Tận dụng ngữ cảnh ở mức trung bình
   - Điểm 2: Ít tận dụng ngữ cảnh được cung cấp

## BÀI TOÁN CẦN GIẢI QUYẾT

{question}

## ĐÁP ÁN CHUẨN

{correct_answer}

## CÂU TRẢ LỜI CỦA MÔ HÌNH CẦN ĐÁNH GIÁ

{model_answer}

## ĐÁNH GIÁ THEO THANG ĐIỂM 5

Hãy đánh giá và cho điểm từ 1-5 cho từng tiêu chí, trong đó 1 là kém nhất và 5 là tốt nhất. 
Hãy khách quan dựa trên chất lượng thực tế của câu trả lời, không quá khắt khe với các lỗi nhỏ nếu chúng không ảnh hưởng đến kết quả cuối cùng:

1. Độ chính xác (accuracy): ?/5
2. Độ suy luận hợp lý (reasoning): ?/5
3. Tính nhất quán (consistency): ?/5
4. Hiệu suất phù hợp với độ khó (difficulty_performance): ?/5
5. Độ phù hợp ngữ cảnh (context_adherence): ?/5

Điểm trung bình (average): ?/5

## GIẢI THÍCH CHI TIẾT

- Độ chính xác: [giải thích chi tiết đánh giá độ chính xác, nêu rõ kết quả có đúng không và lý do]
- Độ suy luận hợp lý: [giải thích chi tiết về chất lượng suy luận, đánh giá từng bước trong quá trình giải]
- Tính nhất quán: [giải thích về tính nhất quán trong câu trả lời, nêu rõ có mâu thuẫn nào không]
- Hiệu suất phù hợp với độ khó: [giải thích mức độ phù hợp với độ khó của bài toán]
- Độ phù hợp ngữ cảnh: [giải thích việc sử dụng ngữ cảnh và sự phù hợp với văn hóa tiếng Việt]

## KẾT LUẬN TỔNG THỂ

[nhận xét tổng quan về chất lượng câu trả lời, nêu bật ưu điểm, nhược điểm chính]
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
            'context_metrics': {},
            'error_analysis_metrics': {},  # Thêm khóa mới cho metrics phân tích lỗi
            'advanced_metrics': {},  # Thêm khóa mới cho các metrics nâng cao
            'errors': []  # Thêm trường để ghi lại các lỗi
        }
        
        # Tính toán metrics cơ bản
        try:
            analysis_results['basic_metrics'] = self._compute_basic_metrics(self.results_df)
        except Exception as e:
            error_msg = f"Lỗi khi tính metrics cơ bản: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            analysis_results['errors'].append({'step': 'basic_metrics', 'error': error_msg})
        
        # Tính toán metrics theo model và prompt type
        try:
            analysis_results['model_prompt_metrics'] = self._compute_metrics_by_model_prompt(self.results_df)
        except Exception as e:
            error_msg = f"Lỗi khi tính metrics theo model và prompt: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            analysis_results['errors'].append({'step': 'model_prompt_metrics', 'error': error_msg})
        
        # Tính toán metrics theo loại câu hỏi (Thêm chi tiết hơn)
        if 'question_type' in self.results_df.columns:
            try:
                analysis_results['question_type_metrics'] = self._compute_metrics_by_question_type(self.results_df)
                
                # Thêm phân tích chuyên sâu theo loại câu hỏi
                if 'is_correct' in self.results_df.columns:
                    for q_type in self.results_df['question_type'].unique():
                        type_df = self.results_df[self.results_df['question_type'] == q_type]
                        acc = type_df['is_correct'].mean()
                        analysis_results['question_type_metrics'][q_type]['detail'] = {
                            'accuracy': acc,
                            'count': len(type_df),
                            'avg_token_count': type_df['token_count'].mean() if 'token_count' in type_df.columns else None,
                            'avg_latency': type_df['latency'].mean() if 'latency' in type_df.columns else None
                        }
            except Exception as e:
                error_msg = f"Lỗi khi tính metrics theo loại câu hỏi: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                analysis_results['errors'].append({'step': 'question_type_metrics', 'error': error_msg})
        
        # Đánh giá theo các tiêu chí mới
        # 1. Accuracy
        if 'is_correct' in self.results_df.columns:
            try:
                analysis_results['accuracy_metrics'] = self._compute_accuracy_metrics(self.results_df)
            except Exception as e:
                error_msg = f"Lỗi khi tính accuracy metrics: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                analysis_results['errors'].append({'step': 'accuracy_metrics', 'error': error_msg})
        
        # 2. Reasoning Consistency
        reasoning_columns = [col for col in self.results_df.columns if col.startswith('reasoning_') and col != 'reasoning_scores_str']
        if reasoning_columns:
            try:
                analysis_results['reasoning_metrics'] = self._compute_reasoning_metrics(self.results_df)
            except Exception as e:
                error_msg = f"Lỗi khi tính reasoning metrics: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                analysis_results['errors'].append({'step': 'reasoning_metrics', 'error': error_msg})
        
        # 3. Consistency
        consistency_prompts = self.results_df['prompt_type'].str.contains('consistency|cot_self_consistency', case=False, na=False)
        if consistency_prompts.any():
            if self.verbose:
                logger.info("Đánh giá tính nhất quán trong các self-consistency runs")
            
            try:
                self.results_df = self.evaluate_consistency(self.results_df)
                analysis_results["consistency_metrics"] = self._compute_consistency_metrics(self.results_df)
            except Exception as e:
                error_msg = f"Lỗi khi đánh giá tính nhất quán: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                analysis_results['errors'].append({'step': 'consistency_metrics', 'error': error_msg})
        
        # 4. Performance on different difficulty levels
        if 'difficulty' in self.results_df.columns:
            try:
                analysis_results['difficulty_metrics'] = self._compute_difficulty_metrics(self.results_df)
            except Exception as e:
                error_msg = f"Lỗi khi tính difficulty metrics: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                analysis_results['errors'].append({'step': 'difficulty_metrics', 'error': error_msg})
        
        # 5. Context adherence
        if 'prompt_type' in self.results_df.columns:
            try:
                analysis_results['context_metrics'] = self._compute_context_adherence_metrics(self.results_df)
            except Exception as e:
                error_msg = f"Lỗi khi tính context adherence metrics: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                analysis_results['errors'].append({'step': 'context_metrics', 'error': error_msg})
        
        # 6. Phân tích lỗi (Error Analysis)
        if 'is_correct' in self.results_df.columns:
            if self.verbose:
                logger.info("Thực hiện phân tích lỗi cho các câu trả lời sai")
            
            try:
                # Số lượng mẫu phân tích lỗi (mặc định 50)
                error_sample_size = 50
                
                # Chạy phân tích lỗi
                self.results_df = self.analyze_errors(self.results_df, sample_size=error_sample_size)
                
                # Tính metrics từ kết quả phân tích lỗi
                analysis_results['error_analysis_metrics'] = self._compute_error_metrics(self.results_df)
                
                if self.verbose:
                    # Tính số lượng mẫu đã phân tích lỗi
                    error_analyzed = sum(self.results_df['error_type'] != '')
                    logger.info(f"Đã phân tích lỗi cho {error_analyzed} câu trả lời")
            except Exception as e:
                error_msg = f"Lỗi khi thực hiện phân tích lỗi: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                analysis_results['errors'].append({'step': 'error_analysis', 'error': error_msg})
        
        # 7. Tính toán F1-score, precision, recall ở cấp độ model/prompt 
        if 'is_correct' in self.results_df.columns:
            try:
                f1_metrics = self._compute_f1_metrics(self.results_df)
                analysis_results['advanced_metrics'].update(f1_metrics)
                if self.verbose:
                    logger.info(f"Đã tính F1 scores cho {len(f1_metrics)} model/prompt combinations")
            except Exception as e:
                error_msg = f"Lỗi khi tính F1 metrics: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                analysis_results['errors'].append({'step': 'f1_metrics', 'error': error_msg})
        
        # 8. Tính toán METEOR và BERTScore cho văn bản
        if 'response' in self.results_df.columns and 'correct_answer' in self.results_df.columns:
            try:
                from ..utils.metrics_utils import calculate_meteor_score, calculate_bertscore
                
                # Lấy mẫu để giảm thời gian tính toán (chỉ với BERTScore vì nó nặng)
                sample_size = min(100, len(self.results_df))
                sample_df = self.results_df.sample(sample_size, random_state=42) if len(self.results_df) > sample_size else self.results_df
                
                # Tính METEOR
                meteor_results = {}
                for model in sample_df['model_name'].unique():
                    model_df = sample_df[sample_df['model_name'] == model]
                    meteor_score = calculate_meteor_score(
                        predictions=model_df['response'].tolist(),
                        references=model_df['correct_answer'].tolist()
                    )
                    meteor_results[model] = meteor_score.get('meteor', 0)
                
                # Tính BERTScore - chỉ trên mẫu nhỏ hơn vì tính toán nặng
                bertscore_results = {}
                bert_sample_size = min(50, sample_size)
                bert_sample_df = sample_df.sample(bert_sample_size, random_state=42) if len(sample_df) > bert_sample_size else sample_df
                
                for model in bert_sample_df['model_name'].unique():
                    model_df = bert_sample_df[bert_sample_df['model_name'] == model]
                    bert_scores = calculate_bertscore(
                        predictions=model_df['response'].tolist(),
                        references=model_df['correct_answer'].tolist(),
                        lang="vi" if self.language == "vietnamese" else "en"
                    )
                    bertscore_results[model] = {
                        'precision': bert_scores.get('bertscore_precision', 0),
                        'recall': bert_scores.get('bertscore_recall', 0),
                        'f1': bert_scores.get('bertscore_f1', 0)
                    }
                
                # Thêm kết quả vào advanced_metrics
                analysis_results['advanced_metrics']['meteor_scores'] = meteor_results
                analysis_results['advanced_metrics']['bertscore'] = bertscore_results
                
                if self.verbose:
                    logger.info(f"Đã tính METEOR scores cho {len(meteor_results)} models")
                    logger.info(f"Đã tính BERTScore cho {len(bertscore_results)} models")
                    
            except Exception as e:
                error_msg = f"Lỗi khi tính METEOR/BERTScore metrics: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                analysis_results['errors'].append({'step': 'semantic_metrics', 'error': error_msg})
        
        # 8. Tính thêm các metrics nâng cao ở cấp độ mẫu (F1 token overlap, METEOR, BERT)
        try:
            self.results_df = self.calculate_additional_metrics()
            
            # Thêm thông tin tổng hợp vào advanced_metrics
            if 'f1_score' in self.results_df.columns:
                f1_by_model = self.results_df.groupby('model_name')['f1_score'].mean().to_dict()
                analysis_results['advanced_metrics']['f1_token_overlap'] = f1_by_model
            
            if 'meteor_score' in self.results_df.columns:
                meteor_by_model = self.results_df.groupby('model_name')['meteor_score'].mean().to_dict()
                # Đảm bảo chỉ cập nhật, không ghi đè meteor_scores nếu đã có
                if 'meteor_scores' not in analysis_results['advanced_metrics']:
                    analysis_results['advanced_metrics']['meteor_scores'] = meteor_by_model
                else:
                    analysis_results['advanced_metrics']['meteor_scores'].update(meteor_by_model)
            
            if 'bert_score' in self.results_df.columns:
                bert_by_model = self.results_df.groupby('model_name')['bert_score'].mean().to_dict()
                for model, score in bert_by_model.items():
                    if 'bertscore' not in analysis_results['advanced_metrics']:
                        analysis_results['advanced_metrics']['bertscore'] = {}
                    if model not in analysis_results['advanced_metrics']['bertscore']:
                        analysis_results['advanced_metrics']['bertscore'][model] = {'f1': score}
                    else:
                        analysis_results['advanced_metrics']['bertscore'][model]['f1'] = score
                
            if self.verbose:
                logger.info("Đã tính xong các metrics nâng cao ở cấp độ mẫu")
                
        except Exception as e:
            error_msg = f"Lỗi khi tính additional metrics: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            analysis_results['errors'].append({'step': 'additional_metrics', 'error': error_msg})
        
        # Lưu kết quả phân tích vào đối tượng
        self.analysis_results = analysis_results
        
        if self.verbose:
            logger.info("✅ Đã hoàn thành phân tích kết quả đánh giá")
            
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
        Đánh giá chất lượng lý luận của câu trả lời mô hình.
        Đánh giá này sử dụng một mô hình ngôn ngữ lớn (model_reasoning) để phân tích.
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            sample_size (int): Số lượng mẫu để đánh giá
            random_seed (int): Seed ngẫu nhiên cho việc lấy mẫu
            
        Returns:
            pd.DataFrame: DataFrame đã bổ sung đánh giá chất lượng lý luận
        """
        # Đảm bảo các cột cần thiết tồn tại
        for col in ['question_text', 'response', 'correct_answer']:
            if col not in results_df.columns:
                logger.warning(f"Thiếu cột {col} để đánh giá chất lượng lý luận")
                return results_df
        
        # Thêm các cột đánh giá nếu chưa có
        for criterion in self.reasoning_criteria.keys():
            col_name = f"reasoning_{criterion}"
            if col_name not in results_df.columns:
                results_df[col_name] = None
        
        if "reasoning_average" not in results_df.columns:
            results_df["reasoning_average"] = None
        
        if "reasoning_explanation" not in results_df.columns:
            results_df["reasoning_explanation"] = None
        
        # Lọc bỏ các hàng đã có đánh giá
        unevaluated_mask = results_df['reasoning_accuracy'].isna()
        unevaluated_df = results_df[unevaluated_mask]
        
        if len(unevaluated_df) == 0:
            logger.info("Tất cả các câu trả lời đã được đánh giá chất lượng lý luận")
            return results_df
        
        # Lấy mẫu để đánh giá
        sample_size = min(sample_size, len(unevaluated_df))
        random.seed(random_seed)
        sample_indices = random.sample(list(unevaluated_df.index), sample_size)
        
        logger.info(f"Đánh giá chất lượng lý luận cho {sample_size} câu trả lời ngẫu nhiên")
        
        # Đánh giá từng mẫu
        evaluation_count = 0
        error_count = 0
        for idx in sample_indices:
            try:
                question = results_df.at[idx, 'question_text']
                model_answer = results_df.at[idx, 'response']
                correct_answer = results_df.at[idx, 'correct_answer']
                
                # Kiểm tra dữ liệu hợp lệ
                if not isinstance(question, str) or not isinstance(model_answer, str):
                    logger.warning(f"Dữ liệu không hợp lệ cho index {idx}: question_text hoặc response không phải chuỗi")
                    continue
                
                if not question or not model_answer:
                    logger.warning(f"Dữ liệu rỗng cho index {idx}")
                    continue
                
                # Đánh giá chất lượng lý luận
                evaluation_result = self._evaluate_single_reasoning(question, correct_answer, model_answer)
                
                if evaluation_result:
                    # Cập nhật DataFrame
                    for criterion, score in evaluation_result.items():
                        if criterion != 'explanation' and criterion != 'avg_score':
                            results_df.at[idx, f"reasoning_{criterion}"] = score
                    
                    results_df.at[idx, "reasoning_average"] = evaluation_result.get('avg_score', 0)
                    results_df.at[idx, "reasoning_explanation"] = evaluation_result.get('explanation', '')
                    
                    # Lưu chuỗi JSON của đánh giá
                    json_str = {k: v for k, v in evaluation_result.items() if k != 'explanation'}
                    results_df.at[idx, "reasoning_scores_str"] = str(json_str)
                    
                    evaluation_count += 1
                    
                    # Log kết quả định kỳ
                    if evaluation_count % 5 == 0:
                        logger.info(f"Đã đánh giá {evaluation_count}/{sample_size} mẫu")
            
            except Exception as e:
                error_count += 1
                error_msg = f"Lỗi khi đánh giá reasoning cho index {idx}: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                
                # Ghi nhận lỗi vào DataFrame nhưng vẫn tiếp tục với các mẫu khác
                results_df.at[idx, "reasoning_error"] = error_msg
                
                # Nếu quá nhiều lỗi, dừng lại
                if error_count > sample_size / 3:  # Dừng nếu hơn 1/3 số mẫu bị lỗi
                    logger.error(f"Quá nhiều lỗi ({error_count}) khi đánh giá reasoning. Dừng quá trình đánh giá.")
                    break
        
        logger.info(f"Hoàn thành đánh giá reasoning: {evaluation_count} thành công, {error_count} lỗi")
        
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
            # Thêm nhiều thông tin trong cấu hình generation để tránh lỗi
            generation_config = {
                "temperature": 0.1,  # Giảm temperature để có kết quả ổn định
                "max_tokens": 1000,   # Đủ dài cho đánh giá chi tiết
                "top_p": 0.95,
                "top_k": 40
            }
            
            # Nếu là Gemini, thêm tên model cụ thể
            if "gemini" in self.reasoning_model.lower():
                generation_config["model"] = "gemini-1.5-pro"  # Hoặc model_name phù hợp
                
            # Gọi hàm generate_text với cấu hình mở rộng
            eval_response, stats = generate_text(
                model_name=self.reasoning_model,
                prompt=evaluation_prompt,
                generation_config=generation_config
            )
            
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
        Tính toán các metrics về chất lượng lập luận từ kết quả đánh giá.
        Bao gồm phân tích theo loại câu hỏi.
        
        Args:
            df (pd.DataFrame): DataFrame kết quả đánh giá
            
        Returns:
            Dict[str, Any]: Các metrics về chất lượng lập luận
        """
        # Code hiện có để tính toán reasoning metrics
        
        # Thêm phân tích reasoning theo loại câu hỏi
        reasoning_by_question_type = {}
        
        if 'question_type' in df.columns and any(col.startswith('reasoning_') for col in df.columns):
            for q_type in df['question_type'].unique():
                type_df = df[df['question_type'] == q_type]
                
                type_metrics = {}
                for criteria in ['reasoning_accuracy', 'reasoning_logic', 'reasoning_consistency', 
                               'reasoning_difficulty', 'reasoning_context', 'reasoning_average']:
                    if criteria in df.columns:
                        type_metrics[criteria] = type_df[criteria].mean()
                
                reasoning_by_question_type[q_type] = type_metrics
        
        result = {
            'reasoning_metrics': {},  # Giữ nguyên metrics hiện có
            'reasoning_by_question_type': reasoning_by_question_type  # Thêm metrics mới
        }
        
        return result
    
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
        
        # Advanced metrics
        advanced_metrics = analysis_results.get('advanced_metrics', {})
        if advanced_metrics:
            summary.append("\n## Metrics Nâng Cao")
            
            # F1 Scores
            has_f1_data = False
            f1_data = {}
            
            # Priority to use f1_token_overlap if available
            if 'f1_token_overlap' in advanced_metrics:
                f1_data = advanced_metrics['f1_token_overlap']
                has_f1_data = True
            else:
                for key, metrics in advanced_metrics.items():
                    if isinstance(metrics, dict) and key not in ('meteor_scores', 'bertscore'):
                        if isinstance(metrics, dict) and any('f1' in v for v in metrics.values() if isinstance(v, dict)):
                            for model, model_metrics in metrics.items():
                                if model.endswith('_by_prompt'):
                                    continue  # Skip by_prompt data
                                if isinstance(model_metrics, dict):
                                    if 'f1' in model_metrics:
                                        f1_data[model] = model_metrics['f1']
                                        has_f1_data = True
            
            # METEOR Scores
            meteor_data = {}
            has_meteor_data = False
            if 'meteor_scores' in advanced_metrics and advanced_metrics['meteor_scores']:
                meteor_data = advanced_metrics['meteor_scores']
                has_meteor_data = True
            
            # BERT Scores
            bert_data = {}
            has_bert_data = False
            if 'bertscore' in advanced_metrics and advanced_metrics['bertscore']:
                for model, scores in advanced_metrics['bertscore'].items():
                    if isinstance(scores, dict) and 'f1' in scores:
                        bert_data[model] = scores['f1']
                        has_bert_data = True
            
            # F1 Score Table
            if has_f1_data:
                summary.append("\n### F1-Score (Token Overlap)")
                summary.append("| Model | F1-Score |")
                summary.append("|-------|----------|")
                
                for model, value in f1_data.items():
                    summary.append(f"| {model} | {value:.4f} |")
            
            # METEOR Score Table
            if has_meteor_data:
                summary.append("\n### METEOR Score")
                summary.append("| Model | METEOR Score |")
                summary.append("|-------|--------------|")
                
                for model, value in meteor_data.items():
                    summary.append(f"| {model} | {value:.4f} |")
            
            # BERT Score Table
            if has_bert_data:
                summary.append("\n### BERT-Score")
                summary.append("| Model | BERT-Score |")
                summary.append("|-------|-----------|")
                
                for model, value in bert_data.items():
                    summary.append(f"| {model} | {value:.4f} |")
        
        return "\n".join(summary)
    
    def evaluate_consistency(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Đánh giá tính nhất quán trong các self-consistency runs.
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            
        Returns:
            pd.DataFrame: DataFrame đã bổ sung đánh giá tính nhất quán
        """
        try:
            # Import calculate_consistency_metrics từ metrics_utils
            # Đảm bảo thư mục gốc nằm trong sys.path
            import sys
            import os
            import numpy as np
            import logging
            
            # Thiết lập logging
            logger = logging.getLogger(__name__)
            
            # Thêm thư mục gốc vào sys.path để tránh lỗi import
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Thư mục gốc của dự án
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                logger.info(f"Đã thêm {parent_dir} vào sys.path")
            
            from llm_evaluation.utils.metrics_utils import calculate_consistency_metrics
        except ImportError as e:
            logger.error(f"Lỗi khi import module calculate_consistency_metrics: {str(e)}")
            logger.error("Đảm bảo thư mục gốc của dự án đã được thêm vào sys.path")
            return results_df
        
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
        
        if 'consistency_unique_answers' not in results_df.columns:
            results_df['consistency_unique_answers'] = np.nan
        
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
            
            # Sử dụng hàm calculate_consistency_metrics để tính metrics
            group_key = f"{model}_{question_id}_{prompt_type}"
            try:
                consistency_results = calculate_consistency_metrics(
                    responses=[responses],  # Truyền vào list của list
                    final_answers=[final_answers] if final_answers != responses else None,
                    groupby_keys=[group_key]
                )
                
                # Lấy kết quả cho nhóm này
                if group_key in consistency_results:
                    metrics = consistency_results[group_key]
                    
                    # Cập nhật điểm nhất quán cho từng dòng trong nhóm
                    for idx in group.index:
                        results_df.at[idx, 'consistency_score'] = metrics["consistency_score"]
                        results_df.at[idx, 'consistency_agreement_rate'] = metrics["agreement_rate"]
                        results_df.at[idx, 'consistency_most_common'] = metrics["most_common_answer"]
                        results_df.at[idx, 'consistency_unique_answers'] = metrics["unique_answers"]
            except Exception as e:
                logger.error(f"Lỗi khi tính toán consistency metrics cho nhóm {group_key}: {str(e)}")
        
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
            # Triển khai với sentence-transformers
            from sentence_transformers import SentenceTransformer
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Lazy loading model
            if not hasattr(self, '_embedding_model'):
                logger.info(f"Đang tải embedding model: {self.similarity_model}")
                self._embedding_model = SentenceTransformer(self.similarity_model)
                
            # Tiền xử lý text
            # Chuẩn hóa khoảng trắng và loại bỏ ký tự đặc biệt nếu cần
            text1 = " ".join(text1.split())
            text2 = " ".join(text2.split())
            
            # Đảm bảo text không rỗng
            if not text1.strip() or not text2.strip():
                return 0.0
                
            # Tính embeddings
            embedding1 = self._embedding_model.encode([text1])[0]
            embedding2 = self._embedding_model.encode([text2])[0]
            
            # Chuẩn hóa embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Tính cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
                
        except ImportError:
            logger.warning("Không thể tính embedding similarity vì thiếu thư viện sentence-transformers. Hãy cài đặt: pip install sentence-transformers")
            return 0.0
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
        available_cols = [col for col in similarity_cols if col in df.columns]
        
        # Lọc các hàng có ít nhất một metric similarity được tính toán
        similarity_df = df[df[available_cols].notna().any(axis=1)].copy()
        
        if len(similarity_df) == 0:
            logger.warning("Không có dữ liệu similarity metrics để phân tích.")
            return metrics
        
        # 1. Metrics tổng thể
        metrics["overall"] = {}
        for col in available_cols:
            if similarity_df[col].notna().any():
                metrics["overall"][col] = {
                    "mean": similarity_df[col].mean(),
                    "median": similarity_df[col].median(),
                    "std": similarity_df[col].std(),
                    "min": similarity_df[col].min(),
                    "max": similarity_df[col].max()
                }
        
        # 2. Metrics theo model
        metrics["by_model"] = {}
        for model in similarity_df['model_name'].unique():
            metrics["by_model"][model] = {}
            model_df = similarity_df[similarity_df['model_name'] == model]
            
            for col in available_cols:
                if model_df[col].notna().any():
                    metrics["by_model"][model][col] = {
                        "mean": model_df[col].mean(),
                        "median": model_df[col].median(),
                        "std": model_df[col].std()
                    }
        
        # 3. Metrics theo prompt type
        metrics["by_prompt_type"] = {}
        for prompt in similarity_df['prompt_type'].unique():
            metrics["by_prompt_type"][prompt] = {}
            prompt_df = similarity_df[similarity_df['prompt_type'] == prompt]
            
            for col in available_cols:
                if prompt_df[col].notna().any():
                    metrics["by_prompt_type"][prompt][col] = {
                        "mean": prompt_df[col].mean(),
                        "median": prompt_df[col].median(),
                        "std": prompt_df[col].std()
                    }
        
        # 4. Metrics theo model và prompt type
        metrics["by_model_prompt"] = {}
        for model in similarity_df['model_name'].unique():
            metrics["by_model_prompt"][model] = {}
            model_df = similarity_df[similarity_df['model_name'] == model]
            
            for prompt in model_df['prompt_type'].unique():
                metrics["by_model_prompt"][model][prompt] = {}
                prompt_df = model_df[model_df['prompt_type'] == prompt]
                
                for col in available_cols:
                    if prompt_df[col].notna().any():
                        metrics["by_model_prompt"][model][prompt][col] = prompt_df[col].mean()
        
        # 5. Tương quan giữa accuracy và các metrics similarity
        if 'is_correct' in similarity_df.columns:
            metrics["correlation"] = {}
            
            for col in available_cols:
                if similarity_df[col].notna().any():
                    valid_rows = similarity_df['is_correct'].notna() & similarity_df[col].notna()
                    if valid_rows.sum() > 5:  # Chỉ tính tương quan nếu có đủ mẫu
                        corr = similarity_df.loc[valid_rows, ['is_correct', col]].corr().iloc[0, 1]
                        metrics["correlation"][f"{col}_vs_accuracy"] = corr
        
        # 6. Tương quan giữa các similarity metrics
        metrics["inter_metric_correlation"] = {}
        for i, col1 in enumerate(available_cols):
            for col2 in available_cols[i+1:]:
                valid_rows = similarity_df[col1].notna() & similarity_df[col2].notna()
                if valid_rows.sum() > 5:  # Chỉ tính tương quan nếu có đủ mẫu
                    corr = similarity_df.loc[valid_rows, [col1, col2]].corr().iloc[0, 1]
                    metrics["inter_metric_correlation"][f"{col1}_vs_{col2}"] = corr
        
        # 7. Thêm thông tin về nhóm điểm cao/thấp
        # Phân nhóm theo embedding_similarity nếu có
        if 'embedding_similarity' in available_cols and similarity_df['embedding_similarity'].notna().any():
            # Tạo phân vị
            similarity_df['similarity_quantile'] = pd.qcut(
                similarity_df['embedding_similarity'], 
                q=4, 
                labels=['Rất thấp', 'Thấp', 'Cao', 'Rất cao']
            )
            
            # Tính accuracy theo từng nhóm phân vị nếu có
            if 'is_correct' in similarity_df.columns:
                accuracy_by_quantile = similarity_df.groupby('similarity_quantile')['is_correct'].mean()
                metrics["accuracy_by_similarity_level"] = accuracy_by_quantile.to_dict()
        
        return metrics

    def _compute_accuracy_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán các metrics liên quan đến độ chính xác của câu trả lời.
        
        Args:
            df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            
        Returns:
            Dict: Các metrics về độ chính xác
        """
        metrics = {}
        
        # Tính accuracy tổng thể
        if 'is_correct' in df.columns:
            metrics['overall_accuracy'] = df['is_correct'].mean()
            
            # Tính accuracy theo model
            metrics['accuracy_by_model'] = df.groupby('model_name')['is_correct'].mean().to_dict()
            
            # Tính accuracy theo prompt type
            metrics['accuracy_by_prompt'] = df.groupby('prompt_type')['is_correct'].mean().to_dict()
            
            # Tính accuracy theo model và prompt type
            metrics['accuracy_by_model_prompt'] = df.groupby(['model_name', 'prompt_type'])['is_correct'].mean().unstack().to_dict('index')
            
            # Tính accuracy theo question type nếu có
            if 'question_type' in df.columns:
                metrics['accuracy_by_question_type'] = df.groupby('question_type')['is_correct'].mean().to_dict()
        
        # Tính các metrics nâng cao cho text generation nếu có cả đáp án chuẩn và response
        if 'correct_answer' in df.columns and 'response' in df.columns:
            try:
                from ..utils.metrics_utils import (
                    calculate_exact_match_accuracy, 
                    calculate_token_overlap,
                    calculate_text_generation_metrics,
                    calculate_rouge_scores,
                    calculate_bleu_scores
                )
                
                # Lọc hàng có cả đáp án và câu trả lời
                valid_rows = ~df['correct_answer'].isna() & ~df['response'].isna()
                valid_df = df[valid_rows]
                
                if len(valid_df) > 0:
                    # Lấy danh sách đáp án và câu trả lời
                    references = valid_df['correct_answer'].tolist()
                    predictions = valid_df['response'].tolist()
                    
                    # Sử dụng hàm calculate_text_generation_metrics để tính toán đầy đủ các metrics
                    text_gen_metrics = calculate_text_generation_metrics(
                        predictions, 
                        references,
                        include_bleu=True,
                        include_rouge=True,
                        include_token_overlap=True,
                        case_sensitive=False,
                        remove_punctuation=True
                    )
                    
                    # Thêm các metrics vào kết quả
                    metrics['text_generation_metrics'] = text_gen_metrics
                    
                    # Tính metrics theo model và prompt type
                    for model in valid_df['model_name'].unique():
                        model_mask = valid_df['model_name'] == model
                        model_predictions = valid_df[model_mask]['response'].tolist()
                        model_references = valid_df[model_mask]['correct_answer'].tolist()
                        
                        if not model_predictions:
                            continue
                            
                        model_metrics = calculate_text_generation_metrics(
                            model_predictions, 
                            model_references,
                            include_bleu=True,
                            include_rouge=True
                        )
                        
                        if 'text_gen_by_model' not in metrics:
                            metrics['text_gen_by_model'] = {}
                        metrics['text_gen_by_model'][model] = model_metrics
                        
                    # Tính metrics theo prompt type
                    for prompt in valid_df['prompt_type'].unique():
                        prompt_mask = valid_df['prompt_type'] == prompt
                        prompt_predictions = valid_df[prompt_mask]['response'].tolist()
                        prompt_references = valid_df[prompt_mask]['correct_answer'].tolist()
                        
                        if not prompt_predictions:
                            continue
                            
                        prompt_metrics = calculate_text_generation_metrics(
                            prompt_predictions, 
                            prompt_references,
                            include_bleu=True,
                            include_rouge=True
                        )
                        
                        if 'text_gen_by_prompt' not in metrics:
                            metrics['text_gen_by_prompt'] = {}
                        metrics['text_gen_by_prompt'][prompt] = prompt_metrics
            except Exception as e:
                logger.warning(f"Lỗi khi tính text generation metrics: {str(e)}")
                logger.debug(traceback.format_exc())
        
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

    def generate_qualitative_examples(self, 
                               results_df: pd.DataFrame, 
                               num_examples: int = 5, 
                               selection_criteria: str = 'interesting'
                              ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Tạo bảng so sánh các ví dụ cụ thể cho phân tích định tính.
        
        Args:
            results_df: DataFrame kết quả đánh giá
            num_examples: Số lượng ví dụ cần lấy
            selection_criteria: Tiêu chí lựa chọn ví dụ
                - 'interesting': Các ví dụ có sự khác biệt lớn giữa các model
                - 'errors': Các ví dụ mà các model đều sai nhiều
                - 'success': Các ví dụ mà các model đều đúng nhiều
                - 'mixed': Các ví dụ có kết quả đúng/sai đan xen giữa các model
            
        Returns:
            Dict[str, List[Dict]]: Dictionary các ví dụ theo từng loại
        """
        if len(results_df) == 0:
            logger.warning("Không có dữ liệu để tạo ví dụ định tính")
            return {}
            
        # Đảm bảo có các cột cần thiết
        required_cols = ['question_id', 'question_text', 'model_name', 'prompt_type', 'is_correct', 'response']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        
        if missing_cols:
            logger.warning(f"Thiếu các cột cần thiết cho phân tích định tính: {missing_cols}")
            return {}
            
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        df = results_df.copy()
        
        # Tạo dictionary kết quả
        examples = {
            'interesting_cases': [],
            'common_errors': [],
            'successful_cases': [],
            'mixed_results': []
        }
        
        try:
            # 1. Phân tích độ chính xác theo từng câu hỏi và model
            accuracy_by_question = df.pivot_table(
                index='question_id',
                columns='model_name',
                values='is_correct',
                aggfunc='mean'
            ).fillna(0)
            
            # 2. Tính toán các case cho từng loại
            
            # 2.1 Interesting cases: Có sự khác biệt lớn giữa các model
            if len(accuracy_by_question.columns) > 1:
                # Tính sự chênh lệch giữa model tốt nhất và tệ nhất
                accuracy_by_question['max_diff'] = accuracy_by_question.max(axis=1) - accuracy_by_question.min(axis=1)
                # Lấy các question có sự chênh lệch lớn nhất
                interesting_questions = accuracy_by_question.nlargest(num_examples, 'max_diff').index.tolist()
                
                # Thêm vào kết quả
                for q_id in interesting_questions:
                    # Lấy thông tin cơ bản của câu hỏi
                    q_info = df[df['question_id'] == q_id].iloc[0]
                    
                    # Lấy câu trả lời từ các model
                    model_responses = {}
                    for model in df['model_name'].unique():
                        model_df = df[(df['question_id'] == q_id) & (df['model_name'] == model)]
                        if len(model_df) > 0:
                            # Lấy response từ prompt_type đầu tiên (hoặc có thể lọc theo prompt_type cụ thể)
                            model_responses[model] = {
                                'response': model_df.iloc[0]['response'],
                                'is_correct': bool(model_df.iloc[0]['is_correct']),
                                'prompt_type': model_df.iloc[0]['prompt_type']
                            }
                    
                    # Tạo ví dụ
                    example = {
                        'question_id': q_id,
                        'question_text': q_info['question_text'],
                        'question_type': q_info.get('question_type', ''),
                        'difficulty': q_info.get('difficulty', ''),
                        'correct_answer': q_info.get('correct_answer', ''),
                        'model_responses': model_responses,
                        'max_diff': accuracy_by_question.loc[q_id, 'max_diff']
                    }
                    
                    examples['interesting_cases'].append(example)
            
            # 2.2 Common errors: Các câu hỏi mà hầu hết model đều trả lời sai
            accuracy_by_question['avg_accuracy'] = accuracy_by_question.mean(axis=1)
            error_questions = accuracy_by_question.nsmallest(num_examples, 'avg_accuracy').index.tolist()
            
            for q_id in error_questions:
                # Lấy thông tin cơ bản của câu hỏi
                q_info = df[df['question_id'] == q_id].iloc[0]
                
                # Lấy câu trả lời từ các model
                model_responses = {}
                for model in df['model_name'].unique():
                    model_df = df[(df['question_id'] == q_id) & (df['model_name'] == model)]
                    if len(model_df) > 0:
                        model_responses[model] = {
                            'response': model_df.iloc[0]['response'],
                            'is_correct': bool(model_df.iloc[0]['is_correct']),
                            'prompt_type': model_df.iloc[0]['prompt_type']
                        }
                
                # Tạo ví dụ
                example = {
                    'question_id': q_id,
                    'question_text': q_info['question_text'],
                    'question_type': q_info.get('question_type', ''),
                    'difficulty': q_info.get('difficulty', ''),
                    'correct_answer': q_info.get('correct_answer', ''),
                    'model_responses': model_responses,
                    'avg_accuracy': accuracy_by_question.loc[q_id, 'avg_accuracy']
                }
                
                examples['common_errors'].append(example)
            
            # 2.3 Successful cases: Các câu hỏi mà hầu hết model đều trả lời đúng
            success_questions = accuracy_by_question.nlargest(num_examples, 'avg_accuracy').index.tolist()
            
            for q_id in success_questions:
                # Lấy thông tin cơ bản của câu hỏi
                q_info = df[df['question_id'] == q_id].iloc[0]
                
                # Lấy câu trả lời từ các model
                model_responses = {}
                for model in df['model_name'].unique():
                    model_df = df[(df['question_id'] == q_id) & (df['model_name'] == model)]
                    if len(model_df) > 0:
                        model_responses[model] = {
                            'response': model_df.iloc[0]['response'],
                            'is_correct': bool(model_df.iloc[0]['is_correct']),
                            'prompt_type': model_df.iloc[0]['prompt_type']
                        }
                
                # Tạo ví dụ
                example = {
                    'question_id': q_id,
                    'question_text': q_info['question_text'],
                    'question_type': q_info.get('question_type', ''),
                    'difficulty': q_info.get('difficulty', ''),
                    'correct_answer': q_info.get('correct_answer', ''),
                    'model_responses': model_responses,
                    'avg_accuracy': accuracy_by_question.loc[q_id, 'avg_accuracy']
                }
                
                examples['successful_cases'].append(example)
            
            # 2.4 Mixed results: Một số model đúng, một số model sai (độ phân tán cao)
            if len(accuracy_by_question.columns) > 1:
                # Tính độ phân tán (standard deviation) giữa các model
                accuracy_by_question['std_dev'] = accuracy_by_question.iloc[:, :-2].std(axis=1)
                # Lấy các question có độ phân tán lớn nhất và accuracy trung bình ~ 0.5
                accuracy_by_question['mixed_score'] = (0.5 - (accuracy_by_question['avg_accuracy'] - 0.5).abs()) * accuracy_by_question['std_dev']
                mixed_questions = accuracy_by_question.nlargest(num_examples, 'mixed_score').index.tolist()
                
                for q_id in mixed_questions:
                    # Lấy thông tin cơ bản của câu hỏi
                    q_info = df[df['question_id'] == q_id].iloc[0]
                    
                    # Lấy câu trả lời từ các model
                    model_responses = {}
                    for model in df['model_name'].unique():
                        model_df = df[(df['question_id'] == q_id) & (df['model_name'] == model)]
                        if len(model_df) > 0:
                            model_responses[model] = {
                                'response': model_df.iloc[0]['response'],
                                'is_correct': bool(model_df.iloc[0]['is_correct']),
                                'prompt_type': model_df.iloc[0]['prompt_type']
                            }
                    
                    # Tạo ví dụ
                    example = {
                        'question_id': q_id,
                        'question_text': q_info['question_text'],
                        'question_type': q_info.get('question_type', ''),
                        'difficulty': q_info.get('difficulty', ''),
                        'correct_answer': q_info.get('correct_answer', ''),
                        'model_responses': model_responses,
                        'std_dev': accuracy_by_question.loc[q_id, 'std_dev'],
                        'avg_accuracy': accuracy_by_question.loc[q_id, 'avg_accuracy']
                    }
                    
                    examples['mixed_results'].append(example)
            
            # Trả về kết quả dựa trên tiêu chí được chọn
            if selection_criteria == 'interesting':
                return {'interesting_cases': examples['interesting_cases']}
            elif selection_criteria == 'errors':
                return {'common_errors': examples['common_errors']}
            elif selection_criteria == 'success':
                return {'successful_cases': examples['successful_cases']}
            elif selection_criteria == 'mixed':
                return {'mixed_results': examples['mixed_results']}
            else:
                return examples
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo ví dụ định tính: {str(e)}")
            traceback.print_exc()
            return {}

    def _compute_f1_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Tính F1-score, precision, recall cho từng cặp model/prompt.
        
        Args:
            df: DataFrame chứa kết quả đánh giá
            
        Returns:
            Dict chứa metrics F1, precision, recall theo model và prompt
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        results = {}
        
        # Tính metrics cho mỗi model
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            # Sử dụng is_correct làm ground truth và tạo prediction là tất cả 1 để so sánh với ground truth
            y_true = model_df['is_correct'].astype(bool)
            y_pred = np.ones_like(y_true)
            
            results[model] = {
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0)
            }
        
        # Tính metrics cho mỗi cặp model/prompt
        for model in df['model_name'].unique():
            model_results = {}
            for prompt in df[df['model_name'] == model]['prompt_type'].unique():
                subset = df[(df['model_name'] == model) & (df['prompt_type'] == prompt)]
                y_true = subset['is_correct'].astype(bool)
                y_pred = np.ones_like(y_true)  # Giả định mọi câu trả lời đều là correct
                
                # Metrics chỉ có ý nghĩa khi có cả true/false
                if y_true.nunique() > 1:
                    model_results[prompt] = {
                        'f1': f1_score(y_true, y_pred, zero_division=0),
                        'precision': precision_score(y_true, y_pred, zero_division=0),
                        'recall': recall_score(y_true, y_pred, zero_division=0)
                    }
                else:
                    # Nếu tất cả các câu trả lời đều đúng hoặc đều sai
                    is_all_correct = y_true.all()
                    model_results[prompt] = {
                        'f1': 1.0 if is_all_correct else 0.0,
                        'precision': 1.0 if is_all_correct else 0.0,
                        'recall': 1.0 if is_all_correct else 0.0
                    }
            
            results[f"{model}_by_prompt"] = model_results
        
        # Tính metrics theo loại câu hỏi nếu có
        if 'question_type' in df.columns:
            question_type_results = {}
            for q_type in df['question_type'].unique():
                subset = df[df['question_type'] == q_type]
                y_true = subset['is_correct'].astype(bool)
                y_pred = np.ones_like(y_true)  # Giả định mọi câu trả lời đều là correct
                
                if len(y_true) > 0:
                    question_type_results[q_type] = {
                        'f1': f1_score(y_true, y_pred, zero_division=0),
                        'precision': precision_score(y_true, y_pred, zero_division=0),
                        'recall': recall_score(y_true, y_pred, zero_division=0)
                    }
            
            results['by_question_type'] = question_type_results
        
        return results

    def calculate_additional_metrics(self) -> pd.DataFrame:
        """
        Tính toán các metrics nâng cao như EM (Exact Match), ROUGE, BLEU, 
        BERT score, METEOR score, F1-score và thêm vào DataFrame kết quả.
        
        Returns:
            pd.DataFrame: DataFrame với các metrics đã được thêm vào
        """
        try:
            # Đảm bảo NLTK resources được tải
            self._ensure_nltk_resources()
            
            # Check for alternative column names
            response_columns = [col for col in self.results_df.columns if col.lower() in (
                'response', 'answer', 'model_answer', 'output', 'generation', 'predicted_answer', 
                'model_response', 'prediction', 'model_output'
            )]
            
            correct_answer_columns = [col for col in self.results_df.columns if col.lower() in (
                'correct_answer', 'expected_answer', 'ground_truth', 'reference', 'true_answer',
                'human_answer', 'gold_answer', 'gold_standard', 'reference_answer'
            )]
            
            # Choose the first available columns
            response_col = response_columns[0] if response_columns else None
            correct_answer_col = correct_answer_columns[0] if correct_answer_columns else None
            
            if not response_col or not correct_answer_col:
                available_cols = ', '.join(self.results_df.columns.tolist())
                logger.warning(f"Không tìm thấy cột 'correct_answer' hoặc 'response' để tính metrics nâng cao. Các cột hiện có: {available_cols}")
                logger.warning(f"Đã tìm thấy các cột response: {response_columns}")
                logger.warning(f"Đã tìm thấy các cột correct_answer: {correct_answer_columns}")
                return self.results_df
                
            logger.info(f"Sử dụng cột '{response_col}' cho câu trả lời và '{correct_answer_col}' cho đáp án chuẩn")
            logger.info(f"Bắt đầu tính toán metrics nâng cao cho {len(self.results_df)} hàng dữ liệu...")
            
            # Khởi tạo các cột metrics mới nếu chưa có
            # F1 score
            if 'f1_score' not in self.results_df.columns:
                self.results_df['f1_score'] = np.nan
                
            # METEOR score
            if 'meteor_score' not in self.results_df.columns:
                self.results_df['meteor_score'] = np.nan
                
            # BERT score
            if 'bert_score' not in self.results_df.columns:
                self.results_df['bert_score'] = np.nan
                
            # Thêm các cột mới: EM, ROUGE, BLEU
            if 'exact_match' not in self.results_df.columns:
                self.results_df['exact_match'] = np.nan
                
            # ROUGE scores (thêm các cột chính)
            for rouge_metric in ['rouge1_f', 'rouge2_f', 'rougeL_f']:
                if rouge_metric not in self.results_df.columns:
                    self.results_df[rouge_metric] = np.nan
                    
            # BLEU scores
            for bleu_metric in ['bleu', 'bleu1', 'bleu2', 'bleu4']:
                if bleu_metric not in self.results_df.columns:
                    self.results_df[bleu_metric] = np.nan
            
            # Lọc các hàng có cả correct_answer và response
            valid_rows = self.results_df[pd.notna(self.results_df[correct_answer_col]) & pd.notna(self.results_df[response_col])]
            logger.info(f"Tìm thấy {len(valid_rows)} hàng có cả '{correct_answer_col}' và '{response_col}'")
            
            count_processed = 0
            count_success = 0
            
            # Tính toán và thêm vào DataFrame
            for idx, row in valid_rows.iterrows():
                count_processed += 1
                try:
                    # Lấy câu trả lời và đáp án chuẩn, đảm bảo là string
                    response = str(row[response_col])
                    correct_answer = str(row[correct_answer_col])
                    
                    # 1. Tính F1 Score dựa trên token overlap
                    try:
                        f1_value = self._calculate_f1_token_overlap(correct_answer, response)
                        if f1_value == 0.0:
                            # Thử phương pháp dự phòng nếu f1_value = 0
                            f1_value = self._calculate_fallback_f1(correct_answer, response)
                        self.results_df.at[idx, 'f1_score'] = f1_value
                    except Exception as f1_err:
                        logger.warning(f"Lỗi khi tính F1 score: {str(f1_err)}")
                        self.results_df.at[idx, 'f1_score'] = 0.0
                    
                    # 2. Tính Exact Match (EM)
                    try:
                        # Tính EM cho 1 cặp cụ thể
                        em_value = calculate_exact_match_accuracy(
                            [response], [correct_answer],
                            normalize=True, 
                            case_sensitive=False, 
                            remove_punctuation=True,
                            remove_whitespace=True,
                            relaxed_match=True
                        )
                        self.results_df.at[idx, 'exact_match'] = em_value
                    except Exception as em_err:
                        logger.warning(f"Lỗi khi tính Exact Match score: {str(em_err)}")
                        self.results_df.at[idx, 'exact_match'] = 0.0

                    # 3. Tính ROUGE Scores
                    try:
                        rouge_scores = calculate_rouge_scores(
                            [response], [correct_answer],
                            rouge_types=['rouge1', 'rouge2', 'rougeL']
                        )
                        
                        # Thêm các điểm ROUGE chính vào DataFrame
                        for metric, value in rouge_scores.items():
                            if metric in ['rouge1_f', 'rouge2_f', 'rougeL_f']:
                                self.results_df.at[idx, metric] = value
                    except Exception as rouge_err:
                        logger.warning(f"Lỗi khi tính ROUGE scores: {str(rouge_err)}")
                        for metric in ['rouge1_f', 'rouge2_f', 'rougeL_f']:
                            self.results_df.at[idx, metric] = 0.0
                    
                    # 4. Tính BLEU Scores
                    try:
                        bleu_scores = calculate_bleu_scores(
                            [response], [correct_answer],
                            max_ngram=4,
                            lowercase=True
                        )
                        
                        # Thêm các điểm BLEU vào DataFrame
                        for metric, value in bleu_scores.items():
                            if metric in ['bleu', 'bleu1', 'bleu2', 'bleu4']:
                                self.results_df.at[idx, metric] = value
                    except Exception as bleu_err:
                        logger.warning(f"Lỗi khi tính BLEU scores: {str(bleu_err)}")
                        for metric in ['bleu', 'bleu1', 'bleu2', 'bleu4']:
                            self.results_df.at[idx, metric] = 0.0
                    
                    # 5. Tính METEOR Score
                    try:
                        from nltk.translate.meteor_score import single_meteor_score
                        # Tokenize texts
                        try:
                            # Thử sử dụng word_tokenize
                            correct_tokens = word_tokenize(correct_answer)
                            response_tokens = word_tokenize(response)
                        except LookupError:
                            # Nếu không thành công, sử dụng split đơn giản
                            logger.debug("Sử dụng phương pháp tokenize đơn giản cho METEOR score")
                            correct_tokens = correct_answer.lower().split()
                            response_tokens = response.lower().split()
                        
                        # Log để debug
                        logger.debug(f"Tokens cho correct_answer: {correct_tokens[:10]}...")
                        logger.debug(f"Tokens cho response: {response_tokens[:10]}...")
                        
                        # Sử dụng single_meteor_score thay vì meteor_score để đơn giản hóa
                        try:
                            meteor_value = single_meteor_score(correct_tokens, response_tokens)
                        except (AttributeError, LookupError) as e:
                            # Fallback nếu single_meteor_score không có sẵn hoặc bị lỗi
                            from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
                            meteor_score = nltk_meteor_score
                            logger.debug(f"Không thể sử dụng single_meteor_score: {str(e)}. Thử sử dụng meteor_score")
                            meteor_value = meteor_score([[correct_tokens]], response_tokens)
                            
                        logger.debug(f"METEOR score cho hàng {idx}: {meteor_value}")
                        self.results_df.at[idx, 'meteor_score'] = meteor_value
                    except Exception as meteor_err:
                        logger.warning(f"Lỗi khi tính METEOR score cho hàng {idx}: {str(meteor_err)}")
                        # Fallback sử dụng một hàm tính tương tự METEOR đơn giản hơn
                        try:
                            # Tính tương tự METEOR bằng cách tính Overlap giữa unigrams
                            from collections import Counter
                            # Không sử dụng word_tokenize để tránh lỗi
                            tokens1 = correct_answer.lower().split()
                            tokens2 = response.lower().split()
                            
                            counter1 = Counter(tokens1)
                            counter2 = Counter(tokens2)
                            
                            # Tính precision và recall dựa trên tần suất xuất hiện của từng từ
                            common = sum((counter1 & counter2).values())
                            if sum(counter2.values()) == 0:
                                precision = 0
                            else:
                                precision = common / sum(counter2.values())
                                
                            if sum(counter1.values()) == 0:
                                recall = 0
                            else:
                                recall = common / sum(counter1.values())
                                
                            # Tính F-score với alpha=0.9 (giống cách tính METEOR)
                            if precision + recall > 0:
                                alpha = 0.9
                                meteor_fallback = precision * recall / (alpha * precision + (1 - alpha) * recall)
                                self.results_df.at[idx, 'meteor_score'] = meteor_fallback
                                logger.debug(f"METEOR fallback score cho hàng {idx}: {meteor_fallback}")
                            else:
                                self.results_df.at[idx, 'meteor_score'] = 0.0
                        except Exception as e:
                            logger.debug(f"Lỗi khi tính METEOR fallback: {str(e)}")
                            self.results_df.at[idx, 'meteor_score'] = 0.0
                    
                    # 6. Tính BERT Score nếu có GPU và thư viện
                    try:
                        # Kiểm tra nếu BERTScorer được import thành công
                        import torch
                        from bert_score import score as bert_score_func
                        has_bert_score = True
                    except ImportError:
                        logger.warning("Thư viện bert-score không được cài đặt, sẽ bỏ qua tính toán BERT score")
                        has_bert_score = False
                    
                    if has_bert_score:
                        try:
                            # Thử sử dụng GPU nếu có
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            # Đặt batch_size thấp hơn nếu sử dụng CPU để tránh OOM
                            batch_size = 4 if device == "cpu" else 8
                            P, R, F1 = bert_score_func(
                                [response], 
                                [correct_answer], 
                                lang="vi" if self.language == "vietnamese" else "en", 
                                device=device,
                                batch_size=batch_size
                            )
                            self.results_df.at[idx, 'bert_score'] = F1.item()
                        except Exception as bert_err:
                            logger.debug(f"Lỗi khi tính BERT score cho hàng {idx}: {str(bert_err)}")
                            self.results_df.at[idx, 'bert_score'] = 0.0
                    
                    count_success += 1
                    
                    # Log thông báo tiến độ
                    if count_processed % 50 == 0:
                        logger.info(f"Đã xử lý {count_processed}/{len(valid_rows)} hàng")
                        
                except Exception as e:
                    logger.debug(f"Lỗi khi tính metrics cho hàng {idx}: {str(e)}")
            
            logger.info(f"Hoàn thành tính toán metrics nâng cao. Thành công: {count_success}/{count_processed}")
            
            # Kiểm tra kết quả và log thông tin
            metrics_summary = {
                'f1_score': {'count': self.results_df['f1_score'].notna().sum(), 'avg': self.results_df['f1_score'].mean()},
                'exact_match': {'count': self.results_df['exact_match'].notna().sum(), 'avg': self.results_df['exact_match'].mean()},
                'rouge1_f': {'count': self.results_df['rouge1_f'].notna().sum(), 'avg': self.results_df['rouge1_f'].mean()},
                'rouge2_f': {'count': self.results_df['rouge2_f'].notna().sum(), 'avg': self.results_df['rouge2_f'].mean()},
                'rougeL_f': {'count': self.results_df['rougeL_f'].notna().sum(), 'avg': self.results_df['rougeL_f'].mean()},
                'bleu': {'count': self.results_df['bleu'].notna().sum(), 'avg': self.results_df['bleu'].mean()},
                'meteor_score': {'count': self.results_df['meteor_score'].notna().sum(), 'avg': self.results_df['meteor_score'].mean()},
                'bert_score': {'count': self.results_df['bert_score'].notna().sum(), 'avg': self.results_df['bert_score'].mean()} if 'bert_score' in self.results_df.columns else {'count': 0, 'avg': 0}
            }
            
            for metric, stats in metrics_summary.items():
                logger.info(f"Metric {metric}: {stats['count']} giá trị, trung bình: {stats['avg']:.4f}")
            
            return self.results_df
        except Exception as e:
            logger.warning(f"Không thể tính metrics nâng cao: {str(e)}")
            logger.debug(traceback.format_exc())
            return self.results_df
    
    def _calculate_f1_token_overlap(self, text1: str, text2: str) -> float:
        """
        Tính toán F1 score dựa trên sự trùng lặp giữa các token, phù hợp với tiếng Việt.
        
        Args:
            text1 (str): Văn bản thứ nhất (đáp án chuẩn)
            text2 (str): Văn bản thứ hai (câu trả lời)
            
        Returns:
            float: F1 score
        """
        try:
            # Chuẩn hóa và tokenize
            text1 = str(text1).lower().strip() if text1 is not None else ""
            text2 = str(text2).lower().strip() if text2 is not None else ""
            
            # Đảm bảo các văn bản không rỗng
            if not text1 or not text2:
                logger.debug("Một trong hai văn bản rỗng")
                return 0.0
            
            try:
                # Thử sử dụng NLTK word_tokenize nếu có thể
                import nltk
                from nltk.tokenize import word_tokenize
                
                tokens1_list = word_tokenize(text1)
                tokens2_list = word_tokenize(text2)
                logger.debug(f"Đã tokenize sử dụng NLTK - tokens1: {len(tokens1_list)}, tokens2: {len(tokens2_list)}")
            except (ImportError, LookupError, AttributeError) as e:
                # Fallback sang phương pháp đơn giản khi không có hoặc lỗi NLTK
                logger.debug(f"Không thể sử dụng NLTK word_tokenize: {str(e)}")
                tokens1_list = text1.split()
                tokens2_list = text2.split()
                logger.debug(f"Đã tokenize sử dụng split() - tokens1: {len(tokens1_list)}, tokens2: {len(tokens2_list)}")
            
            # Kiểm tra xem sau khi tokenize có tìm được tokens nào không
            if not tokens1_list or not tokens2_list:
                logger.debug("Sau khi tokenize, một trong hai danh sách tokens rỗng")
                return 0.0
                
            # Tính F1 bằng Counter để đếm chính xác tần suất của từng token
            try:
                from collections import Counter
                
                # Đếm token 
                counter1 = Counter(tokens1_list)
                counter2 = Counter(tokens2_list)
                
                # Đếm tokens chung dựa trên tần suất
                common_counter = counter1 & counter2
                
                # Tính tổng tần suất
                total1 = sum(counter1.values())
                total2 = sum(counter2.values())
                common_total = sum(common_counter.values())
                
                # Tính precision và recall
                precision = common_total / total2 if total2 > 0 else 0.0
                recall = common_total / total1 if total1 > 0 else 0.0
                
                logger.debug(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Common: {common_total}, Total1: {total1}, Total2: {total2}")
                
                # Tính F1
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    return float(f1)
                else:
                    return 0.0
                    
            except Exception as counter_err:
                # Fallback sang phương pháp đơn giản dựa trên set nếu Counter gặp lỗi
                logger.debug(f"Lỗi khi tính F1 bằng Counter: {str(counter_err)}")
                
                # Sử dụng set để tính overlap
                tokens1_set = set(tokens1_list)
                tokens2_set = set(tokens2_list)
                
                common_tokens = tokens1_set.intersection(tokens2_set)
                
                # Tính precision và recall dựa trên số lượng token unique
                precision = len(common_tokens) / len(tokens2_set) if tokens2_set else 0.0
                recall = len(common_tokens) / len(tokens1_set) if tokens1_set else 0.0
                
                logger.debug(f"Fallback - Precision: {precision:.4f}, Recall: {recall:.4f}")
                
                # Tính F1
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    return float(f1)
                    
                return 0.0
                
        except Exception as e:
            logger.warning(f"Lỗi khi tính F1 token overlap: {str(e)}")
            # Sử dụng phương pháp dự phòng đơn giản nhất
            return self._calculate_fallback_f1(text1, text2)

    def _calculate_fallback_f1(self, text1: str, text2: str) -> float:
        """
        Phương pháp tính F1 dự phòng đơn giản nhất khi các phương pháp khác không hoạt động.
        
        Args:
            text1 (str): Văn bản thứ nhất (đáp án chuẩn)
            text2 (str): Văn bản thứ hai (câu trả lời)
            
        Returns:
            float: F1 score
        """
        try:
            # Chuẩn hóa văn bản một cách đơn giản nhất
            text1 = str(text1).lower().strip() if text1 is not None else ""
            text2 = str(text2).lower().strip() if text2 is not None else ""
            
            # Nếu một trong hai văn bản rỗng
            if not text1 or not text2:
                return 0.0
                
            # Đơn giản hóa văn bản: Chỉ giữ lại chữ cái, số và dấu cách
            import re
            # Loại bỏ các ký tự đặc biệt, chỉ giữ lại khoảng trắng và chữ/số
            text1_clean = re.sub(r'[^\w\s]', ' ', text1)
            text2_clean = re.sub(r'[^\w\s]', ' ', text2)
            
            # Loại bỏ khoảng trắng dư thừa
            text1_clean = re.sub(r'\s+', ' ', text1_clean).strip()
            text2_clean = re.sub(r'\s+', ' ', text2_clean).strip()
            
            # Tokenize đơn giản bằng split()
            tokens1 = text1_clean.split()
            tokens2 = text2_clean.split()
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Có thể sử dụng exact match nếu văn bản rất ngắn
            if len(tokens1) <= 3 and len(tokens2) <= 3:
                return 1.0 if text1_clean == text2_clean else 0.0
                
            # Tính overlap cách 1: F1 dựa trên ký tự
            chars1 = set(text1_clean)
            chars2 = set(text2_clean)
            common_chars = chars1.intersection(chars2)
            char_precision = len(common_chars) / len(chars2) if chars2 else 0
            char_recall = len(common_chars) / len(chars1) if chars1 else 0
            
            # Tính overlap cách 2: F1 dựa trên từ
            words1 = set(tokens1)
            words2 = set(tokens2)
            common_words = words1.intersection(words2)
            word_precision = len(common_words) / len(words2) if words2 else 0
            word_recall = len(common_words) / len(words1) if words1 else 0
            
            # Tính F1 cho cả hai cách
            if char_precision + char_recall > 0:
                char_f1 = 2 * char_precision * char_recall / (char_precision + char_recall)
            else:
                char_f1 = 0.0
                
            if word_precision + word_recall > 0:
                word_f1 = 2 * word_precision * word_recall / (word_precision + word_recall)
            else:
                word_f1 = 0.0
                
            # Sử dụng F1 từ hoặc F1 ký tự, ưu tiên F1 từ
            f1 = word_f1 if word_f1 > 0 else char_f1
            
            # Nếu độ dài quá chênh lệch, giảm điểm F1
            len_ratio = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) > 0 else 0
            adjusted_f1 = f1 * len_ratio
            
            logger.debug(f"Fallback F1 - word: {word_f1:.4f}, char: {char_f1:.4f}, adjusted: {adjusted_f1:.4f}")
            
            return float(adjusted_f1)
        except Exception as e:
            logger.debug(f"Lỗi khi tính fallback F1 (đơn giản): {str(e)}")
            
            # Phương pháp cuối cùng: so sánh đơn giản
            try:
                simple_ratio = sum(1 for c in text1 if c in text2) / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 0
                logger.debug(f"Simple ratio fallback: {simple_ratio:.4f}")
                return float(simple_ratio)
            except:
                return 0.0

    def _ensure_nltk_resources(self) -> None:
        """
        Đảm bảo tất cả các NLTK resources cần thiết đã được tải.
        """
        try:
            import nltk
            
            # Danh sách các resources cần thiết
            required_resources = [
                ('punkt', 'tokenizers/punkt'),
                ('wordnet', 'corpora/wordnet'),
                ('omw-1.4', 'corpora/omw-1.4'),
                ('stopwords', 'corpora/stopwords')
            ]
            
            # Thêm punkt_tab cho tiếng Anh (cần thiết cho tokenization)
            try:
                nltk.data.find('tokenizers/punkt_tab/english/')
                logger.debug("Đã tìm thấy NLTK resource: punkt_tab")
            except LookupError:
                # Các thư mục chuẩn NLTK
                nltk_data_paths = nltk.data.path
                # Lấy thư mục đầu tiên để lưu dữ liệu NLTK
                nltk_data_dir = nltk_data_paths[0] if nltk_data_paths else None
                
                if nltk_data_dir:
                    # Tạo đường dẫn cho punkt_tab
                    import os
                    punkt_tab_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english')
                    os.makedirs(punkt_tab_dir, exist_ok=True)
                    
                    # Sao chép dữ liệu từ punkt sang punkt_tab
                    try:
                        punkt_dir = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
                        if os.path.exists(punkt_dir):
                            import shutil
                            for file in os.listdir(punkt_dir):
                                if file.endswith('.pickle'):
                                    src = os.path.join(punkt_dir, file)
                                    dst = os.path.join(punkt_tab_dir, file)
                                    shutil.copy2(src, dst)
                                    logger.info(f"Đã sao chép {file} từ punkt sang punkt_tab")
                    except Exception as e:
                        logger.warning(f"Không thể sao chép dữ liệu punkt: {str(e)}")
                
                # Nếu không thể tạo punkt_tab, tải punkt bình thường
                logger.info("Tải xuống NLTK resource: punkt thay thế cho punkt_tab")
                nltk.download('punkt', quiet=True)
            
            for resource_name, resource_path in required_resources:
                try:
                    nltk.data.find(resource_path)
                    logger.debug(f"Đã tìm thấy NLTK resource: {resource_name}")
                except LookupError:
                    logger.info(f"Tải xuống NLTK resource: {resource_name}")
                    nltk.download(resource_name, quiet=True)
                    logger.debug(f"Đã tải NLTK resource: {resource_name}")
                    
            # Kiểm tra lại sau khi tải
            missing_resources = []
            for resource_name, resource_path in required_resources:
                try:
                    nltk.data.find(resource_path)
                except LookupError:
                    missing_resources.append(resource_name)
            
            if missing_resources:
                logger.warning(f"Không thể tải các NLTK resources: {', '.join(missing_resources)}")
            else:
                logger.info("Tất cả NLTK resources đã sẵn sàng")
                
        except ImportError:
            logger.warning("Không thể import nltk. Một số tính năng có thể không hoạt động.")
        except Exception as e:
            logger.warning(f"Lỗi khi kiểm tra NLTK resources: {str(e)}")
