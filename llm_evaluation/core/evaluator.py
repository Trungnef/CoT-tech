"""
Class Evaluator điều phối toàn bộ quá trình đánh giá LLM.
Bao gồm việc quản lý vòng lặp đánh giá, xử lý checkpointing,
kết hợp các thành phần khác nhau của hệ thống.
"""

import os
import time
import logging
import pandas as pd
import datetime
from pathlib import Path
import traceback
import json
from tqdm import tqdm
import sys

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(str(Path(__file__).parents[1].absolute()))

# Import các module cần thiết
import config
from core.model_interface import ModelInterface
from core.prompt_builder import create_prompt
from core.checkpoint_manager import CheckpointManager
from core.result_analyzer import ResultAnalyzer
from core.reporting import ReportGenerator
from utils.logging_setup import get_logger, log_evaluation_start, log_evaluation_progress, log_evaluation_complete, log_api_error, log_checkpoint, log_checkpoint_resume, log_section

# Lấy logger cho module này
logger = get_logger("evaluator")

class Evaluator:
    """
    Class Evaluator điều phối quá trình đánh giá các model LLM với nhiều loại prompt.
    """
    
    def __init__(self, models_to_evaluate, prompts_to_evaluate, questions, 
                 results_dir=None, batch_size=5, checkpoint_frequency=5, 
                 use_cache=True, reasoning_evaluation_enabled=True,
                 parallel=False, gpu_ids=None, timestamp=None):
        """
        Khởi tạo đối tượng Evaluator.
        
        Args:
            models_to_evaluate (list): Danh sách các models cần đánh giá
            prompts_to_evaluate (list): Danh sách các loại prompt cần đánh giá
            questions (list): Danh sách các câu hỏi
            results_dir (str): Thư mục để lưu kết quả
            batch_size (int): Kích thước batch để xử lý câu hỏi
            checkpoint_frequency (int): Số câu hỏi giữa mỗi lần lưu checkpoint
            use_cache (bool): Có sử dụng cache model hay không
            reasoning_evaluation_enabled (bool): Có đánh giá khả năng suy luận hay không
            parallel (bool): Có chạy đánh giá song song không
            gpu_ids (list): Danh sách ID GPU để sử dụng
            timestamp (str): Timestamp dùng cho tên file kết quả
        """
        self.models = models_to_evaluate
        self.prompts = prompts_to_evaluate
        self.questions = questions
        self.base_results_dir = results_dir or config.RESULTS_DIR
        self.batch_size = batch_size
        self.checkpoint_frequency = checkpoint_frequency
        self.use_cache = use_cache
        self.reasoning_evaluation_enabled = reasoning_evaluation_enabled
        self.parallel = parallel
        self.gpu_ids = gpu_ids or [0]
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Tạo thư mục kết quả cho lần chạy này dựa trên timestamp
        self.run_dir = os.path.join(self.base_results_dir, f"run_{self.timestamp}")
        self.results_dir = self.run_dir
        
        # Tạo cấu trúc thư mục cho lần chạy hiện tại
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "raw_results"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "analyzed_results"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
        
        # Khởi tạo các thành phần phụ thuộc
        self.model_interface = ModelInterface(use_cache=use_cache)
        
        # Khởi tạo checkpoint manager
        checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir, 
            timestamp=self.timestamp,
            max_checkpoints=config.MAX_CHECKPOINTS
        )
        
        # Các biến trạng thái
        self.results = []  # Danh sách kết quả đánh giá
        self.current_model = None
        self.current_prompt = None
        self.current_question_idx = 0
        self.completed_combinations = set()  # Tập hợp các (model, prompt, question_id) đã hoàn thành
        self.eval_start_time = None
        
        # Tạo đường dẫn file kết quả
        self.results_file = os.path.join(
            self.run_dir, 
            "raw_results", 
            f"evaluation_results_{self.timestamp}.csv"
        )
        
        logger.info(f"Khởi tạo Evaluator với {len(self.models)} models, "
                   f"{len(self.prompts)} prompts, và {len(self.questions)} câu hỏi.")
        logger.info(f"Kết quả sẽ được lưu vào thư mục: {self.run_dir}")
        
    def run_evaluation(self, resume=False):
        """
        Chạy quá trình đánh giá cho tất cả model, prompt, và câu hỏi.
        
        Args:
            resume (bool): Có tiếp tục từ checkpoint gần nhất không
        """
        # Tải checkpoint nếu cần tiếp tục đánh giá
        if resume:
            self._resume_from_checkpoint()
        
        self.eval_start_time = time.time()
        
        # Hiển thị thông tin bắt đầu đánh giá
        log_section(logger, "Bắt đầu đánh giá LLM")
        logger.info(f"Models: {self.models}")
        logger.info(f"Prompts: {self.prompts}")
        logger.info(f"Số câu hỏi: {len(self.questions)}")
        
        if self.parallel:
            self._run_evaluation_parallel()
        else:
            self._run_evaluation_sequential()
        
        # Lưu kết quả cuối cùng
        self._save_results()
        
        # Phân tích kết quả và tạo báo cáo
        self._analyze_and_report()
        
        # Hiển thị tổng kết
        total_time = time.time() - self.eval_start_time
        log_section(logger, "Đánh giá hoàn thành")
        logger.info(f"Thời gian thực hiện: {total_time:.2f} giây")
        logger.info(f"Tổng số kết quả: {len(self.results)}")
        logger.info(f"Kết quả được lưu tại: {self.results_file}")
    
    def _run_evaluation_sequential(self):
        """Thực hiện đánh giá tuần tự cho tất cả tổ hợp model, prompt, câu hỏi."""
        total_combinations = len(self.models) * len(self.prompts) * len(self.questions)
        completed_count = len(self.completed_combinations)
        
        with tqdm(total=total_combinations, initial=completed_count, 
                 desc="Tổng tiến trình") as pbar:
            
            # Vòng lặp qua các model
            for model_name in self.models:
                self.current_model = model_name
                
                # Vòng lặp qua các loại prompt
                for prompt_type in self.prompts:
                    self.current_prompt = prompt_type
                    
                    # Log bắt đầu đánh giá model/prompt
                    log_evaluation_start(logger, model_name, prompt_type, len(self.questions))
                    prompt_start_time = time.time()
                    
                    # Vòng lặp qua các câu hỏi
                    for q_idx, question in enumerate(self.questions):
                        question_id = question.get('id', q_idx)
                        
                        # Kiểm tra xem tổ hợp này đã được đánh giá chưa
                        combination_key = (model_name, prompt_type, question_id)
                        if combination_key in self.completed_combinations:
                            logger.debug(f"Bỏ qua tổ hợp đã đánh giá: {combination_key}")
                            continue
                        
                        # Đánh giá câu hỏi hiện tại
                        try:
                            self.current_question_idx = q_idx
                            question_text = question.get('question', question.get('text', ''))
                            
                            logger.debug(f"Đánh giá câu hỏi {q_idx+1}/{len(self.questions)}: "
                                       f"ID={question_id}, Text={question_text[:50]}...")
                            
                            # Log tiến độ sau mỗi 5 câu hỏi hoặc theo cấu hình checkpoint_frequency
                            if q_idx % max(1, min(5, self.checkpoint_frequency)) == 0:
                                elapsed = time.time() - prompt_start_time
                                log_evaluation_progress(logger, model_name, prompt_type, q_idx, 
                                                      len(self.questions), elapsed)
                            
                            result = self._evaluate_single_combination(
                                model_name, prompt_type, question, q_idx
                            )
                            
                            if result:
                                self.results.append(result)
                                self.completed_combinations.add(combination_key)
                                pbar.update(1)
                            
                            # Lưu checkpoint theo tần suất đã cấu hình
                            if (len(self.results) % self.checkpoint_frequency == 0 and 
                                len(self.results) > 0):
                                self.save_checkpoint()
                                
                        except Exception as e:
                            logger.error(f"Lỗi khi đánh giá tổ hợp {combination_key}: {str(e)}")
                            logger.debug(f"Chi tiết lỗi: {traceback.format_exc()}")
                            
                            # Cố gắng lưu checkpoint ngay cả khi có lỗi
                            self.save_checkpoint()
                    
                    # Log kết thúc đánh giá model/prompt
                    prompt_time = time.time() - prompt_start_time
                    # Tính accuracy nếu có
                    accuracy = None
                    if 'is_correct' in self.results[0] if self.results else False:
                        model_prompt_results = [r for r in self.results 
                                            if r['model_name'] == model_name and r['prompt_type'] == prompt_type]
                        if model_prompt_results:
                            correct_count = sum(1 for r in model_prompt_results if r.get('is_correct', False))
                            accuracy = correct_count / len(model_prompt_results)
                    
                    log_evaluation_complete(logger, model_name, prompt_type, 
                                         len(self.questions), prompt_time, accuracy)
        
        # Lưu kết quả cuối cùng
        self._save_results()
    
    def _run_evaluation_parallel(self):
        """Thực hiện đánh giá song song dựa trên các model khác nhau."""
        # Chức năng song song sẽ được triển khai ở đây
        # Cơ chế có thể sử dụng multiprocessing hoặc concurrent.futures
        logger.error("Tính năng đánh giá song song chưa được triển khai")
        self._run_evaluation_sequential()
    
    def _evaluate_single_combination(self, model_name, prompt_type, question, question_idx):
        """
        Đánh giá một tổ hợp cụ thể (model, prompt, question).
        
        Args:
            model_name (str): Tên của model
            prompt_type (str): Loại prompt
            question (dict): Thông tin câu hỏi
            question_idx (int): Chỉ số của câu hỏi trong danh sách
            
        Returns:
            dict: Kết quả đánh giá hoặc None nếu có lỗi
        """
        start_time = time.time()
        result = {
            'model_name': model_name,
            'prompt_type': prompt_type,
            'question_id': question.get('id', question_idx),
            'question_text': question.get('question', question.get('text', '')),
            'timestamp': datetime.datetime.now().isoformat(),
        }
        
        if 'category' in question:
            result['category'] = question['category']
        if 'difficulty' in question:
            result['difficulty'] = question['difficulty']
        
        expected_answer = question.get('answer', question.get('expected_answer', None))
        if expected_answer:
            result['expected_answer'] = expected_answer
        
        try:
            # Tạo prompt từ câu hỏi và loại prompt
            question_text = question.get('question', question.get('text', ''))
            task_type = question.get('task_type', question.get('type', 'general'))
            custom_examples = question.get('examples', None)
            
            prompt = create_prompt(
                query=question_text,
                prompt_type=prompt_type,
                task_type=task_type,
                custom_examples=custom_examples
            )
            result['prompt'] = prompt
            
            # Lấy cấu hình model cho việc sinh text
            model_config = config.MODEL_CONFIGS.get(model_name, {}).copy()
            
            # Sinh response từ model
            start_generate = time.time()
            response, response_stats = self.model_interface.generate_text(
                model_name=model_name,
                prompt=prompt,
                config=model_config
            )
            end_generate = time.time()
            
            # Kiểm tra lỗi trong response
            if response_stats.get("has_error", False):
                logger.warning(f"Phát hiện lỗi khi sinh response từ {model_name}: {response_stats.get('error_message', '')}")
                result['error'] = response_stats.get('error_message', 'Unknown error')
            
            # Lưu kết quả và thống kê
            result['response'] = response
            result['elapsed_time'] = end_generate - start_generate
            result['token_count'] = response_stats.get('token_count', 0)
            result['tokens_per_second'] = response_stats.get('tokens_per_second', 0)
            
            # Kiểm tra đáp án nếu có
            if expected_answer:
                is_correct = self._check_answer(response, expected_answer, task_type)
                result['is_correct'] = is_correct
                
            # Thông tin thêm nếu có
            if 'correct_answer' in question:
                result['correct_answer'] = question['correct_answer']
            
            logger.debug(f"Đã đánh giá {model_name}/{prompt_type}, câu hỏi {question_idx+1}. "
                       f"Thời gian: {result['elapsed_time']:.2f}s")
            
            return result
            
        except Exception as e:
            question_id = question.get('id', question_idx)
            log_api_error(logger, model_name, e, question_id)
            
            # Thêm thông tin lỗi vào kết quả
            result['error'] = str(e)
            result['elapsed_time'] = time.time() - start_time
            return result
    
    def _save_results(self):
        """Lưu kết quả hiện tại vào file CSV."""
        if not self.results:
            logger.warning("Không có kết quả để lưu.")
            return
        
        try:
            results_df = pd.DataFrame(self.results)
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
            
            # Lưu kết quả
            results_df.to_csv(self.results_file, index=False)
            logger.info(f"Đã lưu {len(self.results)} kết quả vào {self.results_file}")
            
            # Lưu thêm bản JSON nếu cần
            json_file = self.results_file.replace(".csv", ".json")
            results_df.to_json(json_file, orient="records", lines=True)
            logger.info(f"Đã lưu bản sao JSON vào {json_file}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu kết quả: {e}")
            logger.error(traceback.format_exc())
            
            # Cố gắng lưu vào file tạm thời
            try:
                emergency_file = os.path.join(
                    self.run_dir, 
                    "raw_results", 
                    f"emergency_results_{self.timestamp}.json"
                )
                with open(emergency_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, ensure_ascii=False, indent=2)
                logger.info(f"Đã lưu kết quả khẩn cấp vào {emergency_file}")
            except:
                logger.critical("Không thể lưu kết quả khẩn cấp!")
    
    def save_checkpoint(self) -> str:
        """
        Lưu trạng thái hiện tại vào checkpoint.
        
        Returns:
            str: Đường dẫn đến file checkpoint đã lưu
        """
        checkpoint_state = {
            'timestamp': datetime.datetime.now().isoformat(),
            'results': self.results,
            'current_model': self.current_model,
            'current_prompt': self.current_prompt,
            'current_question_idx': self.current_question_idx,
            'completed_combinations': list(self.completed_combinations),
            'total_questions': len(self.questions),
            'models': self.models,
            'prompts': self.prompts,
            'results_file': self.results_file
        }
        
        # Lưu checkpoint sử dụng CheckpointManager
        checkpoint_path = self.checkpoint_manager.save_checkpoint(checkpoint_state)
        
        # Log thông tin checkpoint
        if checkpoint_path:
            log_checkpoint(logger, checkpoint_path, self.current_model, 
                         self.current_prompt, self.current_question_idx, 
                         len(self.questions))
        
        return checkpoint_path
    
    def _resume_from_checkpoint(self):
        """
        Khôi phục trạng thái từ checkpoint gần nhất.
        """
        # Tải checkpoint gần nhất
        checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()
        
        if not checkpoint_data:
            logger.warning("Không tìm thấy checkpoint để khôi phục.")
            return False
        
        try:
            # Khôi phục trạng thái từ checkpoint
            self.results = checkpoint_data.get('results', [])
            self.current_model = checkpoint_data.get('current_model')
            self.current_prompt = checkpoint_data.get('current_prompt')
            self.current_question_idx = checkpoint_data.get('current_question_idx', 0)
            
            # Chuyển đổi completed_combinations từ list sang set
            completed_list = checkpoint_data.get('completed_combinations', [])
            if completed_list:
                # Đảm bảo mỗi mục trong danh sách là tuple
                self.completed_combinations = set(
                    tuple(item) if isinstance(item, list) else item 
                    for item in completed_list
                )
            else:
                self.completed_combinations = set()
            
            # Log thông tin khôi phục
            checkpoint_path = checkpoint_data.get('checkpoint_path', 'unknown')
            log_checkpoint_resume(logger, checkpoint_path, self.current_model, 
                               self.current_prompt, self.current_question_idx, 
                               len(self.questions))
            
            # Log thêm chi tiết
            logger.info(f"Đã khôi phục {len(self.results)} kết quả đánh giá từ checkpoint.")
            logger.info(f"Đã hoàn thành {len(self.completed_combinations)} tổ hợp (model, prompt, câu hỏi).")
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi khôi phục từ checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _analyze_and_report(self):
        """Phân tích kết quả và tạo báo cáo."""
        if not self.results:
            logger.warning("Không có kết quả để phân tích.")
            return
        
        try:
            logger.info("Bắt đầu phân tích kết quả...")
            
            # Chuyển kết quả sang DataFrame
            results_df = pd.DataFrame(self.results)
            
            # Đánh giá reasoning nếu được bật
            if self.reasoning_evaluation_enabled:
                logger.info("Đánh giá khả năng suy luận...")
                analyzer = ResultAnalyzer(
                    results_df=results_df, 
                    reasoning_evaluation_config=config.REASONING_EVALUATION_CONFIG
                )
                results_df = analyzer.analyze()
                
                # Lưu kết quả đã phân tích
                analyzed_file = os.path.join(
                    self.run_dir,
                    "analyzed_results",
                    f"evaluation_results_{self.timestamp}_analyzed.csv"
                )
                results_df.to_csv(analyzed_file, index=False)
                logger.info(f"Đã lưu kết quả phân tích vào {analyzed_file}")
            
            # Tạo báo cáo
            logger.info("Tạo báo cáo...")
            report_generator = ReportGenerator(
                results_df=results_df,
                output_dir=self.run_dir,
                timestamp=self.timestamp
            )
            report_generator.generate_reports()
            
            logger.info("Phân tích và báo cáo hoàn thành.")
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích kết quả: {e}")
            logger.error(traceback.format_exc())
