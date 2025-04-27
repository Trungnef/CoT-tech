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
import re
import multiprocessing as mp
from functools import partial
import torch

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(str(Path(__file__).parents[1].absolute()))

# Import các module cần thiết
import config
from core.model_interface import ModelInterface
from core.prompt_builder import create_prompt
from core.checkpoint_manager import CheckpointManager
from core.result_analyzer import ResultAnalyzer
from core.reporting import Reporting
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
    
    def run_evaluation(self, resume=False, checkpoint_path=None):
        """
        Thực hiện đánh giá trên tất cả các tổ hợp model, prompt, và câu hỏi.
        
        Args:
            resume (bool): Có tiếp tục từ checkpoint gần nhất không
            checkpoint_path (str, optional): Đường dẫn đến checkpoint cụ thể để tải
            
        Returns:
            dict: Kết quả đánh giá (DataFrame) và các metric
        """
        import config as app_config
        
        # Tải checkpoint nếu cần tiếp tục đánh giá
        if resume:
            if checkpoint_path:
                self._resume_from_checkpoint(checkpoint_path)
            else:
                self._resume_from_checkpoint()
        
        # Đánh dấu thời điểm bắt đầu
        self.eval_start_time = time.time()
        
        # Reset trạng thái nếu không resume
        if not resume:
            self.results = []
            self.completed_combinations = set()
            self.current_model = None
            self.current_prompt = None
        
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
        self._analyze_results()
        
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
                    
                    # Xử lý câu hỏi theo batch để tận dụng tài nguyên GPU
                    # Mỗi batch sẽ xử lý self.batch_size câu hỏi liên tiếp
                    # với batch_size=15, mỗi lần chạy sẽ xử lý 15 câu hỏi cùng lúc
                    # Giá trị batch_size=15 là tối ưu cho hệ thống 3 GPU,
                    # cho phép tận dụng bộ nhớ và khả năng xử lý song song
                    question_batches = [self.questions[i:i+self.batch_size] for i in range(0, len(self.questions), self.batch_size)]
                    batch_count = len(question_batches)
                    
                    logger.info(f"Đánh giá {len(self.questions)} câu hỏi theo {batch_count} batch, mỗi batch {self.batch_size} câu hỏi")
                    
                    for batch_idx, batch in enumerate(question_batches):
                        # Tính và hiển thị thời gian còn lại
                        if batch_idx > 0:
                            elapsed_time = time.time() - prompt_start_time
                            avg_time_per_batch = elapsed_time / batch_idx
                            remaining_batches = batch_count - batch_idx
                            remaining_time = avg_time_per_batch * remaining_batches
                            
                            logger.info(f"Batch {batch_idx+1}/{batch_count}, ước tính thời gian còn lại: {remaining_time/60:.1f} phút")
                        
                        # Xử lý batch câu hỏi song song thay vì tuần tự từng câu
                        batch_results = []
                        batch_questions_to_evaluate = []
                        batch_question_ids = []
                        batch_question_idxs = []
                        
                        # Lọc câu hỏi chưa được đánh giá trong batch
                        for q_idx, question in enumerate(batch):
                            question_id = question.get('id', (batch_idx * self.batch_size) + q_idx)
                            combination_key = (model_name, prompt_type, question_id)
                            
                            if combination_key not in self.completed_combinations:
                                batch_questions_to_evaluate.append(question)
                                batch_question_ids.append(question_id)
                                batch_question_idxs.append((batch_idx * self.batch_size) + q_idx)
                        
                        # Nếu có câu hỏi để đánh giá trong batch
                        if batch_questions_to_evaluate:
                            # Log tiến độ
                            elapsed = time.time() - prompt_start_time
                            log_evaluation_progress(logger, model_name, prompt_type, 
                                                 (batch_idx * self.batch_size), 
                                                 len(self.questions), elapsed)
                            
                            # Chuẩn bị danh sách prompts cho tất cả câu hỏi trong batch
                            batch_prompts = []
                            for question in batch_questions_to_evaluate:
                                question_text = question.get('question', question.get('text', ''))
                                prompt = create_prompt(question_text, prompt_type=prompt_type)
                                batch_prompts.append(prompt)
                            
                            logger.info(f"Đang xử lý batch {batch_idx+1}/{batch_count} với {len(batch_prompts)} câu hỏi")
                            
                            try:
                                # Xử lý đồng thời tất cả câu hỏi trong batch nếu là model local (Llama, Qwen)
                                if model_name.lower() in ["llama", "qwen"]:
                                    # Chuẩn bị cấu trúc dữ liệu đúng cho batch processing
                                    batch_prompt_info = []
                                    for i, prompt in enumerate(batch_prompts):
                                        batch_prompt_info.append({
                                            "prompt": prompt,
                                            "prompt_type": prompt_type,
                                            "question_idx": batch_question_idxs[i]
                                        })
                                    
                                    # Xử lý batch cùng lúc với mô hình local
                                    start_time_batch = time.time()
                                    batch_results = self._evaluate_local_model_batch(
                                        model_name, batch_prompt_info, batch_questions_to_evaluate)
                                    batch_latency = time.time() - start_time_batch
                                    
                                    # Thêm kết quả vào kết quả chính
                                    if batch_results:
                                        self.results.extend(batch_results)
                                        
                                        # Cập nhật các combination đã hoàn thành
                                        for result in batch_results:
                                            question_id = result.get('question_id')
                                            self.completed_combinations.add((model_name, prompt_type, question_id))
                                            pbar.update(1)
                                    else:
                                        logger.warning(f"Không có kết quả từ batch processing cho {model_name}")
                                        # Nếu xử lý batch không thành công, thử xử lý tuần tự từng câu
                                        for i, question in enumerate(batch_questions_to_evaluate):
                                            question_id = batch_question_ids[i]
                                            question_idx = batch_question_idxs[i]
                                            
                                            result = self._evaluate_single_combination(
                                                model_name, prompt_type, question, question_idx
                                            )
                                            
                                            if result:
                                                self.results.append(result)
                                                self.completed_combinations.add((model_name, prompt_type, question_id))
                                                pbar.update(1)
                                
                                # Đối với API models hoặc nếu xử lý batch không thành công, xử lý tuần tự từng câu
                                else:
                                    for i, question in enumerate(batch_questions_to_evaluate):
                                        question_id = batch_question_ids[i]
                                        question_idx = batch_question_idxs[i]
                                        
                                        # Đánh giá từng câu hỏi
                                        result = self._evaluate_single_combination(
                                            model_name, prompt_type, question, question_idx
                                        )
                                        
                                        if result:
                                            batch_results.append(result)
                                            self.completed_combinations.add((model_name, prompt_type, question_id))
                                            pbar.update(1)
                            
                            except Exception as e:
                                logger.error(f"Lỗi khi xử lý batch {batch_idx+1}: {str(e)}")
                                logger.debug(f"Chi tiết lỗi: {traceback.format_exc()}")
                                
                                # Nếu xử lý batch lỗi, thử xử lý tuần tự từng câu
                                logger.info("Đang thử xử lý tuần tự cho batch bị lỗi...")
                                for i, question in enumerate(batch_questions_to_evaluate):
                                    try:
                                        question_id = batch_question_ids[i]
                                        question_idx = batch_question_idxs[i]
                                        
                                        result = self._evaluate_single_combination(
                                            model_name, prompt_type, question, question_idx
                                        )
                                        
                                        if result:
                                            batch_results.append(result)
                                            self.completed_combinations.add((model_name, prompt_type, question_id))
                                            pbar.update(1)
                                    except Exception as inner_e:
                                        logger.error(f"Lỗi khi xử lý câu hỏi {question_id}: {str(inner_e)}")
                            
                            # Thêm tất cả kết quả batch vào danh sách kết quả chính
                            self.results.extend(batch_results)
                        
                        # Lưu checkpoint sau mỗi batch
                        if len(self.results) > 0:
                            self.save_checkpoint()
                            logger.info(f"Đã lưu checkpoint sau batch {batch_idx+1}/{batch_count}")
                    
                    # Log kết thúc đánh giá model/prompt
                    prompt_time = time.time() - prompt_start_time
                    # Tính accuracy nếu có
                    accuracy = None
                    if self.results and 'is_correct' in self.results[0]:
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
        """
        Thực hiện đánh giá song song bằng cách chạy nhiều model trên các GPU khác nhau.
        Sử dụng multiprocessing để tận dụng tối đa tài nguyên đa GPU.
        """
        import json  # Thêm import json
        
        # Lấy tổng số GPU để điều phối
        if torch.cuda.is_available():
            total_gpus = torch.cuda.device_count()
            logger.info(f"Đánh giá song song với {total_gpus} GPU")
        else:
            logger.warning("Không phát hiện GPU, sẽ chạy tuần tự")
            return self._run_evaluation_sequential()
        
        # Các model cần đánh giá (lọc đã hoàn thành)
        remaining_models = []
        for model in self.models:
            # Kiểm tra xem model có đã được đánh giá hết chưa
            combinations = [(model, p, q.get('id', i)) 
                            for p in self.prompts 
                            for i, q in enumerate(self.questions)]
            
            # Lọc ra các combination chưa đánh giá
            remaining = [c for c in combinations if c not in self.completed_combinations]
            
            if remaining:
                remaining_models.append(model)
        
        # Nếu không còn model nào cần đánh giá
        if not remaining_models:
            logger.info("Không còn model nào cần đánh giá")
            return
        
        # Xác định số lượng quá trình tối đa nên chạy
        num_processes = min(len(remaining_models), total_gpus)
        
        # Tạo danh sách GPU IDs để chỉ định GPU cụ thể cho mỗi quá trình
        gpu_assignments = {}
        for i, model in enumerate(remaining_models):
            gpu_id = i % total_gpus
            gpu_assignments[model] = gpu_id
            
        logger.info(f"Phân bổ GPU: {gpu_assignments}")
        
        # Hàm để chạy đánh giá cho một model cụ thể
        def evaluate_model(model_name, gpu_id, prompts, questions, completed_combinations):
            # Đặt biến môi trường để giới hạn GPU hiển thị cho quá trình này
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            logger.info(f"Quá trình con cho {model_name} bắt đầu trên GPU {gpu_id}")
            
            # Khởi tạo model interface cho quá trình con
            from core.model_interface import ModelInterface
            model_interface = ModelInterface(use_cache=self.use_cache)
            
            results = []
            
            # Vòng lặp qua các prompt
            for prompt_type in prompts:
                logger.info(f"[GPU {gpu_id}] Đánh giá {model_name} với prompt {prompt_type}")
                
                # Vòng lặp qua các câu hỏi
                for q_idx, question in enumerate(questions):
                    question_id = question.get('id', q_idx)
                    
                    # Kiểm tra xem tổ hợp này đã được đánh giá chưa
                    combination_key = (model_name, prompt_type, question_id)
                    if combination_key in completed_combinations:
                        continue
                        
                    # Tạo prompt
                    from core.prompt_builder import create_prompt
                    prompt = create_prompt(
                        question.get('question', question.get('text', '')),
                        prompt_type=prompt_type
                    )
                    
                    # Lấy câu trả lời
                    start_time = time.time()
                    try:
                        response = model_interface.get_response(
                            model_name=model_name,
                            prompt=prompt,
                            max_tokens=config.MODEL_CONFIGS[model_name]['max_tokens']
                        )
                        latency = time.time() - start_time
                        
                        # Kiểm tra đáp án
                        expected_answer = question.get('solution', question.get('answer', ''))
                        is_correct = self._check_answer(response, expected_answer, question.get('type', 'general'))
                        
                        # Lưu kết quả
                        result = {
                            'model_name': model_name,
                            'prompt_type': prompt_type,
                            'question_id': question_id,
                            'question_text': question.get('question', question.get('text', '')),
                            'question_type': question.get('type', 'general'),
                            'difficulty': question.get('difficulty', 'Không xác định'),
                            'response': response,
                            'expected_answer': expected_answer,
                            'is_correct': is_correct,
                            'latency': latency,
                            'timestamp': datetime.datetime.now().isoformat(),
                            # Khởi tạo các giá trị reasoning riêng biệt
                            'reasoning_accuracy': 0,
                            'reasoning_reasoning': 0,
                            'reasoning_completeness': 0,
                            'reasoning_explanation': 0,
                            'reasoning_cultural_context': 0,
                            'reasoning_average': 0,
                            # Lưu trữ dictionary dưới dạng chuỗi JSON
                            'reasoning_scores_str': json.dumps({
                                'accuracy': 0,
                                'reasoning': 0,
                                'completeness': 0,
                                'explanation': 0,
                                'cultural_context': 0,
                                'average': 0
                            }),
                        }
                        
                        results.append(result)
                        
                        # Log tiến độ
                        if (q_idx + 1) % 10 == 0:
                            logger.info(f"[GPU {gpu_id}] {model_name}/{prompt_type}: Đã đánh giá {q_idx + 1}/{len(questions)} câu hỏi")
                            
                    except Exception as e:
                        logger.error(f"[GPU {gpu_id}] Lỗi khi đánh giá {model_name}/{prompt_type}/{question_id}: {str(e)}")
            
            logger.info(f"[GPU {gpu_id}] Hoàn thành đánh giá model {model_name}, tổng số kết quả: {len(results)}")
            return results
        
        # Tạo pool process
        logger.info(f"Khởi động {num_processes} processes cho đánh giá song song")
        with mp.Pool(processes=num_processes) as pool:
            # Tạo danh sách các model để đánh giá song song
            model_tasks = []
            for model in remaining_models[:num_processes]:  # Chỉ chạy tối đa num_processes models cùng lúc
                gpu_id = gpu_assignments[model]
                # Tạo tác vụ đánh giá
                task = partial(
                    evaluate_model,
                    model,
                    gpu_id,
                    self.prompts,
                    self.questions,
                    self.completed_combinations
                )
                model_tasks.append(task)
            
            # Chạy các tác vụ song song
            results_list = pool.map(lambda f: f(), model_tasks)
            
            # Kết hợp kết quả
            for model_results in results_list:
                if model_results:
                    self.results.extend(model_results)
                    # Cập nhật completed_combinations
                    for result in model_results:
                        combination_key = (result['model_name'], result['prompt_type'], result['question_id'])
                        self.completed_combinations.add(combination_key)
            
            # Lưu kết quả
            self._save_results()
            
        # Xử lý tất cả các model còn lại (nếu có)
        remaining = [m for m in remaining_models if m not in remaining_models[:num_processes]]
        if remaining:
            logger.info(f"Còn {len(remaining)} model cần đánh giá. Tiếp tục với batch mới.")
            self.models = remaining
            self._run_evaluation_parallel()
        
        logger.info("Hoàn thành đánh giá song song tất cả các model")
    
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
        import json  # Thêm import json ở đầu hàm
        import config as app_config  # Import trực tiếp app_config
        
        question_id = question.get('id', question_idx)
        question_text = question.get('question', question.get('text', ''))
        expected_answer = question.get('solution', question.get('answer', ''))
        question_type = question.get('type', 'general')
        difficulty = question.get('difficulty', 'Không xác định')
        
        # Tạo prompt cho câu hỏi
        prompt = create_prompt(
            question_text, 
            prompt_type=prompt_type
        )
        
        # Lấy max_tokens từ cấu hình theo loại prompt
        max_tokens = app_config.get_max_tokens(model_name, prompt_type)
        logger.debug(f"Sử dụng max_tokens={max_tokens} cho {model_name}/{prompt_type}")
        
        # Đặt mô hình trả lời câu hỏi
        start_time = time.time()
        try:
            response = self.model_interface.get_response(
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens
            )
            latency = time.time() - start_time
            
            # Tính toán các metric cơ bản
            is_correct = self._check_answer(response, expected_answer, question_type)
            
            # Khởi tạo các giá trị mặc định cho reasoning scores
            reasoning_accuracy = 0
            reasoning_reasoning = 0
            reasoning_completeness = 0
            reasoning_explanation = 0
            reasoning_cultural_context = 0
            reasoning_average = 0
            
            # Đánh giá khả năng suy luận
            if self.reasoning_evaluation_enabled and app_config.REASONING_EVALUATION_CONFIG.get('enabled', True):
                try:
                    logger.debug(f"Bắt đầu đánh giá reasoning cho {model_name}/{prompt_type}/{question_id}")

                    # Tính heuristic accuracy dự trên kết quả boolean is_correct
                    accuracy_heuristic = 5 if is_correct else 1

                    # Tạo prompt đánh giá với các tiêu chí mới tối ưu cho tiếng Việt
                    reasoning_prompt = self._create_reasoning_evaluation_prompt(question_text, response, expected_answer)

                    # Lấy phản hồi đánh giá từ mô hình giám khảo (Llama3-70B qua Groq API)
                    logger.debug(f"Gửi yêu cầu đánh giá reasoning đến model: {app_config.REASONING_EVALUATION_CONFIG.get('model', 'llama3-70b-8192')}")
                    
                    # Định dạng tên model đúng cho Groq
                    reasoning_model = "groq/" + app_config.REASONING_EVALUATION_CONFIG.get('model', 'llama3-70b-8192')
                    
                    reasoning_eval = self.model_interface.get_response(
                        model_name=reasoning_model,
                        prompt=reasoning_prompt,
                        max_tokens=800  # Tăng max_tokens để đảm bảo nhận được phản hồi đầy đủ
                    )

                    # Parse kết quả đánh giá với các tiêu chí mới
                    reasoning_scores = self._parse_reasoning_evaluation(reasoning_eval)

                    # Trích xuất các giá trị từ dictionary thành các biến riêng biệt
                    reasoning_accuracy = reasoning_scores.get('accuracy', 0)
                    reasoning_reasoning = reasoning_scores.get('reasoning', 0)
                    reasoning_completeness = reasoning_scores.get('completeness', 0)
                    reasoning_explanation = reasoning_scores.get('explanation', 0)
                    reasoning_cultural_context = reasoning_scores.get('cultural_context', 0)
                    reasoning_average = reasoning_scores.get('average', 0)

                    # Đảm bảo có điểm accuracy nếu LLM đánh giá không trả về
                    if reasoning_accuracy == 0:
                        reasoning_accuracy = accuracy_heuristic
                        logger.debug(f"Sử dụng accuracy heuristic: {accuracy_heuristic}/5")
                    
                    # Đánh giá tính nhất quán (consistency) nếu cần
                    if 'consistency' in prompt_type.lower() or prompt_type in ['self_consistency_3', 'self_consistency_5', 'self_consistency_7']:
                        # Sẽ được tính sau trong ReportGenerator
                        logger.debug(f"Prompt type {prompt_type} cần đánh giá consistency, sẽ tính trong ReportGenerator")

                    # Đánh giá phù hợp ngữ cảnh văn hóa nếu sử dụng few-shot hoặc react
                    if any(x in prompt_type.lower() for x in ['few_shot', 'react']):
                        logger.debug(f"Prompt type {prompt_type} được đánh giá cultural_context")

                    # Log kết quả đánh giá reasoning
                    logger.debug(f"Kết quả đánh giá reasoning: accuracy={reasoning_accuracy}, reasoning={reasoning_reasoning}, completeness={reasoning_completeness}, explanation={reasoning_explanation}, cultural_context={reasoning_cultural_context}, average={reasoning_average}")

                except Exception as e:
                    logger.error(f"Lỗi trong quá trình đánh giá reasoning: {str(e)}")
                    logger.debug(traceback.format_exc())
                    # Assign default scores on error
                    reasoning_accuracy = accuracy_heuristic  # Use heuristic accuracy on error
                    reasoning_average = accuracy_heuristic  # Use heuristic average on error
            
            # Ghi log kết quả
            short_response = response[:100] + "..." if len(response) > 100 else response
            logger.debug(f"Kết quả: Model={model_name}, Prompt={prompt_type}, Question={question_id}")
            logger.debug(f"Đúng/Sai: {is_correct}, Thời gian: {latency:.2f}s")
            logger.debug(f"Câu trả lời: {short_response}")
            
            # Tạo đối tượng reasoning_scores để lưu dưới dạng chuỗi JSON
            reasoning_scores_dict = {
                'accuracy': reasoning_accuracy,
                'reasoning': reasoning_reasoning,
                'completeness': reasoning_completeness,
                'explanation': reasoning_explanation,
                'cultural_context': reasoning_cultural_context,
                'average': reasoning_average
            }
            
            # Chuyển đổi dictionary thành chuỗi JSON để lưu
            reasoning_scores_json = json.dumps(reasoning_scores_dict)
            logger.debug(f"Chuỗi JSON reasoning_scores: {reasoning_scores_json}")
            
            # Tạo kết quả với các trường phẳng cho reasoning scores
            result = {
                'model_name': model_name,
                'prompt_type': prompt_type,
                'question_id': question_id,
                'question_text': question_text,
                'question_type': question_type,
                'difficulty': difficulty,
                'response': response,
                'expected_answer': expected_answer,
                'is_correct': is_correct,
                'latency': latency,
                'timestamp': datetime.datetime.now().isoformat(),
                # Phẳng hóa các chỉ số reasoning thành các cột riêng biệt
                'reasoning_accuracy': reasoning_accuracy,
                'reasoning_reasoning': reasoning_reasoning,
                'reasoning_completeness': reasoning_completeness,
                'reasoning_explanation': reasoning_explanation,
                'reasoning_cultural_context': reasoning_cultural_context,
                'reasoning_average': reasoning_average,
                # Lưu trữ dictionary dưới dạng chuỗi JSON hợp lệ
                'reasoning_scores_str': reasoning_scores_json,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá {model_name}/{prompt_type}/{question_id}: {str(e)}")
            logger.debug(f"Chi tiết lỗi: {traceback.format_exc()}")
            return None
    
    def _create_reasoning_evaluation_prompt(self, question, model_answer, correct_answer):
        """
        Tạo prompt để đánh giá chất lượng suy luận theo các tiêu chí tối ưu cho bài toán tiếng Việt.
        """
        return f"""
Là chuyên gia đánh giá chất lượng lời giải toán tiếng Việt, hãy đánh giá chi tiết câu trả lời của mô hình theo 5 tiêu chí sau:

1. Độ chính xác (Accuracy): So sánh kết quả cuối cùng với đáp án chuẩn. Đánh giá cả quá trình giải và kết quả cuối.
   - 5 điểm: Hoàn toàn đúng về kết quả và cách giải
   - 3 điểm: Kết quả đúng nhưng cách giải có sai sót nhỏ
   - 1 điểm: Kết quả sai hoàn toàn

2. Quá trình suy luận (Reasoning): Đánh giá tính logic và hợp lý trong các bước giải.
   - 5 điểm: Trình bày rõ ràng các bước suy luận, công thức và tính toán chính xác
   - 3 điểm: Có các bước suy luận nhưng thiếu một số giải thích quan trọng
   - 1 điểm: Suy luận không hợp lý, thiếu các bước cần thiết

3. Tính đầy đủ (Completeness): Đánh giá việc giải quyết tất cả các yêu cầu của bài toán.
   - 5 điểm: Giải quyết đầy đủ tất cả yêu cầu của bài toán
   - 3 điểm: Có giải quyết các yêu cầu chính nhưng bỏ qua một số yêu cầu phụ
   - 1 điểm: Thiếu nhiều yêu cầu quan trọng của bài toán

4. Khả năng diễn giải (Explanation): Khả năng giải thích rõ ràng bằng tiếng Việt.
   - 5 điểm: Lời giải rõ ràng, dễ hiểu, sử dụng thuật ngữ toán học tiếng Việt chính xác
   - 3 điểm: Giải thích tương đối hiểu được nhưng đôi khi không rõ ràng
   - 1 điểm: Diễn đạt khó hiểu, không mạch lạc

5. Phù hợp ngữ cảnh văn hóa (Cultural Context): Đánh giá mức độ phù hợp với bối cảnh văn hóa Việt Nam.
   - 5 điểm: Lời giải hoàn toàn phù hợp với cách diễn đạt và giải toán trong chương trình giáo dục Việt Nam
   - 3 điểm: Lời giải chấp nhận được nhưng có một số điểm không quen thuộc với học sinh Việt Nam
   - 1 điểm: Lời giải theo phong cách nước ngoài, không phù hợp với cách giải toán ở Việt Nam

BÀI TOÁN:
{question}

ĐÁP ÁN CHUẨN:
{correct_answer}

CÂU TRẢ LỜI CẦN ĐÁNH GIÁ:
{model_answer}

HÃY ĐÁNH GIÁ THEO CÁC TIÊU CHÍ SAU (điểm từ 1-5):
1. Độ chính xác (Accuracy): ?/5
2. Quá trình suy luận (Reasoning): ?/5
3. Tính đầy đủ (Completeness): ?/5
4. Khả năng diễn giải (Explanation): ?/5
5. Phù hợp ngữ cảnh văn hóa (Cultural Context): ?/5

Điểm trung bình: ?/5

Giải thích chi tiết cho từng tiêu chí (nhưng ngắn gọn):
"""

    def _parse_reasoning_evaluation(self, eval_response):
        """
        Phân tích phản hồi đánh giá suy luận và trích xuất điểm số.
        
        Args:
            eval_response (str): Phản hồi từ mô hình đánh giá
            
        Returns:
            dict: Điểm số cho các tiêu chí đánh giá
        """
        # Khởi tạo dict scores mặc định
        scores = {
            'accuracy': 0,
            'reasoning': 0,
            'completeness': 0,
            'explanation': 0,
            'cultural_context': 0,
            'average': 0
        }
        
        # Nếu đầu vào là dict, sử dụng trực tiếp
        if isinstance(eval_response, dict):
            for key in scores.keys():
                if key in eval_response:
                    try:
                        scores[key] = float(eval_response[key])
                    except (ValueError, TypeError):
                        logger.debug(f"Không thể chuyển đổi {key}={eval_response[key]} thành số")
            return scores
        
        # Nếu không phải chuỗi, trả về scores mặc định
        if not isinstance(eval_response, str):
            logger.warning(f"eval_response không phải chuỗi: {type(eval_response)}")
            return scores
            
        try:
            # Kiểm tra xem có nhiều JSON objects bị nối với nhau không
            json_str = eval_response
            
            # Nếu chuỗi chứa nhiều JSON objects
            if json_str.count('{') > 1 and json_str.count('}') > 1:
                # Lấy JSON đầu tiên
                first_open = json_str.find('{')
                first_close = json_str.find('}', first_open) + 1
                if first_open >= 0 and first_close > first_open:
                    json_str = json_str[first_open:first_close]
                    logger.debug(f"Đã tách JSON đầu tiên: {json_str}")
            
            # Thử parse chuỗi JSON
            try:
                data = json.loads(json_str)
                
                # Trích xuất điểm số
                for key in scores.keys():
                    if key in data:
                        try:
                            scores[key] = float(data[key])
                        except (ValueError, TypeError):
                            logger.debug(f"Không thể chuyển đổi {key}={data[key]} thành số")
                
                # Tính điểm trung bình nếu không có
                if scores['average'] == 0:
                    valid_scores = [v for k, v in scores.items() if k != 'average' and v > 0]
                    if valid_scores:
                        scores['average'] = sum(valid_scores) / len(valid_scores)
                        
                return scores
            except json.JSONDecodeError:
                logger.debug(f"Không thể parse JSON: {json_str}")
                # Tiếp tục với regex nếu không parse được JSON
            
            # Các biểu thức chính quy để trích xuất điểm số
            patterns = {
                'accuracy': r'(?:Accuracy|Độ chính xác).*?(\d+)[/\s]*5',
                'reasoning': r'(?:Reasoning|Suy luận|Độ suy luận).*?(\d+)[/\s]*5',
                'completeness': r'(?:Completeness|Tính đầy đủ).*?(\d+)[/\s]*5',
                'explanation': r'(?:Explanation|Giải thích).*?(\d+)[/\s]*5',
                'cultural_context': r'(?:Cultural context|Ngữ cảnh văn hóa).*?(\d+)[/\s]*5'
            }
            
            # Mẫu để trích xuất điểm trung bình
            avg_pattern = r'(?:Average|Trung bình).*?(\d+\.?\d*)[/\s]*5'
            
            # Trích xuất điểm số bằng regex
            import re
            for key, pattern in patterns.items():
                match = re.search(pattern, eval_response, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        scores[key] = float(match.group(1))
                    except (ValueError, TypeError):
                        pass
            
            # Trích xuất điểm trung bình
            avg_match = re.search(avg_pattern, eval_response, re.IGNORECASE | re.DOTALL)
            if avg_match:
                try:
                    scores['average'] = float(avg_match.group(1))
                except (ValueError, TypeError):
                    pass
            else:
                # Tính điểm trung bình nếu không tìm thấy
                valid_scores = [v for k, v in scores.items() if k != 'average' and v > 0]
                if valid_scores:
                    scores['average'] = sum(valid_scores) / len(valid_scores)
            
            # Thêm log để debug
            logger.debug(f"Điểm đánh giá reasoning: {scores}")
                    
        except Exception as e:
            logger.error(f"Lỗi khi phân tích đánh giá suy luận: {str(e)}")
            logger.debug(f"Chi tiết lỗi: {traceback.format_exc()}")
            logger.debug(f"Nội dung phản hồi gốc: {eval_response[:200]}...")
            
        return scores
    
    def _save_results(self):
        """Lưu kết quả hiện tại vào file CSV."""
        if not self.results:
            logger.warning("Không có kết quả để lưu.")
            return
        
        try:
            # Kiểm tra và sửa cột reasoning_scores nếu là chuỗi JSON bị ghép
            for result in self.results:
                if 'reasoning_scores' in result and isinstance(result['reasoning_scores'], str):
                    try:
                        # Kiểm tra xem có nhiều JSON objects bị nối với nhau không
                        json_str = result['reasoning_scores']
                        if json_str.count('{') > 1 and json_str.count('}') > 1:
                            # Lấy JSON đầu tiên
                            first_open = json_str.find('{')
                            first_close = json_str.find('}', first_open) + 1
                            if first_open >= 0 and first_close > first_open:
                                result['reasoning_scores'] = json_str[first_open:first_close]
                                logger.debug(f"Đã sửa JSON bị trùng lặp: {result['reasoning_scores']}")
                    except Exception as e:
                        logger.error(f"Lỗi khi sửa JSON: {str(e)}")
                
                # Chuyển đổi reasoning_scores thành các trường riêng biệt
                if 'reasoning_scores' in result:
                    try:
                        scores = result['reasoning_scores']
                        # Nếu là chuỗi, thử parse thành dict
                        if isinstance(scores, str):
                            try:
                                scores = json.loads(scores)
                            except:
                                # Nếu không thể parse, giữ nguyên
                                pass
                        
                        # Nếu là dict, trích xuất các điểm riêng
                        if isinstance(scores, dict):
                            for key, value in scores.items():
                                if key in ['accuracy', 'reasoning', 'completeness', 'explanation', 'cultural_context', 'average']:
                                    result[f'reasoning_{key}'] = float(value)
                            
                            # Sau khi trích xuất, chuyển reasoning_scores thành chuỗi
                            result['reasoning_scores'] = json.dumps(scores)
                    except Exception as e:
                        logger.error(f"Lỗi khi chuyển đổi reasoning_scores: {str(e)}")
            
            results_df = pd.DataFrame(self.results)
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
            
            # Đảm bảo các cột reasoning_ là số
            for col in results_df.columns:
                if col.startswith('reasoning_') and col not in ['reasoning_scores', 'reasoning_evaluation']:
                    try:
                        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Lỗi khi chuyển đổi cột {col} thành số: {str(e)}")
            
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
    
    def _resume_from_checkpoint(self, checkpoint_path=None):
        """
        Khôi phục trạng thái từ checkpoint.
        
        Args:
            checkpoint_path (str, optional): Đường dẫn đến checkpoint cụ thể cần tải.
                                             Nếu None, sẽ tìm và tải checkpoint gần nhất.
        
        Returns:
            bool: True nếu khôi phục thành công, False nếu không
        """
        # Xác định checkpoint cần tải
        checkpoint_data = None
        
        if checkpoint_path:
            # Tải checkpoint cụ thể nếu được chỉ định
            if os.path.exists(checkpoint_path):
                logger.info(f"Tải checkpoint từ đường dẫn cụ thể: {checkpoint_path}")
                checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            else:
                logger.error(f"Không tìm thấy checkpoint tại: {checkpoint_path}")
                return False
        else:
            # Tìm và tải checkpoint mới nhất
            logger.info("Tìm kiếm checkpoint mới nhất...")
            checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()
            
        if not checkpoint_data:
            logger.warning("Không tìm thấy checkpoint nào để khôi phục.")
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
            cp_path = checkpoint_data.get('checkpoint_path', checkpoint_path or 'unknown')
            log_checkpoint_resume(logger, cp_path, self.current_model, 
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
    
    def _analyze_results(self):
        """Phân tích kết quả và tạo báo cáo."""
        if not self.results:
            logger.warning("Không có kết quả để phân tích")
            return
        
        logger.info("Bắt đầu phân tích kết quả và tạo báo cáo...")
        start_time = time.time()
        
        try:
            # Chuyển đổi kết quả thành DataFrame
            results_df = pd.DataFrame(self.results)
            
            # Log cấu trúc DataFrame trước khi phân tích
            logger.debug(f"Cấu trúc DataFrame trước khi phân tích: {results_df.columns.tolist()}")
            logger.debug(f"Số lượng dòng: {len(results_df)}")
            logger.debug(f"Các cột reasoning: {[col for col in results_df.columns if 'reasoning' in col]}")
            
            # Tạo cột số mới từ chuỗi scores để tránh lỗi chuyển đổi
            reasoning_numeric_cols = []
            
            # Sửa cột reasoning_scores trước khi phân tích
            if 'reasoning_scores' in results_df.columns:
                logger.debug("Kiểm tra và sửa cột reasoning_scores...")
                
                def fix_json_string(json_str):
                    """Sửa chuỗi JSON bị ghép."""
                    if not isinstance(json_str, str):
                        return json_str
                    
                    try:
                        # Nếu đã là dict, không cần xử lý
                        if isinstance(json_str, dict):
                            return json_str
                            
                        # Nếu là chuỗi có nhiều JSON objects
                        if json_str.count('{') > 1 and json_str.count('}') > 1:
                            first_open = json_str.find('{')
                            first_close = json_str.find('}', first_open) + 1
                            if first_open >= 0 and first_close > first_open:
                                clean_json = json_str[first_open:first_close]
                                logger.debug(f"Đã sửa JSON bị ghép: {clean_json}")
                                # Thử chuyển đổi thành dict
                                try:
                                    return json.loads(clean_json)
                                except:
                                    return clean_json
                        else:
                            # Nếu là một JSON object duy nhất, thử parse
                            try:
                                return json.loads(json_str)
                            except:
                                return json_str
                    except Exception as e:
                        logger.error(f"Lỗi khi sửa JSON: {str(e)}")
                        return json_str
                
                # Áp dụng hàm sửa cho cột reasoning_scores
                results_df['reasoning_scores'] = results_df['reasoning_scores'].apply(fix_json_string)
                logger.debug("Đã sửa xong cột reasoning_scores")
                
                # Chuyển đổi điểm từ dạng dict hoặc chuỗi thành các cột số
                def extract_score(row, key):
                    try:
                        scores = row['reasoning_scores']
                        if isinstance(scores, dict) and key in scores:
                            return float(scores[key])
                        elif isinstance(scores, str):
                            # Thử phân tích chuỗi JSON
                            try:
                                json_data = json.loads(scores)
                                if key in json_data:
                                    return float(json_data[key])
                            except:
                                pass
                        return None
                    except Exception as e:
                        logger.debug(f"Lỗi khi trích xuất điểm {key}: {str(e)}")
                        return None
                
                # Thêm các cột điểm riêng biệt
                for key in ['accuracy', 'reasoning', 'completeness', 'explanation', 'cultural_context', 'average']:
                    col_name = f'reasoning_{key}'
                    results_df[col_name] = results_df.apply(lambda row: extract_score(row, key), axis=1)
                    reasoning_numeric_cols.append(col_name)
                
                logger.debug(f"Đã tạo các cột số cho điểm đánh giá: {reasoning_numeric_cols}")
            
            # Xử lý cột reasoning_scores để tránh lỗi khi tính toán
            if 'reasoning_scores' in results_df.columns:
                # Chỉ lưu trữ cột này cho mục đích tương thích, không sử dụng trong phân tích
                # Tạo bản sao của cột để lưu trữ
                reasoning_scores_str = results_df['reasoning_scores'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
                # Loại bỏ cột dictionary gốc để tránh lỗi khi tính toán
                results_df = results_df.drop(columns=['reasoning_scores'])
                # Thêm lại cột dưới dạng chuỗi để lưu trữ thông tin mà không gây lỗi khi tính toán
                results_df['reasoning_scores_str'] = reasoning_scores_str
                logger.debug("Đã chuyển đổi cột reasoning_scores thành chuỗi để tránh lỗi")
            
            # Xử lý cột reasoning_scores_str nếu nó chứa nhiều JSON objects ghép lại
            if 'reasoning_scores_str' in results_df.columns:
                logger.debug("Kiểm tra và sửa cột reasoning_scores_str nếu có nhiều JSON objects ghép lại...")
                
                def fix_merged_json_local(json_str):
                    """Sửa chuỗi JSON bị ghép."""
                    if not isinstance(json_str, str):
                        return json_str
                    
                    # Chỉ xử lý nếu có nhiều dấu ngoặc
                    if json_str.count('{') > 1 and json_str.count('}') > 1:
                        try:
                            first_open = json_str.find('{')
                            first_close = json_str.find('}', first_open) + 1
                            if first_open >= 0 and first_close > first_open:
                                return json_str[first_open:first_close]
                        except Exception as e:
                            logger.error(f"Lỗi khi sửa JSON: {e}")
                    return json_str
                
                try:
                    # Áp dụng hàm sửa lỗi
                    results_df['reasoning_scores_str'] = results_df['reasoning_scores_str'].apply(fix_merged_json_local)
                    logger.debug("Đã sửa các chuỗi JSON bị ghép trong cột reasoning_scores_str")
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý cột reasoning_scores_str: {e}")
                    logger.error(traceback.format_exc())
            
            # Khởi tạo ResultAnalyzer với cấu hình thích hợp
            logger.debug("Khởi tạo ResultAnalyzer...")
            analyzer = ResultAnalyzer(
                results_df=results_df,
                reasoning_evaluation_config=config.REASONING_EVALUATION_CONFIG,
                reasoning_model=config.REASONING_EVALUATION_CONFIG.get("model", "groq/llama3-70b-8192"),
                language="vietnamese"
            )
            
            # Thực hiện phân tích
            logger.debug("Bắt đầu phân tích dữ liệu...")
            try:
                analyzed_df = analyzer.analyze()
                logger.debug(f"Phân tích thành công, DataFrame kết quả có {len(analyzed_df)} dòng và {len(analyzed_df.columns)} cột")
            except Exception as analyze_error:
                logger.error(f"Lỗi trong quá trình phân tích: {str(analyze_error)}")
                logger.debug(traceback.format_exc())
                # Kiểm tra xem có phải lỗi dict + dict không
                if "unsupported operand type(s) for +" in str(analyze_error) and "dict" in str(analyze_error):
                    logger.error("Phát hiện lỗi cộng dictionary! Đang kiểm tra chi tiết...")
                    # Log thêm thông tin để debug
                    for col in results_df.columns:
                        if isinstance(results_df[col].iloc[0], dict):
                            logger.error(f"Cột {col} chứa dictionary: {results_df[col].iloc[0]}")
                raise
            
            # Lưu dữ liệu đã phân tích
            analyzed_file = os.path.join(
                self.run_dir, 
                "analyzed_results", 
                f"analyzed_results_{self.timestamp}.csv"
            )
            analyzed_df.to_csv(analyzed_file, index=False)
            
            logger.info(f"Đã lưu kết quả phân tích vào: {analyzed_file}")
            
            # Tạo báo cáo với các biểu đồ và visualizations
            logger.debug("Khởi tạo Reporting module...")
            report_generator = Reporting(
                results_df=analyzed_df,
                output_dir=self.run_dir,
                timestamp=self.timestamp,
                language="vietnamese"
            )
            
            # Tạo báo cáo
            logger.debug("Bắt đầu tạo báo cáo...")
            report_paths = report_generator.generate_reports()
            
            if isinstance(report_paths, dict) and 'html' in report_paths:
                logger.info(f"Đã tạo báo cáo tại: {report_paths['html']}")
            else:
                logger.info("Đã tạo báo cáo thành công")
            
            # In vài thông tin cơ bản về kết quả
            correct_count = len(analyzed_df[analyzed_df['is_correct'] == True])
            total_count = len(analyzed_df)
            overall_accuracy = correct_count / total_count if total_count > 0 else 0
            
            logger.info(f"Kết quả tổng quan: {correct_count}/{total_count} câu đúng (accuracy: {overall_accuracy:.2%})")
            
            # Tính accuracy theo từng model
            for model_name in set(analyzed_df['model_name']):
                model_df = analyzed_df[analyzed_df['model_name'] == model_name]
                model_correct = len(model_df[model_df['is_correct'] == True])
                model_total = len(model_df)
                model_accuracy = model_correct / model_total if model_total > 0 else 0
                
                logger.info(f"Model {model_name}: {model_correct}/{model_total} câu đúng (accuracy: {model_accuracy:.2%})")
            
            # Tính accuracy theo từng loại prompt
            for prompt_type in set(analyzed_df['prompt_type']):
                prompt_df = analyzed_df[analyzed_df['prompt_type'] == prompt_type]
                prompt_correct = len(prompt_df[prompt_df['is_correct'] == True])
                prompt_total = len(prompt_df)
                prompt_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
                
                logger.info(f"Prompt {prompt_type}: {prompt_correct}/{prompt_total} câu đúng (accuracy: {prompt_accuracy:.2%})")
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích kết quả: {str(e)}")
            logger.debug(f"Chi tiết lỗi: {traceback.format_exc()}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Đã hoàn thành phân tích và tạo báo cáo sau {elapsed_time:.2f} giây")
    
    def _check_answer(self, response, expected_answer, task_type="general"):
        """
        Kiểm tra xem câu trả lời có đúng không dựa trên đáp án mong đợi.
        Đã được tối ưu cho các loại bài toán tiếng Việt phổ biến.
        
        Args:
            response (str): Câu trả lời từ model
            expected_answer (str): Đáp án mong đợi
            task_type (str): Loại nhiệm vụ (Bài toán công việc, Bài toán chia kẹo, etc.)
            
        Returns:
            bool: True nếu câu trả lời đúng, False nếu sai
        """
        import re
        
        if not response or not expected_answer:
            return False
            
        # Chuẩn hóa câu trả lời để so sánh
        response = response.strip().lower()
        expected_answer = str(expected_answer).strip().lower()
        
        # Thêm log cho debug
        logger.debug(f"Kiểm tra câu trả lời loại bài toán: {task_type}")
        logger.debug(f"Câu trả lời: {response[:200]}...")
        logger.debug(f"Đáp án mong đợi: {expected_answer[:200]}...")
        
        # Trích xuất tất cả các số từ câu trả lời và đáp án (cải tiến regex để bắt cả các dạng phân số)
        response_numbers = re.findall(r'-?\d+[\.,]?\d*', response)
        expected_numbers = re.findall(r'-?\d+[\.,]?\d*', expected_answer)
        
        # Thêm tìm phân số
        response_fractions = re.findall(r'(\d+)/(\d+)', response)
        expected_fractions = re.findall(r'(\d+)/(\d+)', expected_answer)
        
        # Chuyển đổi các số từ dạng chuỗi sang float để so sánh
        response_nums_float = []
        for num in response_numbers:
            try:
                # Xử lý cả trường hợp dấu phẩy và dấu chấm
                num = num.replace(',', '.')
                response_nums_float.append(float(num))
            except ValueError:
                continue
                
        expected_nums_float = []
        for num in expected_numbers:
            try:
                # Xử lý cả trường hợp dấu phẩy và dấu chấm
                num = num.replace(',', '.')
                expected_nums_float.append(float(num))
            except ValueError:
                continue
                
        # Trường hợp phổ biến: Phản hồi chứa đáp án số chính xác
        # So sánh các số trong response với expected với độ lệch nhỏ hơn 2%
        for resp_num in response_nums_float:
            for exp_num in expected_nums_float:
                if self._is_number_equal(resp_num, exp_num, tolerance=0.02):
                    logger.debug(f"Tìm thấy số khớp trực tiếp: {resp_num} ~ {exp_num}")
                    return True
        
        # 1. Xử lý các loại bài toán cụ thể
        
        # Bài toán công việc - thường có đáp án là giờ hoặc phút
        if "công việc" in task_type.lower():
            # Kiểm tra có số giờ/phút trong đáp án không
            time_patterns = [
                r'(\d+[\.,]?\d*)\s*giờ',
                r'(\d+[\.,]?\d*)\s*phút',
                r'(\d+[\.,]?\d*)\s*giây',
                r'(\d+[\.,]?\d*)\s*h',
                r'(\d+[\.,]?\d*)\s*min',
                r'(\d+[\.,]?\d*)\s*s',
                r't\s*=\s*(\d+[\.,]?\d*)',
                r'thời gian\s*=\s*(\d+[\.,]?\d*)',
                r'thời gian là\s*(\d+[\.,]?\d*)',
            ]
            
            for pattern in time_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        exp_num = float(exp_match.replace(',', '.'))
                        for resp_match in response_matches:
                            resp_num = float(resp_match.replace(',', '.'))
                            if self._is_number_equal(exp_num, resp_num, tolerance=0.02):
                                logger.debug(f"Bài toán công việc: Số giờ/phút khớp: {resp_num} ~ {exp_num}")
                                return True
        
        # Bài toán chia kẹo - thường có đáp án là x viên kẹo
        elif "chia kẹo" in task_type.lower() or "viên kẹo" in expected_answer:
            # Kiểm tra xem có số viên trong câu trả lời không
            candy_patterns = [
                r'(\d+)\s*viên',
                r'x\s*=\s*(\d+)',
                r'được\s*(\d+)\s*viên',
                r'mỗi\s*người\s*\w*\s*(\d+)',
                r'người\s*thứ\s*nhất\s*\w*\s*(\d+)',
            ]
            
            for pattern in candy_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                # So sánh kết quả tìm được
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        exp_num = float(exp_match)
                        for resp_match in response_matches:
                            resp_num = float(resp_match)
                            if self._is_number_equal(resp_num, exp_num, tolerance=0.02):
                                logger.debug(f"Bài toán chia kẹo: Số viên khớp: {resp_num} ~ {exp_num}")
                                return True
                
                # Nếu không tìm thấy từ khóa, thử dùng số trực tiếp
                if not expected_matches and expected_nums_float:
                    # Sắp xếp số theo thứ tự giảm dần vì thường số lớn nhất là kết quả cuối cùng
                    expected_nums_float.sort(reverse=True)
                    
                    # Nếu tìm thấy số trong câu trả lời
                    if response_matches:
                        for resp_match in response_matches:
                            resp_num = float(resp_match)
                            if any(self._is_number_equal(resp_num, exp_num, tolerance=0.02) for exp_num in expected_nums_float):
                                logger.debug(f"Bài toán chia kẹo: Số viên khớp")
                                return True
                    elif response_nums_float:
                        for resp_num in response_nums_float:
                            if any(self._is_number_equal(resp_num, exp_num, tolerance=0.02) for exp_num in expected_nums_float):
                                logger.debug(f"Bài toán chia kẹo: Số khớp (từ số thuần túy)")
                                return True
        
        # Bài toán hỗn hợp - thường liên quan đến grams, percentages
        elif "hỗn hợp" in task_type.lower() or "dung dịch" in expected_answer:
            # Tìm số gram cần thêm
            gram_patterns = [
                r'(\d+[\.,]?\d*)\s*g',
                r'(\d+[\.,]?\d*)\s*gram',
                r'x\s*=\s*(\d+[\.,]?\d*)',
                r'cần thêm\s*(\d+[\.,]?\d*)',
                r'phải thêm\s*(\d+[\.,]?\d*)',
            ]
            
            for pattern in gram_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                # So sánh các số tìm được
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        exp_num = float(exp_match.replace(',', '.'))
                        for resp_match in response_matches:
                            resp_num = float(resp_match.replace(',', '.'))
                            if self._is_number_equal(exp_num, resp_num, tolerance=0.02):
                                logger.debug(f"Bài toán hỗn hợp: Số gram khớp: {resp_num} ~ {exp_num}")
                                return True
            
            # Nếu không tìm được qua pattern, thử dùng số trực tiếp
            if expected_nums_float and response_nums_float:
                # Sắp xếp số theo thứ tự giảm dần
                expected_nums_float.sort(reverse=True)
                response_nums_float.sort(reverse=True)
                
                for resp_num in response_nums_float:
                    if any(self._is_number_equal(resp_num, exp_num, tolerance=0.02) for exp_num in expected_nums_float):
                        logger.debug(f"Bài toán hỗn hợp: Số khớp (từ số thuần túy)")
                        return True
        
        # Bài toán số học - tìm số bài toán
        elif "số học" in task_type.lower() or "tìm số" in expected_answer:
            # Tìm số trong câu trả lời
            number_patterns = [
                r'số cần tìm là:?\s*(\d+)',
                r'số cần tìm:?\s*(\d+)',
                r'kết quả là:?\s*(\d+)',
                r'vậy số là:?\s*(\d+)',
                r'số đó là:?\s*(\d+)',
                r'x\s*=\s*(\d+)'
            ]
            
            # Nếu đáp án nói "không có số thỏa mãn"
            if "không có số" in expected_answer or "không tồn tại" in expected_answer:
                if "không có số" in response or "không tồn tại" in response or "không có" in response:
                    logger.debug(f"Bài toán số học: Đáp án 'không có số thỏa mãn' khớp")
                    return True
                return False
        
            # Tìm số cụ thể
            for pattern in number_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                # So sánh các số tìm được
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        for resp_match in response_matches:
                            if exp_match == resp_match:
                                logger.debug(f"Bài toán số học: Số khớp: {resp_match} == {exp_match}")
            return True
            
            # Nếu không tìm được qua pattern, thử dùng số trực tiếp
            if expected_nums_float and response_nums_float:
                # Lấy số cuối cùng trong expected_answer vì thường là kết quả
                last_expected_num = expected_nums_float[-1] if expected_nums_float else None
                
                if last_expected_num:
                    for resp_num in response_nums_float:
                        if self._is_number_equal(resp_num, last_expected_num, tolerance=0.01):
                            logger.debug(f"Bài toán số học: Số khớp (từ số thuần túy)")
                            return True
        
        # Bài toán hình học - diện tích và chu vi
        elif "hình học" in task_type.lower() or ("diện tích" in expected_answer and "chu vi" in expected_answer):
            # Tìm diện tích và chu vi
            area_patterns = [
                r'diện tích:?\s*(\d+[\.,]?\d*)',
                r'diện tích là:?\s*(\d+[\.,]?\d*)',
                r'diện tích =\s*(\d+[\.,]?\d*)',
                r'diện tích\s*=\s*(\d+[\.,]?\d*)',
                r'(\d+[\.,]?\d*)\s*cm²',
                r'(\d+[\.,]?\d*)\s*m²',
                r'S\s*=\s*(\d+[\.,]?\d*)'
            ]
            
            perimeter_patterns = [
                r'chu vi:?\s*(\d+[\.,]?\d*)',
                r'chu vi là:?\s*(\d+[\.,]?\d*)',
                r'chu vi =\s*(\d+[\.,]?\d*)',
                r'chu vi\s*=\s*(\d+[\.,]?\d*)',
                r'P\s*=\s*(\d+[\.,]?\d*)'
            ]
            
            # Kiểm tra diện tích
            area_found = False
            for pattern in area_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        exp_num = float(exp_match.replace(',', '.'))
                        for resp_match in response_matches:
                            resp_num = float(resp_match.replace(',', '.'))
                            if self._is_number_equal(exp_num, resp_num, tolerance=0.02):
                                logger.debug(f"Bài toán hình học: Diện tích khớp: {resp_num} ~ {exp_num}")
                                area_found = True
            
            # Kiểm tra chu vi
            perimeter_found = False
            for pattern in perimeter_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        exp_num = float(exp_match.replace(',', '.'))
                        for resp_match in response_matches:
                            resp_num = float(resp_match.replace(',', '.'))
                            if self._is_number_equal(exp_num, resp_num, tolerance=0.02):
                                logger.debug(f"Bài toán hình học: Chu vi khớp: {resp_num} ~ {exp_num}")
                                perimeter_found = True
            
            # Nếu bài toán yêu cầu tính cả diện tích và chu vi, cả hai phải đúng
            if "diện tích và chu vi" in expected_answer.lower():
                return area_found and perimeter_found
            # Nếu chỉ yêu cầu một trong hai, hoặc không rõ ràng, chỉ cần một cái đúng
            elif area_found or perimeter_found:
                return True
                
            # Thử tìm các số trong câu trả lời
            if expected_nums_float and len(expected_nums_float) >= 2 and response_nums_float:
                matches = 0
                for exp_num in expected_nums_float:
                    for resp_num in response_nums_float:
                        if self._is_number_equal(exp_num, resp_num, tolerance=0.02):
                            matches += 1
                            break
                
                # Nếu khớp cả hai số
                if matches >= 2:
                    logger.debug(f"Bài toán hình học: Khớp {matches} số")
                    return True
        
        # Bài toán chuyển động - vận tốc, thời gian
        elif "chuyển động" in task_type.lower() or "vận tốc" in expected_answer or "hai xe" in expected_answer.lower():
            # Tìm thời gian gặp nhau
            time_patterns = [
                r'sau\s*(\d+[\.,]?\d*)\s*giờ',
                r't\s*=\s*(\d+[\.,]?\d*)',
                r'thời gian\s*=\s*(\d+[\.,]?\d*)',
                r'thời gian gặp nhau\s*=\s*(\d+[\.,]?\d*)',
                r'thời gian là\s*(\d+[\.,]?\d*)',
                r'gặp nhau sau\s*(\d+[\.,]?\d*)',
            ]
            
            for pattern in time_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        exp_num = float(exp_match.replace(',', '.'))
                        for resp_match in response_matches:
                            resp_num = float(resp_match.replace(',', '.'))
                            if self._is_number_equal(exp_num, resp_num, tolerance=0.02):
                                logger.debug(f"Bài toán chuyển động: Thời gian khớp: {resp_num} ~ {exp_num}")
                return True
                
            # Nếu không tìm được qua pattern, thử dùng số trực tiếp
            if expected_nums_float and response_nums_float:
                # Lấy số cuối cùng trong expected_answer vì thường là kết quả
                last_expected_num = expected_nums_float[-1] if expected_nums_float else None
                
                if last_expected_num:
                    for resp_num in response_nums_float:
                        if self._is_number_equal(resp_num, last_expected_num, tolerance=0.02):
                            logger.debug(f"Bài toán chuyển động: Số khớp (từ số thuần túy)")
                            return True
        
        # Thơ toán học - thường là bài toán chia số dư
        elif "thơ toán học" in task_type.lower() or "chia" in expected_answer and "dư" in expected_answer:
            # Tìm số quả mỗi người và số dư
            per_person_patterns = [
                r'(\d+)\s*quả mỗi người',
                r'mỗi người được\s*(\d+)\s*quả',
                r'mỗi người\s*(\d+)\s*quả',
                r'chia được\s*(\d+)'
            ]
            
            remainder_patterns = [
                r'dư\s*(\d+)\s*quả',
                r'còn lại\s*(\d+)\s*quả',
                r'dư\s*(\d+)'
            ]
            
            # Kiểm tra số quả mỗi người
            per_person_found = False
            for pattern in per_person_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        for resp_match in response_matches:
                            if exp_match == resp_match:
                                logger.debug(f"Thơ toán học: Số quả mỗi người khớp: {resp_match} == {exp_match}")
                                per_person_found = True
            
            # Kiểm tra số dư
            remainder_found = False
            for pattern in remainder_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        for resp_match in response_matches:
                            if exp_match == resp_match:
                                logger.debug(f"Thơ toán học: Số quả dư khớp: {resp_match} == {exp_match}")
                                remainder_found = True
            
            # Cả hai phải đúng
            if per_person_found and remainder_found:
                return True
            
            # Nếu không tìm được qua pattern, thử dùng số trực tiếp
            if not (per_person_found and remainder_found) and expected_nums_float and response_nums_float:
                matched_count = 0
                for exp_num in expected_nums_float:
                    for resp_num in response_nums_float:
                        if self._is_number_equal(exp_num, resp_num, tolerance=0.01):
                            matched_count += 1
                            break
                
                # Nếu khớp cả hai số
                if matched_count >= 2:
                    logger.debug(f"Thơ toán học: Có ít nhất 2 số khớp (từ số thuần túy)")
                    return True
        
        # Bài toán về tuổi
        elif "tuổi" in task_type.lower() or "tuổi" in expected_answer:
            # Tìm số năm cần
            year_patterns = [
                r'(\d+)\s*năm',
                r'sau\s*(\d+)\s*năm',
                r'x\s*=\s*(\d+)'
            ]
            
            for pattern in year_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        for resp_match in response_matches:
                            if exp_match == resp_match:
                                logger.debug(f"Bài toán về tuổi: Số năm khớp: {resp_match} == {exp_match}")
                                return True
            
            # Nếu không tìm được qua pattern, thử dùng số trực tiếp
            if expected_nums_float and response_nums_float:
                # Lấy số cuối cùng trong expected_answer vì thường là kết quả
                last_expected_num = expected_nums_float[-1] if expected_nums_float else None
                
                if last_expected_num:
                    for resp_num in response_nums_float:
                        if self._is_number_equal(resp_num, last_expected_num, tolerance=0.01):
                            logger.debug(f"Bài toán về tuổi: Số khớp (từ số thuần túy)")
                            return True
        
        # Bài toán hồ bơi
        elif "hồ bơi" in task_type.lower() or "hồ nước" in expected_answer or "bể bơi" in expected_answer:
            # Tìm thời gian cần
            time_patterns = [
                r'(\d+[\.,]?\d*)\s*giờ',
                r'thời gian cần:\s*(\d+[\.,]?\d*)',
                r'sẽ đầy sau\s*(\d+[\.,]?\d*)',
                r'thời gian:?\s*(\d+[\.,]?\d*)',
                r'đầy sau\s*(\d+[\.,]?\d*)',
            ]
            
            for pattern in time_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        exp_num = float(exp_match.replace(',', '.'))
                        for resp_match in response_matches:
                            resp_num = float(resp_match.replace(',', '.'))
                            if self._is_number_equal(exp_num, resp_num, tolerance=0.03):
                                logger.debug(f"Bài toán hồ bơi: Thời gian khớp: {resp_num} ~ {exp_num}")
                                return True
            
            # Nếu không tìm được qua pattern, thử dùng số thập phân trực tiếp
            if expected_nums_float and response_nums_float:
                # Ưu tiên số thập phân vì kết quả thường là số giờ có phần thập phân
                decimal_expected = [num for num in expected_nums_float if num != int(num)]
                decimal_response = [num for num in response_nums_float if num != int(num)]
                
                if decimal_expected and decimal_response:
                    for exp_num in decimal_expected:
                        for resp_num in decimal_response:
                            if self._is_number_equal(exp_num, resp_num, tolerance=0.03):
                                logger.debug(f"Bài toán hồ bơi: Thời gian khớp (từ số thập phân): {resp_num} ~ {exp_num}")
                                return True
                
                # Nếu không có số thập phân, thử các số nguyên
                for exp_num in expected_nums_float:
                    for resp_num in response_nums_float:
                        if self._is_number_equal(resp_num, exp_num, tolerance=0.03):
                            logger.debug(f"Bài toán hồ bơi: Số khớp: {resp_num} ~ {exp_num}")
                            return True
        
        # Bài toán phân số
        elif "phân số" in task_type.lower() or "/" in expected_answer:
            # Tìm tổng và tích
            fraction_patterns = [
                r'tổng\s*=\s*(\d+/\d+)',
                r'tích\s*=\s*(\d+/\d+)',
                r'hiệu\s*=\s*(\d+/\d+)',
                r'thương\s*=\s*(\d+/\d+)'
            ]
            
            for pattern in fraction_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        for resp_match in response_matches:
                            if self._compare_fractions(exp_match, resp_match):
                                logger.debug(f"Bài toán phân số: Phân số khớp: {resp_match} == {exp_match}")
                                return True
            
            # Tìm tất cả các phân số
            all_fractions_pattern = r'(\d+/\d+)'
            response_fractions = re.findall(all_fractions_pattern, response)
            expected_fractions = re.findall(all_fractions_pattern, expected_answer)
            
            if response_fractions and expected_fractions:
                matched_count = 0
                for exp_frac in expected_fractions:
                    for resp_frac in response_fractions:
                        if self._compare_fractions(exp_frac, resp_frac):
                            matched_count += 1
                            break
                
                # Nếu khớp đủ số phân số
                if matched_count >= len(expected_fractions):
                    logger.debug(f"Bài toán phân số: Tất cả phân số khớp")
                    return True
                # Hoặc khớp một nửa số phân số (có thể là tổng hoặc tích)
                elif matched_count >= len(expected_fractions) / 2:
                    logger.debug(f"Bài toán phân số: Một nửa số phân số khớp")
                    return True
        
        # Bài toán từ vựng toán học - thường là bài toán mua sắm, giảm giá
        elif "từ vựng" in task_type.lower() or "đồng" in expected_answer or "giảm" in expected_answer:
            # Tìm số tiền trong đáp án
            money_patterns = [
                r'(\d+[\.,]?\d*)\s*đ',
                r'(\d+[\.,]?\d*)\s*đồng',
                r'(\d+[\.,]?\d*)\s*vnd',
                r'số tiền:?\s*(\d+[\.,]?\d*)',
                r'phải trả:?\s*(\d+[\.,]?\d*)',
                r'giá tiền:?\s*(\d+[\.,]?\d*)'
            ]
            
            for pattern in money_patterns:
                response_matches = re.findall(pattern, response)
                expected_matches = re.findall(pattern, expected_answer)
                
                if response_matches and expected_matches:
                    for exp_match in expected_matches:
                        exp_num = float(exp_match.replace(',', '.').replace('.', ''))  # Loại bỏ dấu phẩy trong số tiền
                        for resp_match in response_matches:
                            resp_num = float(resp_match.replace(',', '.').replace('.', ''))
                            if self._is_number_equal(exp_num, resp_num, tolerance=0.02):
                                logger.debug(f"Bài toán từ vựng: Số tiền khớp: {resp_num} ~ {exp_num}")
                                return True
            
            # Nếu không tìm được qua pattern, thử dùng số lớn trong câu trả lời
            if expected_nums_float and response_nums_float:
                # Sắp xếp số theo thứ tự giảm dần vì thường số tiền lớn nhất là kết quả
                expected_nums_float.sort(reverse=True)
                response_nums_float.sort(reverse=True)
                
                for resp_num in response_nums_float:
                    if any(self._is_number_equal(resp_num, exp_num, tolerance=0.02) for exp_num in expected_nums_float):
                        logger.debug(f"Bài toán từ vựng: Số khớp (từ số thuần túy)")
                        return True
                
        # 6. Đối với các loại bài toán khác hoặc không xác định rõ, thử các phương pháp chung
        
        # Kiểm tra nếu có số trong cả hai, so sánh trực tiếp
        if expected_nums_float and response_nums_float:
            matched_count = 0
            for exp_num in expected_nums_float:
                for resp_num in response_nums_float:
                    if self._is_number_equal(exp_num, resp_num, tolerance=0.02):
                        matched_count += 1
                        break
            
            # Nếu khớp từ 70% số trở lên
            if matched_count >= max(1, len(expected_nums_float) * 0.7):
                logger.debug(f"Đa số các số khớp: {matched_count}/{len(expected_nums_float)}")
                return True
            # Nếu có ít nhất 1 số khớp và không phải bài toán phức tạp
            elif matched_count >= 1 and len(expected_nums_float) <= 2:
                logger.debug(f"Số đơn lẻ khớp trong bài toán đơn giản")
                return True
        
        # Kiểm tra xem expected_answer có trong response không (toàn bộ câu trả lời)
        if expected_answer in response:
            logger.debug(f"Toàn bộ đáp án nằm trong câu trả lời")
            return True
        
        # So sánh từng phần (word overlap)
        response_words = set(response.split())
        expected_words = set(expected_answer.split())
        
        # Nếu có ít nhất 70% từ trong expected_answer xuất hiện trong response
        word_overlap = len(response_words.intersection(expected_words)) / len(expected_words) if expected_words else 0
        if word_overlap >= 0.7:
            logger.debug(f"Độ trùng lặp từ: {word_overlap:.2f}")
            return True
            
        logger.debug(f"Không tìm thấy kết quả phù hợp, câu trả lời sai")
        return False
        
    def _is_number_equal(self, num1, num2, tolerance=0.02):
        """
        So sánh hai số với sai số cho trước.
        
        Args:
            num1: Số thứ nhất
            num2: Số thứ hai
            tolerance (float): Sai số cho phép (phần trăm)
            
        Returns:
            bool: True nếu hai số bằng nhau trong phạm vi sai số
        """
        try:
            # Chuyển đổi thành float nếu là string
            if isinstance(num1, str):
                num1 = float(num1.replace(',', '.'))
            if isinstance(num2, str):
                num2 = float(num2.replace(',', '.'))
                
            # Xử lý trường hợp chia cho 0
            if num1 == 0 and num2 == 0:
                return True
            
            # Sử dụng sai số tương đối cho số lớn
            if abs(num1) > 1.0 or abs(num2) > 1.0:
                max_num = max(abs(num1), abs(num2))
                return abs(num1 - num2) <= tolerance * max_num
            
            # Sử dụng sai số tuyệt đối cho số nhỏ
            return abs(num1 - num2) <= tolerance
            
        except (ValueError, TypeError):
            # Nếu không thể chuyển đổi thành số, so sánh chuỗi
            return str(num1) == str(num2)

    def _compare_fractions(self, fraction1, fraction2):
        """
        So sánh hai phân số.
        
        Args:
            fraction1 (str): Phân số thứ nhất dạng "tử/mẫu"
            fraction2 (str): Phân số thứ hai dạng "tử/mẫu"
            
        Returns:
            bool: True nếu hai phân số bằng nhau
        """
        try:
            # Tách tử số và mẫu số
            num1, denom1 = map(int, fraction1.split('/'))
            num2, denom2 = map(int, fraction2.split('/'))
            
            # Rút gọn phân số trước khi so sánh
            # Tìm ước chung lớn nhất để rút gọn
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
            
            gcd1 = gcd(num1, denom1)
            gcd2 = gcd(num2, denom2)
            
            # Rút gọn phân số
            num1, denom1 = num1 // gcd1, denom1 // gcd1
            num2, denom2 = num2 // gcd2, denom2 // gcd2
            
            # So sánh các phân số đã rút gọn
            if num1 == num2 and denom1 == denom2:
                return True
                
            # So sánh giá trị
            return abs(num1/denom1 - num2/denom2) < 0.0001
        except (ValueError, ZeroDivisionError):
            return False

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
        
        # Số lần thử tối đa
        max_retries = 3
        
        # Danh sách mô hình để thử
        all_models = [self.reasoning_model]
        
        # Thêm Gemini nếu có API và khác model hiện tại
        if config.GEMINI_API_KEY and "gemini" not in all_models:
            all_models.append("gemini")
        
        # Thêm các model local nếu đã cấu hình
        if config.LLAMA_MODEL_PATH and "llama" not in all_models:
            all_models.append("llama")
        if config.QWEN_MODEL_PATH and "qwen" not in all_models:
            all_models.append("qwen")
            
        # Khởi tạo biến lưu lỗi
        last_error = None
            
        # Thử từng model với số lần retry
        for model_name in all_models:
            for attempt in range(max_retries):
                try:
                    # Lấy phản hồi đánh giá từ LLM
                    if self.verbose:
                        logger.info(f"Gửi yêu cầu đánh giá reasoning đến model: {model_name} (lần thử {attempt+1}/{max_retries})")
                        
                    # Thiết lập thời gian timeout dài hơn cho các lần thử lại
                    timeout = 30 + (attempt * 15)  # 30s cho lần đầu, sau đó tăng thêm 15s cho mỗi lần thử
                    
                    eval_response = generate_text(
                        model_name=model_name,
                        prompt=evaluation_prompt,
                        generation_config={
                            "temperature": 0.1,  # Giảm temperature để có kết quả ổn định
                            "max_tokens": 1000,   # Đủ dài cho đánh giá chi tiết
                            "timeout": timeout    # Thiết lập timeout tùy chỉnh
                        }
                    )
                    
                    # Nếu response là tuple (text, stats), lấy text
                    if isinstance(eval_response, tuple) and len(eval_response) > 0:
                        eval_response = eval_response[0]
                        
                    # Kiểm tra xem có lỗi không
                    if isinstance(eval_response, str):
                        if eval_response.startswith("[Error:"):
                            error_msg = eval_response.strip("[]")
                            logger.warning(f"Lỗi từ {model_name}: {error_msg}")
                            
                            # Lưu lỗi và tiếp tục thử
                            last_error = Exception(error_msg)
                            
                            # Chuyển sang lần thử tiếp theo nếu còn lần thử
                            if attempt < max_retries - 1:
                                logger.info(f"Thử lại với {model_name} (lần {attempt+2}/{max_retries})")
                                time.sleep(1)  # Chờ 1 giây trước khi thử lại
                                continue
                            else:
                                # Hết số lần thử với model này, chuyển sang model khác
                                break
                        else:
                            # Thành công, parse kết quả đánh giá
                            return self._parse_reasoning_evaluation(eval_response)
                
                except Exception as e:
                    last_error = e
                    error_type = str(e).split(":")[0] if ":" in str(e) else "UNKNOWN_ERROR"
                    
                    # Log lỗi
                    logger.error(f"Lỗi trong quá trình đánh giá reasoning: {str(e)}")
                    
                    # Kiểm tra nếu là lỗi cần retry
                    retriable_errors = [
                        "RETRIABLE_ERROR", "UNKNOWN_ERROR", "QUOTA_LIMIT_ERROR", 
                        "TIMEOUT_ERROR", "CIRCUIT_BREAKER_OPEN"
                    ]
                    
                    if any(err_type in error_type for err_type in retriable_errors):
                        # Nếu còn lần thử
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Tăng thời gian chờ theo số lần thử
                            logger.info(f"Lỗi có thể thử lại. Chờ {wait_time}s và thử lại với {model_name}")
                            time.sleep(wait_time)
                            continue
                    
                    # Các lỗi khác hoặc hết số lần thử với model hiện tại
                    logger.warning(f"Thử lại không thành công với {model_name}. Chuyển sang model khác nếu có.")
                    break
        
        # Nếu tất cả model đều thất bại, trả về kết quả mặc định
        error_msg = str(last_error) if last_error else "Không xác định"
        logger.error(f"Tất cả model đều thất bại trong việc đánh giá suy luận. Lỗi cuối: {error_msg}")
        
        return {
            'accuracy': 0,
            'reasoning': 0,
            'completeness': 0,
            'explanation': 0,
            'cultural_context': 0,
            'average': 0,
            'comment': f"Lỗi khi đánh giá: {error_msg}"
        }

    def analyze_and_report(self):
        """
        Phân tích các kết quả thu thập được và tạo báo cáo.
        """
        import config as app_config
        
        if self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("Không có kết quả để phân tích")
            return
        
        self.logger.info(f"Bắt đầu phân tích {len(self.results_df)} kết quả đánh giá")
        
        # Tạo đối tượng ResultAnalyzer
        analyzer = ResultAnalyzer(
            results_df=self.results_df,
            reasoning_evaluation_config=self.reasoning_evaluation_config,
            reasoning_model=self.reasoning_model,
            language=self.language,
            verbose=True
        )
        
        # Phân tích kết quả cơ bản
        self.results_df = analyzer.analyze()
        
        # Tính toán các metrics nâng cao (BERT, METEOR, F1)
        self.logger.info("Tính toán các metrics nâng cao (BERT score, METEOR score, F1 score)...")
        self.results_df = analyzer.calculate_additional_metrics()
        
        # Xuất kết quả thô
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        raw_results_dir = os.path.join(self.results_dir, "raw_results")
        os.makedirs(raw_results_dir, exist_ok=True)
        
        # Lưu kết quả dạng CSV
        csv_path = os.path.join(raw_results_dir, f"evaluation_results_{timestamp}.csv")
        self.results_df.to_csv(csv_path, index=False)
        self.logger.info(f"Đã lưu kết quả thô vào {csv_path}")
        
        # Lưu kết quả dạng JSON
        json_path = os.path.join(raw_results_dir, f"evaluation_results_{timestamp}.json")
        self.results_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        self.logger.info(f"Đã lưu kết quả thô vào {json_path}")
        
        # Tạo báo cáo
        self.logger.info("Tạo báo cáo từ kết quả phân tích...")
        
        from core.reporting import Reporting
        reporter = Reporting(
            results_df=self.results_df, 
            output_dir=self.results_dir,
            timestamp=timestamp
        )
        
        report_paths = reporter.generate_reports()
        
        if report_paths:
            for report_type, path in report_paths.items():
                self.logger.info(f"Đã tạo báo cáo {report_type}: {path}")
        else:
            self.logger.warning("Không thể tạo báo cáo")
        
        self.logger.info("Quá trình phân tích và báo cáo hoàn tất")
        
        return {
            'csv_path': csv_path,
            'json_path': json_path,
            'reports': report_paths
        }

    def _evaluate_local_model_batch(self, model_name, batch_prompts, batch_questions):
        """
        Đánh giá một batch câu hỏi với model local để tối ưu hiệu suất.
        
        Args:
            model_name (str): Tên model
            batch_prompts (list): Danh sách prompts theo từng loại
            batch_questions (list): Danh sách questions tương ứng
            
        Returns:
            list: Danh sách kết quả đánh giá
        """
        if not batch_prompts:
            return []
            
        results = []
        
        # Group các prompts theo loại để áp dụng max_tokens phù hợp
        prompt_type_groups = {}
        for prompt_info in batch_prompts:
            prompt_type = prompt_info["prompt_type"]
            if prompt_type not in prompt_type_groups:
                prompt_type_groups[prompt_type] = []
            prompt_type_groups[prompt_type].append(prompt_info)
        
        # Import config để lấy max_tokens dựa trên prompt_type
        import config as app_config
        
        # Xử lý từng nhóm prompt_type
        for prompt_type, prompts_in_group in prompt_type_groups.items():
            # Lấy max_tokens phù hợp cho prompt_type này
            max_tokens = app_config.get_max_tokens(model_name, prompt_type)
            logger.debug(f"Batch processing: Sử dụng max_tokens={max_tokens} cho {model_name}/{prompt_type}")
            
            # Chuẩn bị input cho model
            input_prompts = [p["prompt"] for p in prompts_in_group]
            question_indices = [p["question_idx"] for p in prompts_in_group]
            
            # Generate các responses
            batch_config = {
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95
            }
            
            start_time = time.time()
            responses = self.model_interface.batch_generate_text(model_name, input_prompts, batch_config)
            batch_latency = time.time() - start_time
            
            # Average latency per question
            avg_latency = batch_latency / len(input_prompts) if input_prompts else 0
            
            # Xử lý từng response
            for i, response in enumerate(responses):
                prompt_info = prompts_in_group[i]
                question_idx = question_indices[i]
                question = batch_questions[i]
                
                # Xử lý kết quả và thêm vào danh sách
                result = self._process_evaluation_result(
                    model_name=model_name,
                    prompt_type=prompt_type,
                    question=question,
                    question_idx=question_idx,
                    response=response,
                    latency=avg_latency
                )
                
                if result:
                    results.append(result)
        
        return results
    
    def _process_evaluation_result(self, model_name, prompt_type, question, question_idx, response, latency):
        """
        Xử lý kết quả đánh giá từ một câu trả lời.
        
        Args:
            model_name (str): Tên của model
            prompt_type (str): Loại prompt
            question (dict): Thông tin câu hỏi
            question_idx (int): Chỉ số của câu hỏi
            response (str): Câu trả lời từ model
            latency (float): Thời gian trả lời
            
        Returns:
            dict: Kết quả đánh giá
        """
        import json
        import config as app_config  # Import trực tiếp app_config
        
        question_id = question.get('id', question_idx)
        question_text = question.get('question', question.get('text', ''))
        expected_answer = question.get('solution', question.get('answer', ''))
        question_type = question.get('type', 'general')
        difficulty = question.get('difficulty', 'Không xác định')
        
        try:
            # Kiểm tra đáp án
            is_correct = self._check_answer(response, expected_answer, question_type)
            
            # Khởi tạo các giá trị mặc định cho reasoning scores
            reasoning_accuracy = 0
            reasoning_reasoning = 0
            reasoning_completeness = 0
            reasoning_explanation = 0
            reasoning_cultural_context = 0
            reasoning_average = 0
            
            # Đánh giá khả năng suy luận nếu được bật
            if self.reasoning_evaluation_enabled and app_config.REASONING_EVALUATION_CONFIG.get('enabled', True):
                try:
                    logger.debug(f"Bắt đầu đánh giá reasoning cho {model_name}/{prompt_type}/{question_id}")

                    # Tính heuristic accuracy dự trên kết quả boolean is_correct
                    accuracy_heuristic = 5 if is_correct else 1

                    # Tạo prompt đánh giá
                    reasoning_prompt = self._create_reasoning_evaluation_prompt(question_text, response, expected_answer)

                    # Lấy phản hồi đánh giá
                    reasoning_model = "groq/" + app_config.REASONING_EVALUATION_CONFIG.get('model', 'llama3-70b-8192')
                    
                    reasoning_eval = self.model_interface.get_response(
                        model_name=reasoning_model,
                        prompt=reasoning_prompt,
                        max_tokens=800
                    )

                    # Parse kết quả đánh giá
                    reasoning_scores = self._parse_reasoning_evaluation(reasoning_eval)

                    # Trích xuất các giá trị
                    reasoning_accuracy = reasoning_scores.get('accuracy', 0)
                    reasoning_reasoning = reasoning_scores.get('reasoning', 0)
                    reasoning_completeness = reasoning_scores.get('completeness', 0)
                    reasoning_explanation = reasoning_scores.get('explanation', 0)
                    reasoning_cultural_context = reasoning_scores.get('cultural_context', 0)
                    reasoning_average = reasoning_scores.get('average', 0)

                    # Đảm bảo có điểm accuracy
                    if reasoning_accuracy == 0:
                        reasoning_accuracy = accuracy_heuristic
                        logger.debug(f"Sử dụng accuracy heuristic: {accuracy_heuristic}/5")

                except Exception as e:
                    logger.error(f"Lỗi trong quá trình đánh giá reasoning: {str(e)}")
                    logger.debug(traceback.format_exc())
                    reasoning_accuracy = 5 if is_correct else 1
                    reasoning_average = reasoning_accuracy
            
            # Ghi log kết quả
            short_response = response[:100] + "..." if len(response) > 100 else response
            logger.debug(f"Kết quả: Model={model_name}, Prompt={prompt_type}, Question={question_id}")
            logger.debug(f"Đúng/Sai: {is_correct}, Thời gian: {latency:.2f}s")
            logger.debug(f"Câu trả lời: {short_response}")
            
            # Tạo đối tượng reasoning_scores để lưu dưới dạng chuỗi JSON
            reasoning_scores_dict = {
                'accuracy': reasoning_accuracy,
                'reasoning': reasoning_reasoning,
                'completeness': reasoning_completeness,
                'explanation': reasoning_explanation,
                'cultural_context': reasoning_cultural_context,
                'average': reasoning_average
            }
            
            # Chuyển đổi dictionary thành chuỗi JSON
            reasoning_scores_json = json.dumps(reasoning_scores_dict)
            
            # Tạo kết quả
            result = {
                'model_name': model_name,
                'prompt_type': prompt_type,
                'question_id': question_id,
                'question_text': question_text,
                'question_type': question_type,
                'difficulty': difficulty,
                'response': response,
                'expected_answer': expected_answer,
                'is_correct': is_correct,
                'latency': latency,
                'timestamp': datetime.datetime.now().isoformat(),
                # Phẳng hóa các chỉ số reasoning
                'reasoning_accuracy': reasoning_accuracy,
                'reasoning_reasoning': reasoning_reasoning,
                'reasoning_completeness': reasoning_completeness,
                'reasoning_explanation': reasoning_explanation,
                'reasoning_cultural_context': reasoning_cultural_context,
                'reasoning_average': reasoning_average,
                # Lưu trữ dictionary dưới dạng chuỗi JSON
                'reasoning_scores_str': reasoning_scores_json,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý kết quả cho {model_name}/{prompt_type}/{question_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
