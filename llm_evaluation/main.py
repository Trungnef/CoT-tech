"""
Điểm vào chính cho ứng dụng đánh giá LLM.
Xử lý tham số dòng lệnh và khởi chạy quá trình đánh giá.
"""

import argparse
import sys
import os
from pathlib import Path
import datetime
import traceback

# Thêm thư mục cha vào sys.path để import các module
sys.path.append(str(Path(__file__).parent.absolute()))

# Import các module cần thiết
import config
from core.evaluator import Evaluator
from core.checkpoint_manager import CheckpointManager
from utils.data_loader import load_questions
from utils.logging_setup import setup_logging, get_logger, log_section

def parse_arguments():
    """Phân tích các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Đánh giá hiệu suất của các mô hình LLM với các loại prompt khác nhau")
    
    # Các tham số chính
    parser.add_argument("--models", nargs='+', default=config.DEFAULT_MODELS, 
                        help="Danh sách các mô hình cần đánh giá (ví dụ: llama qwen gemini)")
    
    parser.add_argument("--prompts", nargs='+', default=config.DEFAULT_PROMPTS,
                        help="Danh sách các loại prompt cần đánh giá (ví dụ: zero_shot few_shot_3 cot_self_consistency_3 react)")
    
    parser.add_argument("--questions-file", default=str(config.QUESTIONS_FILE),
                        help="Đường dẫn đến file JSON chứa danh sách câu hỏi")
    
    parser.add_argument("--results-dir", default=str(config.RESULTS_DIR),
                        help="Thư mục để lưu kết quả đánh giá")
    
    parser.add_argument("--max-questions", type=int, default=config.DEFAULT_MAX_QUESTIONS,
                        help="Số lượng câu hỏi tối đa cần đánh giá. Mặc định là tất cả")
    
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE,
                        help="Kích thước batch cho đánh giá")
    
    parser.add_argument("--checkpoint-frequency", type=int, default=config.DEFAULT_CHECKPOINT_FREQUENCY,
                        help="Tần suất lưu checkpoint (số câu hỏi)")
    
    # Lọc câu hỏi theo tags, difficulty và question_type
    parser.add_argument("--include-tags", nargs='+', 
                        help="Chỉ bao gồm câu hỏi có ít nhất một trong các tags được chỉ định")
    
    parser.add_argument("--exclude-tags", nargs='+',
                        help="Loại trừ câu hỏi có bất kỳ tag nào trong danh sách này")
    
    parser.add_argument("--difficulty-levels", nargs='+', choices=["Dễ", "Trung bình", "Khó"],
                        help="Chỉ bao gồm câu hỏi có độ khó được chỉ định (Dễ, Trung bình, Khó)")
    
    parser.add_argument("--question-types", nargs='+',
                        help="Chỉ bao gồm câu hỏi có loại được chỉ định (VD: logic, math, text)")
    
    # Các flag
    parser.add_argument("--test-run", action="store_true", 
                        help="Chạy thử nghiệm với số lượng câu hỏi, mô hình và prompt giới hạn")
    
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục từ checkpoint gần nhất")
    
    parser.add_argument("--checkpoint", type=str,
                        help="Đường dẫn đến checkpoint cụ thể để khôi phục")
    
    parser.add_argument("--skip-reasoning-eval", action="store_true",
                        help="Bỏ qua việc đánh giá khả năng suy luận")
    
    parser.add_argument("--no-cache", action="store_true",
                        help="Không sử dụng cache model (luôn tải lại model)")
    
    parser.add_argument("--question-ids", nargs='+', type=int,
                        help="Chỉ đánh giá các câu hỏi với ID cụ thể")
    
    parser.add_argument("--debug", action="store_true",
                        help="Bật chế độ debug với logging chi tiết hơn")
    
    # Tham số cho logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      default="INFO", help="Level logging")
    
    parser.add_argument("--log-file", help="Tên file log. Mặc định tạo file log với timestamp")
    
    # Tham số cho việc song song hóa
    parser.add_argument("--parallel", action="store_true",
                        help="Chạy đánh giá song song cho các mô hình khác nhau")
    
    parser.add_argument("--gpu-ids", nargs='+', type=int, default=[0],
                        help="Danh sách GPU IDs để sử dụng (ví dụ: 0 1 2)")
    
    parser.add_argument("--report-only", action="store_true",
                        help="Chỉ tạo báo cáo từ kết quả hiện có mà không đánh giá")
    
    parser.add_argument("--results-file", 
                        help="Đường dẫn đến file kết quả hiện có để tạo báo cáo (sử dụng với --report-only)")
    
    return parser.parse_args()

def main():
    """Hàm chính của ứng dụng."""
    # Phân tích tham số dòng lệnh
    args = parse_arguments()
    
    # Thiết lập logging
    log_dir = os.path.join(args.results_dir, "logs")
    import logging
    log_level = getattr(logging, args.log_level) if hasattr(logging, args.log_level) else logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    
    # Thiết lập logging tập trung
    logger = setup_logging(
        log_dir=log_dir,
        log_file=args.log_file,
        console_level=log_level,
        file_level=logging.DEBUG,  # Luôn lưu đầy đủ log vào file
        module_levels={
            "llm_evaluation.core.model_interface": logging.INFO if not args.debug else logging.DEBUG,
            "urllib3": logging.WARNING,
            "httpx": logging.WARNING
        }
    )
    
    # Lấy logger cho module main
    logger = get_logger("main")
    
    # Hiển thị thông tin bắt đầu
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_section(logger, f"Bắt đầu đánh giá LLM ({timestamp})")
    logger.info(f"Các tham số: {vars(args)}")
    
    # Kiểm tra và hiển thị cấu hình
    config.validate_config()
    config.display_config_summary()
    
    # Đảm bảo thư mục results chính tồn tại
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Điều chỉnh các tham số cho test run
    if args.test_run:
        logger.info("Chế độ TEST RUN - Sử dụng cấu hình giới hạn để chạy thử nghiệm")
        args.models = args.models[:1] if args.models else []
        args.prompts = args.prompts[:1] if args.prompts else []
        args.max_questions = min(args.max_questions or 2, 2)
        args.resume = False  # Không sử dụng resume trong test run
    
    # Chế độ report-only: chỉ tạo báo cáo từ kết quả hiện có
    if args.report_only:
        if not args.results_file or not os.path.exists(args.results_file):
            logger.error("Chế độ report-only yêu cầu phải chỉ định --results-file tới file kết quả hiện có")
            return
            
        logger.info(f"Chế độ REPORT-ONLY - Tạo báo cáo từ file kết quả: {args.results_file}")
        
        # Tạo thư mục report_only với timestamp hiện tại
        report_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(args.results_dir, f"report_only_{report_timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(os.path.join(report_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(report_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(report_dir, "analyzed_results"), exist_ok=True)
        
        # Cập nhật hàm generate_reports với thư mục mới
        from core.reporting import generate_reports
        generate_reports(args.results_file, report_dir, report_timestamp)
        logger.info(f"Báo cáo đã được tạo trong thư mục: {report_dir}")
        return
    
    # Tải danh sách câu hỏi
    try:
        questions = load_questions(args.questions_file)
        original_count = len(questions)
        
        # Lọc câu hỏi theo ID nếu được chỉ định
        if args.question_ids:
            questions = [q for q in questions if q.get('id') in args.question_ids]
            logger.info(f"Đã lọc {original_count} câu hỏi xuống còn {len(questions)} câu hỏi theo ID đã chỉ định")
            original_count = len(questions)
        
        # Lọc câu hỏi theo tags
        if args.include_tags:
            include_tags = set(tag.lower() for tag in args.include_tags)
            questions = [
                q for q in questions 
                if 'tags' in q and any(tag.lower() in include_tags for tag in q['tags'])
            ]
            logger.info(f"Đã lọc xuống còn {len(questions)} câu hỏi theo include-tags: {args.include_tags}")
            original_count = len(questions)
        
        if args.exclude_tags:
            exclude_tags = set(tag.lower() for tag in args.exclude_tags)
            questions = [
                q for q in questions 
                if 'tags' not in q or not any(tag.lower() in exclude_tags for tag in q['tags'])
            ]
            logger.info(f"Đã lọc xuống còn {len(questions)} câu hỏi sau khi loại trừ tags: {args.exclude_tags}")
            original_count = len(questions)
        
        # Lọc câu hỏi theo độ khó
        if args.difficulty_levels:
            difficulty_levels = set(args.difficulty_levels)
            questions = [
                q for q in questions 
                if 'difficulty' in q and q['difficulty'] in difficulty_levels
            ]
            logger.info(f"Đã lọc xuống còn {len(questions)} câu hỏi theo độ khó: {args.difficulty_levels}")
            original_count = len(questions)
        
        # Lọc câu hỏi theo loại
        if args.question_types:
            question_types = set(qt.lower() for qt in args.question_types)
            questions = [
                q for q in questions 
                if 'type' in q and q['type'].lower() in question_types
            ]
            logger.info(f"Đã lọc xuống còn {len(questions)} câu hỏi theo loại: {args.question_types}")
            original_count = len(questions)
        
        # Giới hạn số lượng câu hỏi nếu cần
        if args.max_questions and args.max_questions < len(questions):
            questions = questions[:args.max_questions]
            logger.info(f"Đã giới hạn xuống {len(questions)} câu hỏi theo max-questions: {args.max_questions}")
            
        logger.info(f"Đã tải và lọc {len(questions)} câu hỏi từ {args.questions_file}")
        
        if len(questions) == 0:
            logger.error("Không có câu hỏi nào sau khi lọc. Vui lòng kiểm tra lại các tiêu chí lọc.")
            return
    except Exception as e:
        logger.error(f"Lỗi khi tải câu hỏi: {str(e)}")
        logger.debug(f"Chi tiết lỗi: {traceback.format_exc()}")
        return
    
    # Cập nhật cấu hình đánh giá reasoning
    reasoning_eval_enabled = config.REASONING_EVALUATION_CONFIG["enabled"] and not args.skip_reasoning_eval
    
    # Khởi tạo Evaluator
    evaluator = Evaluator(
        models_to_evaluate=args.models,
        prompts_to_evaluate=args.prompts,
        questions=questions,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        checkpoint_frequency=args.checkpoint_frequency,
        use_cache=not args.no_cache,
        reasoning_evaluation_enabled=reasoning_eval_enabled,
        parallel=args.parallel,
        gpu_ids=args.gpu_ids,
        timestamp=timestamp
    )
    
    # Kiểm tra checkpoint cụ thể nếu được chỉ định
    if args.checkpoint:
        resume_from_specific = args.checkpoint
        log_section(logger, f"Sử dụng checkpoint đã chỉ định: {resume_from_specific}")
        
        # Tạo checkpoint manager để tải checkpoint
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(args.results_dir, "checkpoints"),
            timestamp=timestamp
        )
        
        # Tải thông tin về checkpoint
        checkpoint_info = checkpoint_manager.get_checkpoint_info(resume_from_specific)
        if checkpoint_info:
            logger.info(f"Thông tin checkpoint:")
            for key, value in checkpoint_info.items():
                logger.info(f"  {key}: {value}")
        
    # Nếu --resume được chỉ định nhưng không có --checkpoint
    elif args.resume:
        log_section(logger, "Tìm kiếm checkpoint gần nhất để tiếp tục")
        
        # Tạo checkpoint manager chỉ để kiểm tra checkpoints
        # Chú ý: Điều này chỉ để hiển thị thông tin, Evaluator sẽ tự tìm checkpoint sau
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(args.results_dir, "checkpoints"),
            timestamp=timestamp
        )
        
        # Tìm checkpoint gần nhất
        checkpoint_files = checkpoint_manager._get_checkpoint_files()
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_info = checkpoint_manager.get_checkpoint_info(latest_checkpoint)
            if checkpoint_info:
                logger.info(f"Tìm thấy checkpoint gần nhất:")
                for key, value in checkpoint_info.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.warning("Tìm thấy checkpoint nhưng không thể đọc thông tin")
        else:
            logger.warning("Không tìm thấy checkpoint để tiếp tục. Sẽ bắt đầu đánh giá mới.")
    
    # Chạy quá trình đánh giá
    try:
        if args.checkpoint:
            # Tải từ checkpoint cụ thể
            resume_from_specific = args.checkpoint
            evaluator.run_evaluation(resume=True, checkpoint_path=resume_from_specific)
        else:
            # Tải từ checkpoint tự động hoặc bắt đầu mới
            evaluator.run_evaluation(resume=args.resume)
            
        logger.info("Đánh giá đã hoàn thành thành công.")
    except KeyboardInterrupt:
        logger.info("Đánh giá bị ngắt bởi người dùng. Lưu trạng thái hiện tại...")
        try:
            checkpoint_path = evaluator.save_checkpoint()
            logger.info(f"Đã lưu checkpoint tại: {checkpoint_path}")
            logger.info("Bạn có thể tiếp tục sau với cờ --resume hoặc --checkpoint")
        except Exception as e:
            logger.error(f"Không thể lưu checkpoint khi bị ngắt: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình đánh giá: {str(e)}")
        logger.debug(f"Chi tiết lỗi: {traceback.format_exc()}")
        logger.info("Thử lưu checkpoint để có thể khôi phục...")
        try:
            checkpoint_path = evaluator.save_checkpoint()
            logger.info(f"Đã lưu checkpoint tại: {checkpoint_path}")
            logger.info("Bạn có thể tiếp tục sau với cờ --resume hoặc --checkpoint path/to/checkpoint.json")
        except Exception as ce:
            logger.error(f"Không thể lưu checkpoint: {str(ce)}")

if __name__ == "__main__":
    main()
