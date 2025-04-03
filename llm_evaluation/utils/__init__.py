"""
Các tiện ích chung cho framework đánh giá LLM.
Các module trong gói này độc lập với logic core.
"""

# File utils
from .file_utils import (
    ensure_dir, get_timestamp, save_json, load_json, 
    save_yaml, load_yaml, save_csv, load_csv,
    save_pickle, load_pickle, get_file_size, safe_file_write
)

# Logging utils
from .logging_utils import (
    setup_logging, get_logger, log_function_call
)

# Text utils
from .text_utils import (
    clean_text, remove_diacritics, normalize_vietnamese_text,
    simple_vietnamese_tokenize, is_vietnamese_text,
    calculate_text_similarity, extract_vietnamese_keywords,
    extract_numbers_from_text, format_vietnamese_number,
    extract_vietnamese_sentences
)

# Metrics utils
from .metrics_utils import (
    calculate_binary_metrics, calculate_multiclass_metrics,
    calculate_regression_metrics, calculate_exact_match_accuracy,
    calculate_token_overlap, calculate_llm_reasoning_metrics,
    calculate_answer_correctness, calculate_latency_metrics
)

# Visualization utils
from .visualization_utils import (
    set_visualization_style, create_accuracy_comparison_plot,
    create_metric_heatmap, create_latency_plot, create_radar_chart,
    create_prompt_type_comparison, create_sample_count_plot,
    save_figure, plot_confusion_matrix
)

# Config utils
from .config_utils import (
    Config, ModelConfig, PromptConfig, EvaluationConfig, LoggingConfig,
    load_config, save_config, validate_config, merge_configs,
    update_config_from_env, create_default_config
)

# Memory utils
from .memory_utils import (
    get_memory_usage, cleanup_memory, start_memory_monitoring,
    stop_memory_monitoring, track_object, untrack_object,
    register_cleanup_callback, estimate_object_size,
    MemoryUsageDecorator
)

__all__ = [
    # File utils
    'ensure_dir', 'get_timestamp', 'save_json', 'load_json', 
    'save_yaml', 'load_yaml', 'save_csv', 'load_csv',
    'save_pickle', 'load_pickle', 'get_file_size', 'safe_file_write',
    
    # Logging utils
    'setup_logging', 'get_logger', 'log_function_call',
    
    # Text utils
    'clean_text', 'remove_diacritics', 'normalize_vietnamese_text',
    'simple_vietnamese_tokenize', 'is_vietnamese_text',
    'calculate_text_similarity', 'extract_vietnamese_keywords',
    'extract_numbers_from_text', 'format_vietnamese_number',
    'extract_vietnamese_sentences',
    
    # Metrics utils
    'calculate_binary_metrics', 'calculate_multiclass_metrics',
    'calculate_regression_metrics', 'calculate_exact_match_accuracy',
    'calculate_token_overlap', 'calculate_llm_reasoning_metrics',
    'calculate_answer_correctness', 'calculate_latency_metrics',
    
    # Visualization utils
    'set_visualization_style', 'create_accuracy_comparison_plot',
    'create_metric_heatmap', 'create_latency_plot', 'create_radar_chart',
    'create_prompt_type_comparison', 'create_sample_count_plot',
    'save_figure', 'plot_confusion_matrix',
    
    # Config utils
    'Config', 'ModelConfig', 'PromptConfig', 'EvaluationConfig', 'LoggingConfig',
    'load_config', 'save_config', 'validate_config', 'merge_configs',
    'update_config_from_env', 'create_default_config',
    
    # Memory utils
    'get_memory_usage', 'cleanup_memory', 'start_memory_monitoring',
    'stop_memory_monitoring', 'track_object', 'untrack_object',
    'register_cleanup_callback', 'estimate_object_size',
    'MemoryUsageDecorator'
]
