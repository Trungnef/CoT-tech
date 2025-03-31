"""
Model evaluation functionality for comparing different LLMs.
"""

from src.evaluators.metrics import (
    evaluate_reasoning_quality,
    evaluate_answer_correctness,
    evaluate_answer_confidence,
    evaluate_answer_by_prompt_type
)

from src.evaluators.evaluator import ModelEvaluator
from src.evaluators.parallel_evaluator import (
    ParallelEvaluator,
    parse_gpu_allocation
)

__all__ = [
    # Metrics
    'evaluate_reasoning_quality',
    'evaluate_answer_correctness',
    'evaluate_answer_confidence',
    'evaluate_answer_by_prompt_type',
    
    # Evaluators
    'ModelEvaluator',
    'ParallelEvaluator',
    'parse_gpu_allocation'
]
