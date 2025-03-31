"""
Metrics for evaluating model responses.
"""

import re
import numpy as np
from typing import Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

def evaluate_reasoning_quality(response: str, expected_solution: str) -> float:
    """
    Evaluate the quality of the reasoning in a response.
    
    Args:
        response: Model's response
        expected_solution: Expected solution
        
    Returns:
        float: Reasoning quality score (0-1)
    """
    # Check for key reasoning steps
    reasoning_patterns = [
        r"step\s*\d+",
        r"first",
        r"second",
        r"third",
        r"next",
        r"finally",
        r"therefore",
        r"consequently",
        r"thus",
        r"hence",
        r"so",
        r"because",
        r"since",
        r"as a result"
    ]
    
    # Count reasoning indicators
    indicator_count = 0
    for pattern in reasoning_patterns:
        matches = re.findall(pattern, response.lower())
        indicator_count += len(matches)
    
    # Check for mathematical expressions
    math_expr_count = len(re.findall(r'(\d+\s*[\+\-\*\/]\s*\d+)', response))
    
    # Check for use of variables
    var_usage_count = len(re.findall(r'([a-zA-Z])\s*=', response))
    
    # Check for equation solving patterns
    equation_steps = len(re.findall(r'([\=\>\<\-\+])', response))
    
    # Calculate structural similarity
    similarity = SequenceMatcher(None, response, expected_solution).ratio()
    
    # Calculate reasoning score (weighted components)
    reasoning_score = min(1.0, (
        (0.4 * min(1.0, indicator_count / 5)) + 
        (0.2 * min(1.0, math_expr_count / 3)) + 
        (0.2 * min(1.0, var_usage_count / 2)) + 
        (0.1 * min(1.0, equation_steps / 10)) +
        (0.1 * similarity)
    ))
    
    return reasoning_score

def evaluate_answer_correctness(response: str, expected_solution: str) -> float:
    """
    Evaluate the correctness of an answer.
    
    Args:
        response: Model's response
        expected_solution: Expected solution
        
    Returns:
        float: Correctness score (0-1)
    """
    # Extract final numerical answers if present
    response_numbers = re.findall(r'(?:answer|result|solution).*?([+-]?\d+(?:\.\d+)?)', response.lower())
    expected_numbers = re.findall(r'(?:answer|result|solution).*?([+-]?\d+(?:\.\d+)?)', expected_solution.lower())
    
    # If no explicit answers found, try to extract any numbers
    if not response_numbers:
        response_numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', response)
    if not expected_numbers:
        expected_numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', expected_solution)
    
    # If numerical answers are found, compare them
    if response_numbers and expected_numbers:
        # Get the last number from each as the final answer
        response_answer = float(response_numbers[-1])
        expected_answer = float(expected_numbers[-1])
        
        # Check if answers are approximately equal
        if abs(response_answer - expected_answer) < 0.01:
            return 1.0
        else:
            # Calculate similarity based on relative difference
            max_val = max(abs(response_answer), abs(expected_answer))
            if max_val > 0:
                diff = abs(response_answer - expected_answer) / max_val
                return max(0.0, 1.0 - min(1.0, diff))
            else:
                return 0.0
    
    # Text similarity approach for non-numerical answers
    similarity = SequenceMatcher(None, response.lower(), expected_solution.lower()).ratio()
    
    # Check for specific keywords in the expected solution
    keywords = re.findall(r'\b(\w+)\b', expected_solution.lower())
    if keywords:
        keyword_match_count = sum(1 for keyword in keywords if keyword in response.lower())
        keyword_score = keyword_match_count / len(keywords)
        
        # Combine similarity and keyword matching
        correctness_score = 0.6 * similarity + 0.4 * keyword_score
    else:
        correctness_score = similarity
    
    return correctness_score

def evaluate_answer_confidence(response: str) -> float:
    """
    Evaluate the confidence level expressed in the response.
    
    Args:
        response: Model's response
        
    Returns:
        float: Confidence score (0-1)
    """
    # High confidence patterns
    high_confidence_patterns = [
        r'chắc chắn', r'dứt khoát', r'rõ ràng', r'không còn nghi ngờ gì',
        r'hiển nhiên', r'kết quả là', r'vậy', r'vì vậy', r'do đó',
        r'kết luận', r'có thể khẳng định', r'khẳng định', r'definitely',
        r'certainly', r'clearly', r'without doubt', r'obviously'
    ]
    
    # Low confidence patterns
    low_confidence_patterns = [
        r'có lẽ', r'có thể', r'không chắc', r'tôi nghĩ', r'dường như',
        r'không rõ', r'chưa chắc', r'tôi đoán', r'tôi tin', r'tôi cho rằng',
        r'tôi ước tính', r'không biết', r'không dám chắc', r'bối rối',
        r'phức tạp', r'khó', r'perhaps', r'maybe', r'possibly', r'I think',
        r'I believe', r'I guess', r'seems', r'unclear', r'not sure'
    ]
    
    # Count confidence markers
    high_confidence_count = sum(1 for pattern in high_confidence_patterns 
                               if re.search(pattern, response.lower()))
    low_confidence_count = sum(1 for pattern in low_confidence_patterns 
                              if re.search(pattern, response.lower()))
    
    total_markers = high_confidence_count + low_confidence_count
    
    if total_markers == 0:
        confidence_score = 0.5  # Neutral if no markers
    else:
        confidence_score = high_confidence_count / total_markers
    
    # Check for numbers in conclusion (indicates confidence)
    conclusion_part = response.split(".")[-3:]  # 3 sentences at the end
    conclusion_text = ".".join(conclusion_part)
    has_numbers_in_conclusion = bool(re.search(r'\d+', conclusion_text))
    
    # Adjust score based on conclusion
    if has_numbers_in_conclusion:
        confidence_score = min(1.0, confidence_score + 0.2)
    
    # Length penalty (very long responses may indicate uncertainty)
    response_length = len(response.split())
    if response_length > 300:
        confidence_score = max(0.1, confidence_score - 0.1)
    
    return confidence_score

def evaluate_answer_by_prompt_type(
    question: str,
    response: str, 
    expected_solution: str, 
    prompt_type: str
) -> Dict[str, float]:
    """
    Evaluate an answer based on the prompt type used.
    
    Args:
        question: The question asked
        response: Model's response
        expected_solution: Expected solution
        prompt_type: Type of prompt used
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Calculate common metrics
    correctness_score = evaluate_answer_correctness(response, expected_solution)
    reasoning_score = evaluate_reasoning_quality(response, expected_solution)
    confidence_score = evaluate_answer_confidence(response)
    
    # Create results dictionary
    results = {
        "correctness_score": correctness_score,
        "reasoning_score": reasoning_score,
        "confidence_score": confidence_score,
    }
    
    # Apply prompt-specific adjustments
    response_length = len(response.split())
    
    # Standard prompts tend to be shorter with less reasoning
    if prompt_type == "standard":
        # Small adjustment for shorter responses
        results["bias_correction"] = 0.05
        
    # Chain of Thought prompts usually have detailed reasoning
    elif prompt_type == "cot":
        # Penalty for very long responses that might ramble
        length_penalty = min(0.1, response_length / 5000)
        results["bias_correction"] = -0.05 * length_penalty
        
        # Bonus for good reasoning
        if reasoning_score > 0.7:
            results["bias_correction"] = 0.05
            
    # Hybrid CoT should balance reasoning and correctness
    elif prompt_type == "hybrid_cot":
        if reasoning_score > 0.6 and correctness_score > 0.7:
            results["bias_correction"] = 0.03
        else:
            results["bias_correction"] = 0.0
            
    # Zero-shot CoT tends to be more variable
    elif prompt_type == "zero_shot_cot":
        results["bias_correction"] = 0.0
        
    # Tree of Thought often explores multiple paths
    elif prompt_type == "tree_of_thought":
        # Reward for exploration (detected by response length)
        if response_length > 200 and reasoning_score > 0.5:
            results["bias_correction"] = 0.04
        else:
            results["bias_correction"] = 0.0
    else:
        results["bias_correction"] = 0.0
        
    # Calculate total score
    results["total_score"] = (
        (0.6 * correctness_score) + 
        (0.3 * reasoning_score) + 
        (0.1 * confidence_score) +
        results.get("bias_correction", 0.0)
    )
    
    # Ensure total score is within range
    results["total_score"] = max(0.0, min(1.0, results["total_score"]))
    
    return results 