"""
Prompt templates for querying language models with different reasoning strategies.
"""

from src.prompts.base_prompts import (
    standard_prompt,
    chain_of_thought_prompt
)

from src.prompts.advanced_prompts import (
    hybrid_cot_prompt,
    zero_shot_cot_prompt,
    tree_of_thought_prompt
)

__all__ = [
    'standard_prompt',
    'chain_of_thought_prompt',
    'hybrid_cot_prompt',
    'zero_shot_cot_prompt',
    'tree_of_thought_prompt'
]
