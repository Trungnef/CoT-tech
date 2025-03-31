"""
Base prompt templates for querying language models with different reasoning strategies.
"""

def standard_prompt(query, task_type="math"):
    """
    Create a standardized prompt for different types of tasks.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', 'coding', 'writing', etc.)
    
    Returns:
    - str: Formatted prompt ready to be sent to an LLM
    """
    prompts = {
        "general": f"""Answer the following question thoroughly and accurately:

Question: {query}

Answer: """,
        
        "math": f"""Solve the following mathematics problem step by step:

Problem: {query}

Solution: """,
        
        "classical_problem": f"""Solve the following classical problem in detail and systematically:

Problem: {query}

Solution: """,

        "vietnamese": f"""Hãy giải bài toán sau một cách chi tiết và có hệ thống:

Bài toán: {query}

Lời giải: """
    }
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        return prompts["vietnamese"]
    
    # Default to general if task_type not found
    return prompts.get(task_type.lower(), prompts["general"])


def chain_of_thought_prompt(query, task_type="math"):
    """
    Create a Chain-of-Thought prompt that explicitly asks the model
    to reason step by step before giving the final answer.
    
    Parameters:
    - query (str): The user's question or problem statement
    - task_type (str): Type of task ('general', 'math', 'coding', 'writing', etc.)
    
    Returns:
    - str: Formatted CoT prompt ready to be sent to an LLM
    """
    prompts = {
        "general": f"""Answer the following question step by step. First, break down the problem, analyze each component, and then provide the final answer.

Question: {query}

Thought process:
1) """,
        
        "math": f"""Solve the following mathematics problem step by step. First analyze what is being asked, identify the key information, plan your approach, then execute each step until you reach the final answer.

Problem: {query}

Thought process:
1) Let me understand what the problem is asking.
2) """,
        
        "classical_problem": f"""Solve the following classical problem using step-by-step reasoning. First, analyze the problem, identify the important information, plan your approach, then execute each calculation step until you reach the final result.

Problem: {query}

Reasoning process:
1) I will carefully analyze the problem to understand what is being asked.
2) """,

        "vietnamese": f"""Hãy giải bài toán sau đây bằng cách suy luận từng bước một. Đầu tiên, phân tích bài toán, xác định thông tin quan trọng, lên kế hoạch giải quyết, sau đó thực hiện từng bước tính toán cho đến khi đạt được kết quả cuối cùng.

Bài toán: {query}

Quá trình suy luận:
1) Tôi sẽ phân tích kỹ bài toán để hiểu yêu cầu đặt ra.
2) """
    }
    
    # Detect language and use Vietnamese prompt if the query contains Vietnamese characters
    if any('\u00C0' <= c <= '\u1EF9' for c in query):
        return prompts["vietnamese"]
    
    # Default to general if task_type not found
    return prompts.get(task_type.lower(), prompts["general"]) 