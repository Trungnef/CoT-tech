# LLM Evaluation Framework

## Overview

This project is a comprehensive evaluation framework for large language models (LLMs), designed to compare the performance of different models (Llama, Qwen, Gemini) across various prompt engineering strategies. The framework processes questions through multiple models and prompt types, generating detailed performance metrics and visualizations that highlight differences in accuracy, response time, response quality, and error rates.

Key features:
- Multi-model evaluation with standardized metrics
- Support for various prompt engineering strategies
- Parallel processing for efficient evaluation
- Comprehensive reporting with interactive visualizations
- Advanced performance metrics and quality assessments
- Memory-optimized loading for large models

## System Requirements

### Hardware
- CUDA-compatible GPU(s) with at least 16GB VRAM (24GB+ recommended)
- Minimum 16GB system RAM (32GB+ recommended)
- 100GB+ storage space for models and evaluation results

### Software
- Python 3.9+
- PyTorch 2.0+
- CUDA Toolkit 11.7+
- Required libraries:
  - transformers
  - accelerate
  - bitsandbytes
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - tqdm
  - psutil
  - rich
  - tenacity
  - python-dotenv

## GPU Configuration

The framework is designed to work with various GPU configurations:

### Single GPU Setup
- Automatically optimizes memory usage
- Supports 4-bit quantization to fit larger models on smaller GPUs
- Manages CPU offloading for memory-intensive operations

### Multi-GPU Setup
- Automatically distributes model layers across available GPUs
- Creates optimal device maps based on available memory
- Balances memory usage across GPUs (uses approximately 85% of each GPU)
- Supports CPU offloading for additional memory requirements

### Optimal Configuration
For best results:
- Use 2+ GPUs with 24GB+ VRAM each
- Configure environment variables:
  - `MAX_GPU_MEMORY_GB`: Maximum GPU memory to use per GPU
  - `SYSTEM_RESERVE_MEMORY_GB`: Memory to reserve for system operations
  - `CPU_OFFLOAD_GB`: Amount of CPU memory for model offloading

## Parallel Processing Methods

The framework leverages several parallel processing techniques:

1. **Concurrent Model Evaluation**: Evaluates different models concurrently using separate threads
2. **Batch Processing**: Processes questions in configurable batch sizes
3. **ThreadPoolExecutor**: Uses thread pools for parallel generation of responses
4. **Multi-GPU Parallelism**: Distributes model computation across available GPUs
5. **Caching Mechanisms**: Implements model caching to avoid redundant loading

Note: API models (like Gemini) are processed sequentially to avoid rate limiting issues.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-evaluation-framework.git
   cd llm-evaluation-framework
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following variables:
   ```
   QWEN_MODEL_PATH=/path/to/qwen/model
   QWEN_TOKENIZER_PATH=/path/to/qwen/tokenizer
   LLAMA_MODEL_PATH=/path/to/llama/model
   LLAMA_TOKENIZER_PATH=/path/to/llama/tokenizer
   GEMINI_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   
   MAX_GPU_MEMORY_GB=47.5
   SYSTEM_RESERVE_MEMORY_GB=2.5
   CPU_OFFLOAD_GB=24
   ```

4. Create necessary directories:
   ```bash
   mkdir -p db/questions results model_cache offload
   ```

## Project Structure

llm-evaluation-framework/
├── evaluate_models.py # Main script for model evaluation
├── model_manager.py # Module for loading and managing models
├── model_evaluator.py # Evaluation logic and metrics
├── prompts.py # Prompt engineering strategies
├── .env # Environment variables
├── requirements.txt # Required packages
├── db/ # Database of questions
│ └── questions/
│ └── problems.json # Input questions for evaluation
├── results/ # Evaluation results
│ └── YYYYMMDD_HHMMSS/ # Results organized by timestamp
│ ├── plots/ # Generated visualizations
│ ├── processed_results.csv
│ ├── model_statistics.csv
│ ├── prompt_statistics.csv
│ ├── combined_statistics.csv
│ └── comprehensive_evaluation_report.html
├── model_cache/ # Cache for loaded models
└── offload/ # Directory for model offloading
```

## Usage

### Basic Usage

Run the evaluation with default settings:

```bash
python evaluate_models.py
```

### Advanced Usage

Customize the evaluation with command-line arguments:

```bash
python evaluate_models.py \
  --questions_json db/questions/custom_problems.json \
  --models llama qwen gemini \
  --prompt_types standard cot hybrid_cot zero_shot_cot \
  --batch_size 10 \
  --max_questions 100 \
  --results_dir custom_results \
  --use_4bit \
  --max_workers 3
```

### Important Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--questions_json` | Path to JSON file with questions | `db/questions/problems.json` |
| `--models` | Models to evaluate | `llama qwen gemini` |
| `--prompt_types` | Prompt types to evaluate | `standard cot hybrid_cot zero_shot_cot` |
| `--batch_size` | Questions to process in parallel | `10` |
| `--max_questions` | Maximum questions to evaluate | `None` (all questions) |
| `--results_dir` | Directory to save results | `results` |
| `--use_4bit` | Use 4-bit quantization | `True` |
| `--max_workers` | Maximum parallel workers | `3` |
| `--resume` | Resume from existing results | `False` |
| `--results_file` | Path to results file to resume from | `None` |

## Running Tests

To run a quick test with a small subset of questions:

```bash
python evaluate_models.py --max_questions 5 --models llama --prompt_types standard
```

For a more comprehensive test:

```bash
python evaluate_models.py --models llama qwen gemini --prompt_types standard cot --max_questions 20
```

## Visualizations and Results

The framework generates a comprehensive HTML report with numerous visualizations for in-depth analysis:

### Core Visualizations
- Response time distributions by model and prompt type
- Error rate heatmaps
- Response length comparisons
- Processing speed comparisons
- Response time vs. length scatter plots

### Advanced Visualizations
- 3D surface plots for model performance
- Interactive sunburst charts for error distribution
- Parallel categories plots for variable relationships
- Time series with confidence intervals
- Quality metrics radar charts
- Stacked area charts for response type distribution

### Performance Metrics
- Response quality scores
- Complexity metrics
- Coherence assessments
- Efficiency scores
- Quality-speed indices
- Cost-efficiency comparisons
- Time consistency measurements
- Useful content ratios

The report is fully interactive, with tooltips, filters, and drill-down capabilities for deeper analysis.

## Example Report

After running an evaluation, access the comprehensive report at:

results/[timestamp]/comprehensive_evaluation_report_[timestamp].html

This report contains all visualizations, metrics, and analyses in an easy-to-navigate interface.

---

For issues, feature requests, or contributions, please open an issue or pull request in the repository.