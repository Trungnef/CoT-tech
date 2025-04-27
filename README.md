# Large Language Model (LLM) Evaluation Framework

This framework provides a toolset for evaluating the performance of Large Language Models (LLMs), with a particular focus on models and tasks related to Vietnamese language. It allows for comparing different models based on various prompt techniques and diverse evaluation metrics.

## Key Features

*   **Multi-Model Support**: Easily integrate and evaluate popular LLMs, including:
    *   **Local Models**: Llama, Qwen (require downloading the model to your machine).
    *   **API Models**: Gemini (Google), Groq.
*   **Diverse Prompting Techniques**: Support for advanced prompting methods to evaluate model capabilities in different situations:
    *   Zero-shot
    *   Few-shot (with customizable example counts, e.g., `few_shot_3`, `few_shot_5`)
    *   Chain-of-Thought (`chain_of_thought`)
    *   Self-Consistency (`cot_self_consistency_3`, `cot_self_consistency_5`)
    *   ReAct (`react`)
*   **Comprehensive Evaluation**: Provides multiple types of metrics:
    *   **Basic Metrics**: Accuracy, Latency (processing time), Token Count, Tokens Per Second.
    *   **Reasoning Evaluation**: Uses an LLM (default is Groq `llama3-70b`) to evaluate the quality of logical reasoning, mathematics, clarity, completeness, and relevance of answers (configurable).
    *   **Consistency Evaluation**: Analyzes consistency in model responses when using self-consistency techniques.
    *   **Completeness Evaluation**: Assesses whether the answer covers all aspects of the question.
    *   **Similarity Evaluation**: Calculates ROUGE, BLEU, and optionally cosine similarity of embeddings (if a model is provided) to compare answers with standard solutions.
    *   **Error Analysis**: Automatically categorizes incorrect answers into error groups (Knowledge, Reasoning, Calculation, Off-topic, etc.) using LLM.
*   **Performance Optimization**:
    *   **Quantization**: Automatically applies 4-bit quantization (BitsAndBytes) for local models to reduce GPU memory requirements.
    *   **Memory Management**: Automatically calculates and allocates GPU/CPU memory (`device_map="auto"`), supports CPU offloading.
    *   **Model Caching**:
        *   *Memory Cache*: Keeps frequently used models in RAM/VRAM for quick access.
        *   *Disk Cache*: Stores loaded and quantized models on disk for faster startup in subsequent runs (can be enabled/disabled).
    *   **API Resilience**: Automatic retry when API calls fail with exponential backoff mechanism, rate limiting management.
*   **Checkpointing**: Automatically saves evaluation state periodically or when interrupted, allowing to continue (`--resume`) from previous runs.
*   **Detailed Reporting**: Automatically generates result summary reports:
    *   **HTML**: Interactive reports with tables, statistics, and visualizations (accuracy, latency, reasoning scores, heatmap, etc.).
    *   **CSV/JSON**: Aggregated data and raw results for deeper analysis.
*   **Flexibility**: Easily extensible to support additional models, prompt types, evaluation metrics, or new question data sources.

## New Features

* **CLI Filtering**: Advanced question filtering from command line by tags, difficulty, and question types:
  * `--include-tags`: Include only questions with at least one of the specified tags
  * `--exclude-tags`: Exclude questions with any tag in this list
  * `--difficulty-levels`: Filter questions by difficulty (Easy, Medium, Hard)
  * `--question-types`: Filter questions by type (e.g., logic, math, text)

* **Advanced Error Handling**: 
  * API errors are categorized in detail and handled gracefully
  * Improved error recovery to avoid crashes when encountering errors in one part of the analysis process
  * More detailed logging for easier debugging

* **Comprehensive Configuration Checks**:
  * Clear distinction between critical errors and warnings
  * Checks that model configuration matches the selected model list
  * Validates question file format and data structure

* **Enhanced Metrics**:
  * ROUGE and BLEU metrics for text generation evaluation
  * Improved token overlap F1 with better Vietnamese language processing
  * Exact Match with flexible normalization options

## Installation

1.  **Clone Repository**:
    ```bash
    git clone https://github.com/Trungnef/CoT-tech.git
    cd llm_evaluation
    ```

2.  **Create Environment (Recommended)**:
    ```bash
    python -m venv venv
    # Linux/macOS
    source venv/bin/activate
    # Windows
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    *   **Installation**:
        ```bash
        pip install -r requirements.txt
        # Install torch compatible with your CUDA system if needed
        # Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   **Download NLTK data (if not already available)**: Required for BLEU score calculation.
        ```python
        import nltk
        nltk.download('punkt')
        ```

4.  **Environment Configuration (`.env`)**:
    *   Copy the `.env.example` file (if available) to `.env` or create a new `.env` file in the project root directory.
    *   Fill in the necessary information:
        ```dotenv
        # --- API Keys ---
        # Configure API keys in the .env file
        # Supports multiple API keys for automatic switching when quota is exhausted

        # Format: comma-separated list of API keys
        GEMINI_API_KEYS="KEY1,KEY2,KEY3"
        GROQ_API_KEYS="KEY1,KEY2,KEY3"
        OPENAI_API_KEYS="KEY1,KEY2,KEY3"

        # --- Local Model Paths ---
        # Absolute or relative path to the directory containing downloaded model and tokenizer
        # Required if using the corresponding local model
        LLAMA_MODEL_PATH="/path/to/your/llama/model"
        LLAMA_TOKENIZER_PATH="/path/to/your/llama/tokenizer" # Usually same path as model
        QWEN_MODEL_PATH="/path/to/your/qwen/model"
        QWEN_TOKENIZER_PATH="/path/to/your/qwen/tokenizer" # Usually same path as model

        # --- GPU & Memory Configuration (Optional - Adjust if needed) ---
        # Maximum GPU memory (GB) per card, framework will calculate the rest
        # MAX_GPU_MEMORY_GB=140 # Example for A100 80GB x 2
        # GPU memory (GB) reserved for system/OS
        SYSTEM_RESERVE_MEMORY_GB=2.5
        # RAM (GB) to use for CPU offloading when GPU is insufficient
        CPU_OFFLOAD_GB=24

        # --- Disk Cache Configuration (Optional) ---
        # Directory to store model cache on disk
        MODEL_CACHE_DIR="./model_cache"
        # Enable/disable disk cache (true/false)
        ENABLE_DISK_CACHE=true
        # Maximum number of models to keep in disk cache (LRU)
        MAX_CACHED_MODELS=2
        ```

5.  **Prepare Question Data**:
    *   Ensure that the questions file (`data/questions/problems.json` by default in `config.py`) exists and has a valid JSON format. Each question should be a JSON object containing at least the fields `id` (unique) and `question`.
    *   Other optional fields may include: `correct_answer`, `category`, `difficulty`, `task_type`, `examples` (for few-shot).

## Usage

### Running Evaluation

Use the `main.py` script from the command line.

**Basic Syntax**:

```bash
python main.py [--models MODEL1 MODEL2 ...] [--prompts PROMPT1 PROMPT2 ...] [OPTIONS]
```

**Main Parameters**:

*   `--models`: (Required if not using default) List of models to evaluate (e.g., `llama qwen gemini`). Model names must match keys in `config.MODEL_CONFIGS` or `.env`.
*   `--prompts`: (Required if not using default) List of prompt types to evaluate (e.g., `zero_shot few_shot_3 cot_self_consistency_5`).
*   `--questions-file`: Path to JSON file containing questions (default: `data/questions/problems.json`).
*   `--results-dir`: Directory to save results (default: `results`).
*   `--max-questions`: Maximum number of questions to evaluate from the file (default: all).
*   `--batch-size`: Batch size when processing questions (affects memory usage, default: 5).
*   `--checkpoint-frequency`: Frequency of saving checkpoints (number of questions, default: 5).
*   `--resume`: Continue from the most recent automatic checkpoint in `results/checkpoints`.
*   `--checkpoint <path>`: Continue from a specific checkpoint file.
*   `--test-run`: Run a quick test with 1 model, 1 prompt, and 2 questions.
*   `--skip-reasoning-eval`: Skip the reasoning evaluation step (even if enabled in `config.py`).
*   `--no-cache`: Disable memory cache (always reload models).
*   `--question-ids ID1 ID2 ...`: Only evaluate questions with specific IDs.
*   `--parallel`: **(Not yet implemented)** Enable parallel evaluation mode.
*   `--gpu-ids ID1 ID2 ...`: Specify GPU IDs to use (default: 0).
*   `--report-only`: Only generate a report from an existing results file without running new evaluations. Requires using with `--results-file`.
*   `--results-file <path>`: Path to results file (`.csv` or `.json`) to generate a report when using `--report-only`.
*   `--log-level`: Log level for console (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, default: `INFO`).
*   `--log-file <name>`: Specific log file name (default: auto-generated name with timestamp in `results/logs`).
*   `--debug`: Enable debug mode (equivalent to `--log-level DEBUG`).

**Examples**:

```bash
# Evaluate Llama and Gemini with zero_shot and few_shot_3 prompts for the first 50 questions
python main.py --models llama gemini --prompts zero_shot few_shot_3 --max-questions 50

# Continue evaluation from previous run
python main.py --resume

# Only generate report from saved results file
python main.py --report-only --results-file results/raw_results/evaluation_results_xxxx.csv
```

### Viewing Results

Evaluation results and reports will be saved in the `--results-dir` directory (default is `results/`) with the following sub-structure:

*   `raw_results/`: Contains raw result files in CSV and JSON formats.
*   `reports/`: Contains summary reports in HTML format.
*   `plots/`: Contains image files of charts created for the HTML report.
*   `checkpoints/`: Contains checkpoint files (`.json`).
*   `logs/`: Contains log files (`.log`).

Open the `.html` file in the `reports/` directory with a browser to view the interactive report.

## Project Structure

```
llm_evaluation/
├── core/                  # Core logic
│   ├── evaluator.py         # Evaluation coordination
│   ├── model_interface.py   # LLM interaction interface
│   ├── prompt_builder.py    # Prompt building
│   ├── result_analyzer.py   # Result analysis, metrics
│   ├── reporting.py         # Report generation
│   ├── checkpoint_manager.py# Checkpoint management
│   └── __init__.py
├── data/                  # Input data
│   └── questions/
│       └── problems.json    # Sample questions file
├── results/               # Results directory (default)
│   ├── raw_results/
│   ├── reports/
│   ├── plots/
│   ├── checkpoints/
│   └── logs/
├── tests/                 # Unit tests
│   ├── test_checkpoint_manager.py
│   └── test_utils.py
├── utils/                 # Utility modules
│   ├── config_utils.py    # Configuration utilities (dataclasses)
│   ├── data_loader.py     # Data loading (NEEDS CHECKING/IMPLEMENTATION)
│   ├── file_utils.py      # File I/O utilities
│   ├── logging_setup.py   # Logging setup
│   ├── logging_utils.py   # Specialized logging functions
│   ├── memory_utils.py    # Memory management utilities
│   ├── metrics_utils.py   # Specific metrics calculation
│   ├── text_utils.py      # Text processing
│   ├── visualization_utils.py # Chart generation (may be integrated into reporting)
│   └── __init__.py
├── model_cache/           # Disk model cache (if enabled)
├── main.py                # Main entry point
├── config.py              # Default configuration and .env loading
├── requirements.txt       # Dependencies list (NEEDS UPDATING)
├── README.md              # Documentation (this file)
└── .env                   # Environment configuration file (needs to be created)
```

## Contributing

Contributions are welcome! Please create Pull Requests or Issues on the repository.

## Author

TRUNE

<!-- ## License

Aloneee -->
