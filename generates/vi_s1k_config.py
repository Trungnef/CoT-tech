"""
Vi-S1K Configuration Module
Cấu hình cho hệ thống dịch Vi-S1K
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class TranslatorConfig:
    """Cấu hình cho translator"""
    
    backend: str = "gemini"  # gemini, openai, local
    use_cache: bool = True
    cache_dir: str = "./translation_cache"
    
    # Gemini config
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = "gemini-2.5-flash-lite"
    
    # OpenAI config
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-3.5-turbo"
    
    # Local model config
    local_model_path: Optional[str] = os.getenv("LOCAL_MODEL_PATH")
    
    # Translation settings
    temperature: float = 0.3
    max_tokens: int = 2000


@dataclass
class DatasetConfig:
    """Cấu hình cho dataset"""
    
    # S1K dataset config
    s1k_dataset_name: str = "simplescaling/s1K-1.1"
    s1k_split: str = "train"
    max_samples: Optional[int] = None
    
    # Language config
    source_language: str = "English"
    target_language: str = "Vietnamese"
    
    # Question fields mapping
    question_field: str = "problem"  # hoặc "question"
    answer_field: str = "solution"   # hoặc "answer"
    domain_field: str = "category"   # hoặc "domain"
    difficulty_field: str = "level"  # hoặc "difficulty"


@dataclass
class QualityConfig:
    """Cấu hình cho kiểm tra chất lượng"""
    
    enable_quality_check: bool = True
    
    # Length check
    min_translated_length: int = 3
    max_length_ratio: float = 1.5
    min_length_ratio: float = 0.5
    
    # Vietnamese character check
    min_vietnamese_chars: int = 2
    
    # Quality score thresholds
    min_quality_score: float = 0.3  # Câu dịch có chất lượng kém sẽ bị cảnh báo
    high_quality_threshold: float = 0.8


@dataclass
class OutputConfig:
    """Cấu hình cho output"""
    
    # Thư mục output
    output_dir: str = "./results/vi_s1k"
    
    # Format output
    formats: List[str] = None  # ["json", "jsonl", "csv"]
    
    # Tên file output
    benchmark_filename: str = "vi_s1k_benchmark"
    stats_filename: str = "statistics.json"
    
    # Pretty print JSON
    json_indent: int = 2
    json_ensure_ascii: bool = False
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ["json", "jsonl"]


@dataclass
class PromptConfig:
    """Cấu hình cho prompts dịch"""
    
    # Prompt templates
    zero_shot_template: str = """Translate the following {source_lang} text to {target_lang}. 
Keep the meaning and tone intact. Only output the translation, nothing else.

{source_lang} text:
{text}

{target_lang} translation:"""

    few_shot_template: str = """Translate {source_lang} text to {target_lang}. 
Keep the meaning and tone intact. Only output the translation.

Examples:
{examples}

{source_lang} text to translate:
{text}

{target_lang} translation:"""

    domain_specific_template: str = """You are a professional {domain} translator specializing in {source_lang} to {target_lang} translation.
{domain_guidance}

Translate the following text to {target_lang}. Only output the translation without any explanation.

Text to translate:
{text}

Translation:"""

    # Domain-specific guidance
    domain_guidance: Dict[str, str] = None
    
    def __post_init__(self):
        if self.domain_guidance is None:
            self.domain_guidance = {
                "mathematics": "Mathematical terminology and symbols should be preserved. Ensure numbers and equations are correct.",
                "academic": "Maintain formal academic tone and preserve technical terms where appropriate.",
                "general": "Maintain natural and fluent language."
            }


# Default configurations
DEFAULT_TRANSLATOR_CONFIG = TranslatorConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_QUALITY_CONFIG = QualityConfig()
DEFAULT_OUTPUT_CONFIG = OutputConfig()
DEFAULT_PROMPT_CONFIG = PromptConfig()


class ViS1KConfig:
    """Main configuration class for Vi-S1K system"""
    
    def __init__(self,
                 translator: TranslatorConfig = None,
                 dataset: DatasetConfig = None,
                 quality: QualityConfig = None,
                 output: OutputConfig = None,
                 prompt: PromptConfig = None):
        
        self.translator = translator or DEFAULT_TRANSLATOR_CONFIG
        self.dataset = dataset or DEFAULT_DATASET_CONFIG
        self.quality = quality or DEFAULT_QUALITY_CONFIG
        self.output = output or DEFAULT_OUTPUT_CONFIG
        self.prompt = prompt or DEFAULT_PROMPT_CONFIG
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ViS1KConfig':
        """Tạo config từ dictionary"""
        return cls(
            translator=TranslatorConfig(**config_dict.get("translator", {})),
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            quality=QualityConfig(**config_dict.get("quality", {})),
            output=OutputConfig(**config_dict.get("output", {})),
            prompt=PromptConfig(**config_dict.get("prompt", {}))
        )
    
    def to_dict(self) -> Dict:
        """Chuyển config thành dictionary"""
        return {
            "translator": self.translator.__dict__,
            "dataset": self.dataset.__dict__,
            "quality": self.quality.__dict__,
            "output": self.output.__dict__,
            "prompt": self.prompt.__dict__
        }
    
    @staticmethod
    def load_from_file(config_path: str) -> 'ViS1KConfig':
        """Tải config từ file JSON"""
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return ViS1KConfig.from_dict(config_dict)
    
    def save_to_file(self, config_path: str):
        """Lưu config vào file JSON"""
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


# Preset configurations cho các use cases khác nhau

def get_lightweight_config() -> ViS1KConfig:
    """Cấu hình nhẹ (CPU-friendly)"""
    return ViS1KConfig(
        translator=TranslatorConfig(
            backend="gemini",
            temperature=0.3,
            max_tokens=500
        ),
        dataset=DatasetConfig(max_samples=100),
        quality=QualityConfig(enable_quality_check=False),
        output=OutputConfig(formats=["json"])
    )


def get_production_config() -> ViS1KConfig:
    """Cấu hình production (high quality)"""
    return ViS1KConfig(
        translator=TranslatorConfig(
            backend="gemini",
            temperature=0.1,  # Thấp hơn để ổn định
            max_tokens=2000
        ),
        dataset=DatasetConfig(),
        quality=QualityConfig(
            enable_quality_check=True,
            min_quality_score=0.7
        ),
        output=OutputConfig(
            formats=["json", "jsonl", "csv"]
        )
    )


def get_experimental_config() -> ViS1KConfig:
    """Cấu hình thử nghiệm (testing)"""
    return ViS1KConfig(
        translator=TranslatorConfig(
            backend="gemini",
            temperature=0.5,
        ),
        dataset=DatasetConfig(max_samples=10),
        quality=QualityConfig(enable_quality_check=True),
        output=OutputConfig(
            output_dir="./results/vi_s1k_test",
            formats=["json"]
        )
    )


def get_local_model_config() -> ViS1KConfig:
    """Cấu hình sử dụng local model"""
    return ViS1KConfig(
        translator=TranslatorConfig(
            backend="local",
            local_model_path="./models/llama2",
        ),
        dataset=DatasetConfig(),
        quality=QualityConfig(enable_quality_check=True),
        output=OutputConfig()
    )


# Configuration presets
PRESETS = {
    "lightweight": get_lightweight_config,
    "production": get_production_config,
    "experimental": get_experimental_config,
    "local": get_local_model_config,
}


def get_config(preset: str = "production") -> ViS1KConfig:
    """
    Lấy configuration với preset
    
    Args:
        preset: "lightweight", "production", "experimental", "local"
    
    Returns:
        ViS1KConfig instance
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    
    return PRESETS[preset]()


# Example configuration (for documentation)
EXAMPLE_CONFIG = {
    "translator": {
        "backend": "gemini",
        "use_cache": True,
        "cache_dir": "./translation_cache",
        "temperature": 0.3,
        "max_tokens": 2000
    },
    "dataset": {
        "s1k_dataset_name": "simplescaling/s1K-1.1",
        "s1k_split": "train",
        "max_samples": 100,
        "source_language": "English",
        "target_language": "Vietnamese"
    },
    "quality": {
        "enable_quality_check": True,
        "min_quality_score": 0.3
    },
    "output": {
        "output_dir": "./results/vi_s1k",
        "formats": ["json", "jsonl"]
    }
}
