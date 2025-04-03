"""
Tiện ích quản lý cấu hình cho framework đánh giá LLM.
Cung cấp các hàm đọc, ghi, cập nhật cấu hình và xác thực cấu hình.
"""

import os
import yaml
import json
import copy
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
import jsonschema
from jsonschema import validate

from .logging_utils import get_logger
from .file_utils import ensure_dir, load_yaml, save_yaml, load_json, save_json

logger = get_logger(__name__)

# Schema cơ bản cho cấu hình
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "models": {
            "type": "object",
            "patternProperties": {
                "^.*$": {
                    "type": "object",
                    "properties": {
                        "api_type": {"type": "string"},
                        "api_key": {"type": "string"},
                        "api_base": {"type": "string"},
                        "max_tokens": {"type": "integer"},
                        "model_name": {"type": "string"},
                        "temperature": {"type": "number"},
                        "rate_limit": {"type": "number"}
                    },
                    "required": ["api_type", "model_name"]
                }
            }
        },
        "prompt_types": {
            "type": "object",
            "patternProperties": {
                "^.*$": {
                    "type": "object",
                    "properties": {
                        "template": {"type": "string"},
                        "examples": {"type": "array"},
                        "format": {"type": "string"},
                        "options": {"type": "object"}
                    },
                    "required": ["template"]
                }
            }
        },
        "evaluation": {
            "type": "object",
            "properties": {
                "metrics": {"type": "array", "items": {"type": "string"}},
                "reasoning_evaluation": {"type": "object"},
                "output_dir": {"type": "string"},
                "checkpoint_dir": {"type": "string"},
                "parallel": {"type": "boolean"},
                "max_workers": {"type": "integer"},
                "timeout": {"type": "number"}
            }
        },
        "logging": {
            "type": "object",
            "properties": {
                "level": {"type": "string"},
                "file": {"type": "string"},
                "console": {"type": "boolean"},
                "format": {"type": "string"}
            }
        }
    },
    "required": ["version", "models", "prompt_types", "evaluation"]
}

@dataclass
class ModelConfig:
    """Cấu hình cho một mô hình LLM."""
    api_type: str
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    rate_limit: float = 1.0
    timeout: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dict."""
        return asdict(self)

@dataclass
class PromptConfig:
    """Cấu hình cho một loại prompt."""
    template: str
    examples: List[Dict[str, Any]] = field(default_factory=list)
    format: str = "text"
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dict."""
        return asdict(self)

@dataclass
class EvaluationConfig:
    """Cấu hình cho quá trình đánh giá."""
    metrics: List[str] = field(default_factory=list)
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    parallel: bool = False
    max_workers: int = 4
    timeout: float = 120.0
    reasoning_evaluation: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dict."""
        return asdict(self)

@dataclass
class LoggingConfig:
    """Cấu hình cho logging."""
    level: str = "INFO"
    file: Optional[str] = None
    console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dict."""
        return asdict(self)

@dataclass
class Config:
    """Cấu hình tổng thể cho framework đánh giá LLM."""
    version: str = "1.0.0"
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    prompt_types: Dict[str, PromptConfig] = field(default_factory=dict)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dict."""
        result = {
            "version": self.version,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "prompt_types": {k: v.to_dict() for k, v in self.prompt_types.items()},
            "evaluation": self.evaluation.to_dict(),
            "logging": self.logging.to_dict()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Tạo đối tượng Config từ dict."""
        # Xử lý models
        models = {}
        for k, v in data.get("models", {}).items():
            models[k] = ModelConfig(**v)
        
        # Xử lý prompt_types
        prompt_types = {}
        for k, v in data.get("prompt_types", {}).items():
            prompt_types[k] = PromptConfig(**v)
        
        # Xử lý evaluation
        evaluation = EvaluationConfig(**data.get("evaluation", {}))
        
        # Xử lý logging
        logging_config = LoggingConfig(**data.get("logging", {}))
        
        return cls(
            version=data.get("version", "1.0.0"),
            models=models,
            prompt_types=prompt_types,
            evaluation=evaluation,
            logging=logging_config
        )

def load_config(config_path: str) -> Config:
    """
    Đọc cấu hình từ file.
    
    Args:
        config_path: Đường dẫn đến file cấu hình
        
    Returns:
        Đối tượng Config
    
    Raises:
        FileNotFoundError: Nếu file không tồn tại
        ValueError: Nếu cấu hình không hợp lệ
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
    
    try:
        # Xác định định dạng file
        ext = os.path.splitext(config_path)[1].lower()
        
        if ext in ['.yaml', '.yml']:
            config_data = load_yaml(config_path)
        elif ext == '.json':
            config_data = load_json(config_path)
        else:
            raise ValueError(f"Định dạng file cấu hình không được hỗ trợ: {ext}")
        
        # Xác thực cấu hình
        validate_config(config_data)
        
        # Chuyển đổi sang đối tượng Config
        config = Config.from_dict(config_data)
        
        logger.info(f"Đã tải cấu hình từ: {config_path}")
        
        return config
    
    except Exception as e:
        logger.error(f"Lỗi khi tải cấu hình: {str(e)}")
        raise

def save_config(config: Config, config_path: str) -> bool:
    """
    Lưu cấu hình vào file.
    
    Args:
        config: Đối tượng Config
        config_path: Đường dẫn đến file cấu hình
        
    Returns:
        True nếu lưu thành công, False nếu thất bại
    """
    try:
        # Chuyển đổi sang dict
        config_data = config.to_dict()
        
        # Xác thực cấu hình
        validate_config(config_data)
        
        # Xác định định dạng file
        ext = os.path.splitext(config_path)[1].lower()
        
        # Tạo thư mục nếu cần
        ensure_dir(os.path.dirname(config_path))
        
        if ext in ['.yaml', '.yml']:
            save_yaml(config_data, config_path)
        elif ext == '.json':
            save_json(config_data, config_path)
        else:
            raise ValueError(f"Định dạng file cấu hình không được hỗ trợ: {ext}")
        
        logger.info(f"Đã lưu cấu hình vào: {config_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi lưu cấu hình: {str(e)}")
        return False

def validate_config(config_data: Dict[str, Any]) -> bool:
    """
    Xác thực cấu hình theo schema.
    
    Args:
        config_data: Dữ liệu cấu hình
        
    Returns:
        True nếu hợp lệ
        
    Raises:
        jsonschema.exceptions.ValidationError: Nếu cấu hình không hợp lệ
    """
    try:
        validate(instance=config_data, schema=CONFIG_SCHEMA)
        return True
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Cấu hình không hợp lệ: {str(e)}")
        raise

def merge_configs(base_config: Config, override_config: Config) -> Config:
    """
    Gộp hai cấu hình, ưu tiên cấu hình ghi đè.
    
    Args:
        base_config: Cấu hình cơ sở
        override_config: Cấu hình ghi đè
        
    Returns:
        Cấu hình mới sau khi gộp
    """
    # Chuyển đổi sang dict
    base_dict = base_config.to_dict()
    override_dict = override_config.to_dict()
    
    # Gộp các dict
    merged_dict = _deep_merge(base_dict, override_dict)
    
    # Chuyển đổi lại sang đối tượng Config
    return Config.from_dict(merged_dict)

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gộp hai dict một cách đệ quy.
    
    Args:
        base: Dict cơ sở
        override: Dict ghi đè
        
    Returns:
        Dict mới sau khi gộp
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result

def update_config_from_env(config: Config, prefix: str = "LLM_EVAL_") -> Config:
    """
    Cập nhật cấu hình từ biến môi trường.
    
    Args:
        config: Đối tượng Config
        prefix: Tiền tố cho biến môi trường
        
    Returns:
        Cấu hình đã cập nhật
    """
    # Chuyển đổi sang dict
    config_dict = config.to_dict()
    
    # Lặp qua các biến môi trường
    for env_var, env_value in os.environ.items():
        if not env_var.startswith(prefix):
            continue
        
        # Loại bỏ tiền tố
        env_var = env_var[len(prefix):].lower()
        
        # Phân tách các phần
        parts = env_var.split("_")
        
        # Cập nhật cấu hình
        _update_dict_from_path(config_dict, parts, env_value)
    
    # Chuyển đổi lại sang đối tượng Config
    return Config.from_dict(config_dict)

def _update_dict_from_path(d: Dict[str, Any], path: List[str], value: Any) -> None:
    """
    Cập nhật dict theo đường dẫn.
    
    Args:
        d: Dict cần cập nhật
        path: Đường dẫn trong dict
        value: Giá trị mới
    """
    if not path:
        return
    
    if len(path) == 1:
        key = path[0]
        # Chuyển đổi giá trị nếu cần
        if key in d and isinstance(d[key], bool):
            d[key] = value.lower() in ["true", "yes", "1", "y"]
        elif key in d and isinstance(d[key], int):
            d[key] = int(value)
        elif key in d and isinstance(d[key], float):
            d[key] = float(value)
        else:
            d[key] = value
    else:
        key = path[0]
        if key not in d:
            d[key] = {}
        _update_dict_from_path(d[key], path[1:], value)

def create_default_config() -> Config:
    """
    Tạo cấu hình mặc định.
    
    Returns:
        Đối tượng Config với các giá trị mặc định
    """
    # Tạo model config mẫu
    models = {
        "gpt-4": ModelConfig(
            api_type="openai",
            model_name="gpt-4",
            max_tokens=1024,
            temperature=0.7,
            rate_limit=3.0
        ),
        "gpt-3.5-turbo": ModelConfig(
            api_type="openai",
            model_name="gpt-3.5-turbo",
            max_tokens=1024,
            temperature=0.7,
            rate_limit=20.0
        )
    }
    
    # Tạo prompt config mẫu
    prompt_types = {
        "zero-shot": PromptConfig(
            template="Hãy trả lời câu hỏi sau:\n\n{question}",
            format="text"
        ),
        "few-shot": PromptConfig(
            template="Dưới đây là một số ví dụ:\n\n{examples}\n\nHãy trả lời câu hỏi sau:\n\n{question}",
            examples=[],
            format="text"
        ),
        "cot": PromptConfig(
            template="Hãy suy nghĩ từng bước để trả lời câu hỏi sau:\n\n{question}",
            format="text",
            options={"reasoning": True}
        )
    }
    
    # Tạo evaluation config
    evaluation = EvaluationConfig(
        metrics=["accuracy", "f1", "latency"],
        output_dir="outputs",
        checkpoint_dir="checkpoints",
        parallel=False,
        max_workers=4,
        reasoning_evaluation={
            "enabled": False,
            "model": "gpt-4",
            "criteria": ["correctness", "relevance", "coherence", "depth"],
            "sample_size": 50
        }
    )
    
    # Tạo logging config
    logging_config = LoggingConfig(
        level="INFO",
        file="logs/llm_evaluation.log",
        console=True
    )
    
    return Config(
        version="1.0.0",
        models=models,
        prompt_types=prompt_types,
        evaluation=evaluation,
        logging=logging_config
    )

def get_config_example() -> str:
    """
    Lấy ví dụ cấu hình YAML.
    
    Returns:
        Chuỗi cấu hình YAML mẫu
    """
    config = create_default_config()
    config_dict = config.to_dict()
    
    # Chuyển đổi sang YAML
    import yaml
    yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
    
    return yaml_str 