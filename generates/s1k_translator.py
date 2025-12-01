"""
S1K Dataset Translation and Vietnamese Benchmark Builder
Converts simplescaling/s1K-1.1 dataset to Vi-S1K (Vietnamese S1K)
"""

import json
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TranslatedQuestion:
    """Represents a translated S1K question - Vietnamese only (13 columns total)."""
    # Vietnamese translations (8 columns)
    vietnamese_question: str
    vietnamese_solution: str
    vietnamese_gemini_thinking_trajectory: str
    vietnamese_gemini_attempt: str
    vietnamese_deepseek_thinking_trajectory: str
    vietnamese_deepseek_attempt: str
    vietnamese_gemini_grade_reason: str
    vietnamese_deepseek_grade_reason: str
    
    # Labels to preserve (4 columns)
    cot_type: str  # Keep: "math"
    source_type: str  # Keep: "qq8933/AIME_1983_2024"
    gemini_grade: str  # Keep: "Yes"/"No"
    deepseek_grade: str  # Keep: "Yes"/"No"
    
    # Metadata (1 column)
    metadata: str  # JSON string - keep as-is


class S1KDatasetLoader:
    """Load and process S1K dataset from Hugging Face."""
    
    def __init__(self):
        try:
            from datasets import load_dataset
            self.load_dataset = load_dataset
        except ImportError:
            raise ImportError("datasets package is required. Install with: pip install datasets")
    
    def load_s1k(self, config_name: str = "default", split: str = "train", 
                 max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load S1K dataset from Hugging Face."""
        try:
            logger.info(f"Loading S1K dataset (config={config_name}, split={split})")
            dataset = self.load_dataset("simplescaling/s1K-1.1", config_name)
            
            if split in dataset:
                data = dataset[split]
            else:
                # Try to use the first available split
                available_splits = list(dataset.keys())
                logger.warning(f"Split '{split}' not found. Available splits: {available_splits}")
                data = dataset[available_splits[0]]
            
            # Convert to list and limit samples if needed
            data_list = [dict(item) for item in data]
            
            if max_samples:
                data_list = data_list[:max_samples]
            
            logger.info(f"Loaded {len(data_list)} samples from S1K dataset")
            return data_list
        
        except Exception as e:
            logger.error(f"Failed to load S1K dataset: {e}")
            raise
    
    @staticmethod
    def extract_question_fields(item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields needed for translation from S1K dataset item."""
        return {
            # Fields to translate
            "question": item.get("question", ""),
            "solution": item.get("solution", ""),
            "gemini_thinking_trajectory": item.get("gemini_thinking_trajectory", ""),
            "gemini_attempt": item.get("gemini_attempt", ""),
            "deepseek_thinking_trajectory": item.get("deepseek_thinking_trajectory", ""),
            "deepseek_attempt": item.get("deepseek_attempt", ""),
            "gemini_grade_reason": item.get("gemini_grade_reason", ""),
            "deepseek_grade_reason": item.get("deepseek_grade_reason", ""),
            # Fields to preserve
            "cot_type": item.get("cot_type", ""),
            "source_type": item.get("source_type", ""),
            "gemini_grade": item.get("gemini_grade", ""),
            "deepseek_grade": item.get("deepseek_grade", ""),
            "metadata": item.get("metadata", "{}"),  # Keep as JSON string
        }


class VietnameseBenchmarkBuilder:
    """Build Vi-S1K benchmark by translating S1K dataset."""
    
    def __init__(self, translator, language_pair: tuple = ("English", "Vietnamese")):
        """
        Initialize benchmark builder.
        
        Args:
            translator: Instance of a translator class (GeminiTranslator, OpenAITranslator, etc.)
            language_pair: Tuple of (source_language, target_language)
        """
        self.translator = translator
        self.source_lang, self.target_lang = language_pair
        self.translated_items = []
        self.failed_items = []
        self.quality_scores = {}
    
    def translate_item(self, item: Dict[str, Any]) -> Optional[TranslatedQuestion]:
        """Translate required fields from S1K item to Vietnamese (13 columns only)."""
        try:
            # Fields to translate (8)
            fields_to_translate = {
                'question': item.get("question", ""),
                'solution': item.get("solution", ""),
                'gemini_thinking_trajectory': item.get("gemini_thinking_trajectory", ""),
                'gemini_attempt': item.get("gemini_attempt", ""),
                'deepseek_thinking_trajectory': item.get("deepseek_thinking_trajectory", ""),
                'deepseek_attempt': item.get("deepseek_attempt", ""),
                'gemini_grade_reason': item.get("gemini_grade_reason", ""),
                'deepseek_grade_reason': item.get("deepseek_grade_reason", ""),
            }
            
            # Translate each field
            translated_fields = {}
            for field_name, field_value in fields_to_translate.items():
                if field_value:  # Only translate non-empty fields
                    translated_fields[f'vietnamese_{field_name}'] = self.translator.translate(
                        field_value,
                        source_lang=self.source_lang,
                        target_lang=self.target_lang,
                        domain="mathematics"  # S1K is primarily mathematics
                    )
                else:
                    translated_fields[f'vietnamese_{field_name}'] = None
            
            # Create TranslatedQuestion with only 13 columns:
            # 8 Vietnamese translations + 4 labels + 1 metadata
            translated = TranslatedQuestion(
                # Vietnamese translations (8)
                vietnamese_question=translated_fields['vietnamese_question'],
                vietnamese_solution=translated_fields['vietnamese_solution'],
                vietnamese_gemini_thinking_trajectory=translated_fields['vietnamese_gemini_thinking_trajectory'],
                vietnamese_gemini_attempt=translated_fields['vietnamese_gemini_attempt'],
                vietnamese_deepseek_thinking_trajectory=translated_fields['vietnamese_deepseek_thinking_trajectory'],
                vietnamese_deepseek_attempt=translated_fields['vietnamese_deepseek_attempt'],
                vietnamese_gemini_grade_reason=translated_fields['vietnamese_gemini_grade_reason'],
                vietnamese_deepseek_grade_reason=translated_fields['vietnamese_deepseek_grade_reason'],
                # Labels to preserve (4)
                cot_type=item.get("cot_type", ""),
                source_type=item.get("source_type", ""),
                gemini_grade=item.get("gemini_grade", ""),
                deepseek_grade=item.get("deepseek_grade", ""),
                # Metadata (1)
                metadata=item.get("metadata", "{}")
            )
            
            return translated
        
        except Exception as e:
            logger.error(f"Failed to translate item: {e}")
            self.failed_items.append(item)
            return None
    
    def build_benchmark(self, dataset: List[Dict[str, Any]], 
                       quality_checker: Optional[Callable] = None,
                       show_progress: bool = True) -> List[TranslatedQuestion]:
        """
        Build Vietnamese benchmark by translating dataset.
        
        Args:
            dataset: List of question items from S1K
            quality_checker: Optional function to check translation quality
            show_progress: Show progress bar
        
        Returns:
            List of translated questions
        """
        logger.info(f"Starting translation of {len(dataset)} items")
        
        iterator = tqdm(dataset, disable=not show_progress, desc="Translating dataset")
        
        for item in iterator:
            translated = self.translate_item(item)
            
            if translated:
                # Optional quality check
                if quality_checker:
                    translated.quality_score = quality_checker(translated)
                
                self.translated_items.append(translated)
        
        logger.info(f"Successfully translated {len(self.translated_items)} items")
        logger.info(f"Failed translations: {len(self.failed_items)}")
        
        return self.translated_items
    
    def save_benchmark(self, output_path: str, format: str = "json"):
        """
        Save benchmark to file.
        
        Args:
            output_path: Path to save benchmark
            format: Output format ('json', 'jsonl', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            self._save_json(output_path)
        elif format == "jsonl":
            self._save_jsonl(output_path)
        elif format == "csv":
            self._save_csv(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Benchmark saved to {output_path}")
    
    def _save_json(self, output_path: Path):
        """Save as JSON format."""
        data = {
            "metadata": {
                "source": "simplescaling/s1K-1.1",
                "target_language": self.target_lang,
                "benchmark_name": "Vi-S1K",
                "total_columns": 13,
                "columns": {
                    "vietnamese_translations": 8,
                    "labels_preserved": 4,
                    "metadata": 1
                },
                "creation_date": datetime.now().isoformat(),
                "total_items": len(self.translated_items),
                "translation_model": getattr(self.translator, 'model_name', 'unknown'),
                "translator_stats": self.translator.get_stats()
            },
            "items": [asdict(item) for item in self.translated_items]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_jsonl(self, output_path: Path):
        """Save as JSONL (one JSON per line) format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.translated_items:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + '\n')
    
    def _save_csv(self, output_path: Path):
        """Save as CSV format."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas package is required for CSV export. Install with: pip install pandas")
        
        data_list = [asdict(item) for item in self.translated_items]
        df = pd.DataFrame(data_list)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get translation statistics."""
        total = len(self.translated_items) + len(self.failed_items)
        success_rate = len(self.translated_items) / total if total > 0 else 0
        
        return {
            "total_items": total,
            "successful": len(self.translated_items),
            "failed": len(self.failed_items),
            "success_rate": success_rate,
            "output_columns": 13,
            "columns_breakdown": {
                "vietnamese_translations": 8,
                "labels_preserved": 4,
                "metadata": 1
            },
            "translator_stats": self.translator.get_stats()
        }


class QualityChecker:
    """Check quality of translations."""
    
    @staticmethod
    def simple_check(translated_item: TranslatedQuestion) -> float:
        """
        Simple quality check based on Vietnamese character content.
        Returns score between 0 and 1.
        """
        if not translated_item.vietnamese_question:
            return 0.0
        
        # Check if translation contains actual vietnamese characters
        text = translated_item.vietnamese_question
        vietnamese_chars = len([c for c in text if ord(c) >= 0x00C0])  # Vietnamese diacritics
        
        # Score based on vietnamese character density
        if len(text) == 0:
            return 0.0
        
        vietnamese_score = vietnamese_chars / len(text)
        
        # Expected: at least 30% of characters should be vietnamese (with diacritics)
        if vietnamese_score >= 0.3:
            return 0.9
        elif vietnamese_score >= 0.15:
            return 0.7
        else:
            return 0.3
    
    @staticmethod
    def llm_check(translated_item: TranslatedQuestion, evaluator=None) -> float:
        """
        Use an LLM to evaluate translation quality.
        Requires an evaluator instance.
        """
        if evaluator is None:
            return QualityChecker.simple_check(translated_item)
        
        prompt = f"""Evaluate the quality of this Vietnamese translation.
Original: {translated_item.question}
Translation: {translated_item.vietnamese_question}

Rate from 0-10 where:
- 10: Perfect translation, preserves all meaning and context
- 7-9: Good translation with minor issues
- 4-6: Acceptable but has some meaning loss or grammar issues
- 1-3: Poor translation with significant issues
- 0: Not a real translation or completely wrong

Respond with only the number."""
        
        try:
            # This would need an evaluator implementation
            # For now, return simple check
            return QualityChecker.simple_check(translated_item)
        except Exception as e:
            logger.warning(f"LLM quality check failed: {e}")
            return QualityChecker.simple_check(translated_item)
