"""
Translation utilities for converting datasets to Vietnamese.
Supports various LLM backends and handles batch translation with caching.
"""

import json
import os
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
from tqdm import tqdm
import logging

# Setup logging
logger = logging.getLogger(__name__)


class TranslationCache:
    """Simple file-based cache for translations to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = "./translation_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key from text and language pair."""
        key_str = f"{source_lang}_{target_lang}_{text}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path from key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Retrieve translation from cache."""
        cache_key = self._get_cache_key(text, source_lang, target_lang)
        cache_file = self._get_cache_file(cache_key)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("translation")
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}")
        
        return None
    
    def set(self, text: str, translation: str, source_lang: str, target_lang: str):
        """Store translation in cache."""
        cache_key = self._get_cache_key(text, source_lang, target_lang)
        cache_file = self._get_cache_file(cache_key)
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "original": text,
                    "translation": translation,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "timestamp": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")


class TranslationPromptBuilder:
    """Builds translation prompts for different LLM models."""
    
    @staticmethod
    def build_zero_shot_prompt(text: str, source_lang: str = "English", 
                               target_lang: str = "Vietnamese") -> str:
        """Build a zero-shot translation prompt."""
        return f"""Translate the following {source_lang} text to {target_lang}. 
Keep the meaning and tone intact. Only output the translation, nothing else.

{source_lang} text:
{text}

{target_lang} translation:"""
    
    @staticmethod
    def build_few_shot_prompt(text: str, examples: Optional[List[Dict[str, str]]] = None,
                              source_lang: str = "English", target_lang: str = "Vietnamese") -> str:
        """Build a few-shot translation prompt with examples."""
        if examples is None:
            examples = []
        
        examples_text = ""
        for i, ex in enumerate(examples, 1):
            examples_text += f"\nExample {i}:\n{source_lang}: {ex.get('source', '')}\n{target_lang}: {ex.get('target', '')}"
        
        prompt = f"""Translate {source_lang} text to {target_lang}. 
Keep the meaning and tone intact. Only output the translation.{examples_text}

{source_lang} text to translate:
{text}

{target_lang} translation:"""
        return prompt
    
    @staticmethod
    def build_domain_specific_prompt(text: str, domain: str = "mathematics",
                                    source_lang: str = "English", 
                                    target_lang: str = "Vietnamese") -> str:
        """Build domain-specific translation prompt."""
        domain_guidance = {
            "mathematics": "mathematical terminology and symbols should be preserved. Ensure numbers and equations are correct.",
            "academic": "maintain formal academic tone and preserve technical terms where appropriate.",
            "general": "maintain natural and fluent language."
        }
        
        guidance = domain_guidance.get(domain, "")
        
        return f"""You are a professional {domain} translator specializing in {source_lang} to {target_lang} translation.
{guidance}

Translate the following text to {target_lang}. Only output the translation without any explanation.

Text to translate:
{text}

Translation:"""


class MultilingualTranslator:
    """Base class for translation using different LLM backends."""
    
    def __init__(self, use_cache: bool = True, cache_dir: str = "./translation_cache"):
        self.use_cache = use_cache
        self.cache = TranslationCache(cache_dir) if use_cache else None
        self.translation_count = 0
        self.cache_hits = 0
    
    def translate(self, text: str, source_lang: str = "English", 
                  target_lang: str = "Vietnamese", domain: str = "general") -> str:
        """Translate text. Override in subclasses."""
        raise NotImplementedError
    
    def translate_batch(self, texts: List[str], source_lang: str = "English",
                       target_lang: str = "Vietnamese", domain: str = "general",
                       batch_size: int = 5, show_progress: bool = True) -> List[str]:
        """Translate a batch of texts."""
        results = []
        
        iterator = tqdm(texts, disable=not show_progress, desc="Translating")
        for text in iterator:
            translated = self.translate(text, source_lang, target_lang, domain)
            results.append(translated)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get translation statistics."""
        return {
            "total_translations": self.translation_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / self.translation_count if self.translation_count > 0 else 0
        }


class GeminiTranslator(MultilingualTranslator):
    """Translator using Google Gemini API with retry logic and key rotation."""
    
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True, 
                 cache_dir: str = "./translation_cache", model_name: str = "gemini-2.5-flash-lite",
                 max_retries: int = 3, retry_delay: float = 1.0, enable_key_rotation: bool = True):
        super().__init__(use_cache, cache_dir)
        
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
        
        # Retry and key rotation configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay  # Base delay in seconds
        self.enable_key_rotation = enable_key_rotation
        self.current_key_index = 0
        self.api_keys = self._load_api_keys(api_key)
        
        if not self.api_keys:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.model_name = model_name
        self._setup_model()
    
    def _load_api_keys(self, api_key: Optional[str]) -> List[str]:
        """Load single or multiple API keys from env.
        Supports GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. for key rotation.
        """
        keys = []
        
        # First check for numbered keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
        if self.enable_key_rotation:
            idx = 1
            while True:
                key = os.getenv(f"GEMINI_API_KEY_{idx}")
                if not key:
                    break
                keys.append(key)
                idx += 1
        
        # Fall back to single GEMINI_API_KEY or provided api_key
        if not keys:
            if api_key is None:
                api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                keys.append(api_key)
        
        if keys:
            logger.info(f"Loaded {len(keys)} Gemini API key(s) for rotation")
        
        return keys
    
    def _setup_model(self):
        """Configure Gemini model with current API key."""
        if self.current_key_index >= len(self.api_keys):
            self.current_key_index = 0
        
        current_key = self.api_keys[self.current_key_index]
        self.genai.configure(api_key=current_key)
        self.model = self.genai.GenerativeModel(self.model_name)
        logger.debug(f"Configured Gemini with key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def _rotate_key(self):
        """Switch to next API key in rotation."""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self._setup_model()
            logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit (429) error."""
        error_str = str(error)
        return "429" in error_str or "quota" in error_str.lower() or "rate_limit" in error_str.lower()
    
    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after delay from error message if available."""
        error_str = str(error)
        # Look for patterns like "Please retry in 9.75 seconds"
        match = re.search(r'Please retry in ([\d.]+)s', error_str)
        if match:
            return float(match.group(1))
        return None
    
    def translate(self, text: str, source_lang: str = "English", 
                  target_lang: str = "Vietnamese", domain: str = "general") -> str:
        """Translate using Gemini API with exponential backoff and key rotation."""
        
        # Check cache first
        if self.use_cache:
            cached = self.cache.get(text, source_lang, target_lang)
            if cached:
                self.cache_hits += 1
                self.translation_count += 1
                return cached
        
        prompt = TranslationPromptBuilder.build_domain_specific_prompt(
            text, domain, source_lang, target_lang
        )
        
        last_error = None
        keys_tried = 0
        
        # Try with exponential backoff and key rotation
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                translation = response.text.strip()
                
                # Store in cache
                if self.use_cache:
                    self.cache.set(text, translation, source_lang, target_lang)
                
                self.translation_count += 1
                return translation
            
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    logger.warning(f"Rate limit error (attempt {attempt + 1}/{self.max_retries}): {error_str[:200]}")
                    
                    # Extract retry-after if available, otherwise use exponential backoff
                    retry_after = self._extract_retry_after(e)
                    if retry_after:
                        wait_time = retry_after
                    else:
                        # Exponential backoff: 1s, 2s, 4s, 8s, ...
                        wait_time = self.retry_delay * (2 ** attempt)
                    
                    if attempt < self.max_retries - 1:
                        logger.info(f"Waiting {wait_time:.2f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        # Last attempt failed with rate limit, try key rotation
                        if self.enable_key_rotation and len(self.api_keys) > 1:
                            keys_tried += 1
                            if keys_tried < len(self.api_keys):
                                logger.info(f"Rotating to next API key ({keys_tried + 1}/{len(self.api_keys)})...")
                                self._rotate_key()
                                attempt = -1  # Reset attempt counter for new key
                                continue
                
                else:
                    # Non-rate-limit error
                    logger.error(f"Translation error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
        
        # All retries exhausted
        logger.error(f"Translation failed after {self.max_retries} retries. Last error: {last_error}")
        return text  # Return original text if all retries fail


class OpenAITranslator(MultilingualTranslator):
    """Translator using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True,
                 cache_dir: str = "./translation_cache", model_name: str = "gpt-3.5-turbo"):
        super().__init__(use_cache, cache_dir)
        
        try:
            from openai import OpenAI
            self.openai = OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = self.openai(api_key=api_key)
        self.model_name = model_name
    
    def translate(self, text: str, source_lang: str = "English",
                  target_lang: str = "Vietnamese", domain: str = "general") -> str:
        """Translate using OpenAI API."""
        
        # Check cache first
        if self.use_cache:
            cached = self.cache.get(text, source_lang, target_lang)
            if cached:
                self.cache_hits += 1
                self.translation_count += 1
                return cached
        
        try:
            prompt = TranslationPromptBuilder.build_domain_specific_prompt(
                text, domain, source_lang, target_lang
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are a professional translator from {source_lang} to {target_lang}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            translation = response.choices[0].message.content.strip()
            
            # Store in cache
            if self.use_cache:
                self.cache.set(text, translation, source_lang, target_lang)
            
            self.translation_count += 1
            return translation
        
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text if translation fails


class LocalLLMTranslator(MultilingualTranslator):
    """Translator using local LLM models (Llama, Qwen, etc.)."""
    
    def __init__(self, model_path: str, use_cache: bool = True,
                 cache_dir: str = "./translation_cache"):
        super().__init__(use_cache, cache_dir)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self.torch = torch
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError("transformers and torch packages are required")
        
        self.model_path = model_path
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading model from {model_path} on device {self.device}")
        self.tokenizer = self.AutoTokenizer.from_pretrained(model_path)
        self.model = self.AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=self.torch.float16 if self.device == "cuda" else self.torch.float32,
            load_in_8bit=True if self.device == "cuda" else False
        )
    
    def translate(self, text: str, source_lang: str = "English",
                  target_lang: str = "Vietnamese", domain: str = "general") -> str:
        """Translate using local LLM."""
        
        # Check cache first
        if self.use_cache:
            cached = self.cache.get(text, source_lang, target_lang)
            if cached:
                self.cache_hits += 1
                self.translation_count += 1
                return cached
        
        try:
            prompt = TranslationPromptBuilder.build_domain_specific_prompt(
                text, domain, source_lang, target_lang
            )
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs,
                max_length=1000,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                no_repeat_ngram_size=2
            )
            
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the translation part (after the prompt)
            translation = translation[len(prompt):].strip()
            
            # Store in cache
            if self.use_cache:
                self.cache.set(text, translation, source_lang, target_lang)
            
            self.translation_count += 1
            return translation
        
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text if translation fails


def get_translator(backend: str = "gemini", **kwargs) -> MultilingualTranslator:
    """Factory function to get translator instance."""
    backends = {
        "gemini": GeminiTranslator,
        "openai": OpenAITranslator,
        "local": LocalLLMTranslator,
    }
    
    if backend not in backends:
        raise ValueError(f"Unknown translator backend: {backend}. Available: {list(backends.keys())}")
    
    return backends[backend](**kwargs)
