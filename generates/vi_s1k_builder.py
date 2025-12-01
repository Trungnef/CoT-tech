#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for building Vi-S1K (Vietnamese S1K Benchmark)
Translates simplescaling/s1K-1.1 dataset to Vietnamese

Usage:
    python vi_s1k_builder.py --translator gemini --max-samples 100 --output results/vi_s1k.json
    python vi_s1k_builder.py --translator openai --backend gpt-4 --resume
"""

import argparse
import json
import logging
import sys
import io
from pathlib import Path
from typing import Optional

# Fix encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))  # generates folder
sys.path.insert(0, str(Path(__file__).parent.parent))  # root folder
sys.path.insert(0, str(Path(__file__).parent.parent / "llm_evaluation"))  # llm_evaluation folder

# Try to load .env from project root and generates/ if python-dotenv is available.
# This is optional and non-fatal; if python-dotenv isn't installed the script still runs
# but environment variables must be set by the user.
try:
    from dotenv import load_dotenv
    # Load root .env first, then generates/.env to allow overrides
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except Exception:
    # python-dotenv not installed or load failed - proceed without it
    pass

from s1k_translator import (
    S1KDatasetLoader,
    VietnameseBenchmarkBuilder,
    QualityChecker
)
from llm_evaluation.utils.translation_utils import get_translator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ViS1KBuilder:
    """Main builder class for creating Vi-S1K benchmark."""
    
    def __init__(self, translator_backend: str = "gemini", 
                 cache_dir: str = "./translation_cache",
                 quality_check: bool = True):
        """Initialize builder."""
        self.translator_backend = translator_backend
        self.quality_check = quality_check
        
        # Get translator instance
        try:
            self.translator = get_translator(
                backend=translator_backend,
                use_cache=True,
                cache_dir=cache_dir
            )
            logger.info(f"Translator initialized: {translator_backend}")
        except Exception as e:
            logger.error(f"Failed to initialize translator: {e}")
            raise
    
    def build(self, output_dir: str = "./results/vi_s1k",
              max_samples: Optional[int] = None,
              split: str = "train",
              output_formats: list = ["json", "jsonl"]) -> dict:
        """
        Build Vi-S1K benchmark.
        
        Args:
            output_dir: Directory to save results
            max_samples: Maximum number of samples to translate
            split: Dataset split to use
            output_formats: Output formats to save
        
        Returns:
            Dictionary with build statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Building Vi-S1K Benchmark")
        logger.info("=" * 60)
        
        # Load dataset
        logger.info("Step 1: Loading S1K dataset from Hugging Face...")
        loader = S1KDatasetLoader()
        raw_dataset = loader.load_s1k(split=split, max_samples=max_samples)
        
        # Extract standard fields
        dataset = [loader.extract_question_fields(item) for item in raw_dataset]
        logger.info(f"Loaded {len(dataset)} questions")
        
        # Build benchmark
        logger.info("\nStep 2: Translating to Vietnamese...")
        builder = VietnameseBenchmarkBuilder(self.translator)
        
        quality_checker = None
        if self.quality_check:
            quality_checker = QualityChecker.simple_check
        
        translated_items = builder.build_benchmark(
            dataset,
            quality_checker=quality_checker,
            show_progress=True
        )
        
        # Save results
        logger.info("\nStep 3: Saving benchmark...")
        for fmt in output_formats:
            output_file = output_path / f"vi_s1k_benchmark.{fmt}"
            builder.save_benchmark(str(output_file), format=fmt)
            logger.info(f"Saved: {output_file}")
        
        # Save statistics
        stats = builder.get_statistics()
        stats_file = output_path / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved statistics: {stats_file}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Build Complete!")
        logger.info("=" * 60)
        self._print_statistics(stats)
        
        return stats
    
    def _print_statistics(self, stats: dict):
        """Print formatted statistics."""
        print(f"\nVi-S1K Benchmark Statistics:")
        print(f"  Total items: {stats['total_items']}")
        print(f"  Successfully translated: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Output columns: {stats.get('output_columns', 'unknown')}")
        print(f"  Columns breakdown: {stats.get('columns_breakdown', {})}")
        
        translator_stats = stats.get('translator_stats', {})
        if translator_stats:
            print(f"\nTranslator Statistics:")
            print(f"  Total translations: {translator_stats.get('total_translations', 0)}")
            print(f"  Cache hits: {translator_stats.get('cache_hits', 0)}")
            print(f"  Cache hit rate: {translator_stats.get('cache_hit_rate', 0):.1%}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Vi-S1K benchmark from S1K dataset"
    )
    
    parser.add_argument(
        "--translator",
        choices=["gemini", "openai", "local"],
        default="gemini",
        help="Translator backend to use"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to translate"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./results/vi_s1k",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (train/test/validation)"
    )
    
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["json", "jsonl"],
        choices=["json", "jsonl", "csv"],
        help="Output formats to save"
    )
    
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Skip quality checking"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="./translation_cache",
        help="Directory for translation cache"
    )
    
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run with only 5 samples for testing"
    )
    
    args = parser.parse_args()
    
    # Handle test run
    if args.test_run:
        args.max_samples = 5
        logger.info("Running in test mode with 5 samples")
    
    try:
        builder = ViS1KBuilder(
            translator_backend=args.translator,
            cache_dir=args.cache_dir,
            quality_check=not args.no_quality_check
        )
        
        builder.build(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            split=args.split,
            output_formats=args.formats
        )
        
    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
