#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if Vi-S1K dataset has been translated.
Verify structure, count rows/columns, and show sample data.
"""

import sys
import io
import json
from pathlib import Path

# Encoding fix for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_translation_output():
    """Check if translation output files exist and are complete."""
    print("\n" + "=" * 80)
    print("VI-S1K TRANSLATION STATUS CHECK")
    print("=" * 80)
    
    output_dir = Path(__file__).parent.parent / "results" / "vi_s1k"
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Exists: {output_dir.exists()}")
    
    if not output_dir.exists():
        print("\n❌ OUTPUT DIRECTORY DOES NOT EXIST - Translation has not run yet")
        return
    
    # Check for output files
    json_file = output_dir / "vi_s1k_benchmark.json"
    jsonl_file = output_dir / "vi_s1k_benchmark.jsonl"
    csv_file = output_dir / "vi_s1k_benchmark.csv"
    stats_file = output_dir / "statistics.json"
    
    print("\nOutput files:")
    files_found = []
    for file_path in [json_file, jsonl_file, csv_file, stats_file]:
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        size_mb = size / (1024 * 1024) if size > 0 else 0
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path.name:40} {size:15,} bytes ({size_mb:8.2f} MB)")
        if exists:
            files_found.append(file_path)
    
    if not files_found:
        print("\n❌ NO TRANSLATION FILES FOUND - Translation has not completed")
        return
    
    # Check statistics
    if stats_file.exists():
        print("\n" + "-" * 80)
        print("TRANSLATION STATISTICS")
        print("-" * 80)
        
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print(f"\nTotal items in dataset: {stats.get('total_items', 'N/A')}")
        print(f"Successfully translated: {stats.get('successful', 0)}")
        print(f"Failed translations: {stats.get('failed', 0)}")
        print(f"Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"Average quality score: {stats.get('average_quality_score', 0):.2f}/1.0")
        
        if 'translator_stats' in stats:
            ts = stats['translator_stats']
            print(f"\nTranslator Statistics:")
            print(f"  Total translations: {ts.get('total_translations', 0)}")
            print(f"  Cache hits: {ts.get('cache_hits', 0)}")
            print(f"  Cache hit rate: {ts.get('cache_hit_rate', 0):.1%}")
    
    # Check dataset structure
    if json_file.exists():
        print("\n" + "-" * 80)
        print("DATASET STRUCTURE")
        print("-" * 80)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                sample = json.loads(first_line)
            
            print(f"\nColumns in translated dataset:")
            columns = sorted(sample.keys())
            for i, col in enumerate(columns, 1):
                value_sample = str(sample[col])[:60].replace('\n', ' ')
                print(f"  {i:2}. {col:40} {value_sample}...")
            
            print(f"\nTotal columns: {len(columns)}")
            
            # Count rows
            with open(json_file, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f)
            
            print(f"Total rows: {row_count}")
            
        except Exception as e:
            print(f"Error reading JSON file: {e}")
    
    # Show expected vs actual columns
    expected_columns = [
        'solution', 'question', 'cot_type', 'source_type', 'metadata',
        'gemini_thinking_trajectory', 'gemini_attempt',
        'deepseek_thinking_trajectory', 'deepseek_attempt',
        'gemini_grade', 'gemini_grade_reason',
        'deepseek_grade', 'deepseek_grade_reason'
    ]
    
    print("\n" + "-" * 80)
    print("EXPECTED VS ACTUAL COLUMNS")
    print("-" * 80)
    
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                sample = json.loads(f.readline())
            
            actual_columns = set(sample.keys())
            expected_set = set(expected_columns)
            
            print(f"\nExpected columns: {len(expected_set)}")
            print(f"Actual columns: {len(actual_columns)}")
            
            missing = expected_set - actual_columns
            extra = actual_columns - expected_set
            
            if missing:
                print(f"\n⚠️  Missing columns ({len(missing)}):")
                for col in sorted(missing):
                    print(f"  - {col}")
            
            if extra:
                print(f"\n✓ Extra columns added ({len(extra)}):")
                for col in sorted(extra):
                    print(f"  + {col}")
            
            if not missing and not extra:
                print("\n✓ All expected columns present, no extra columns")
            
        except Exception as e:
            print(f"Error checking columns: {e}")
    
    # Final status
    print("\n" + "=" * 80)
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f)
        
        if row_count > 0 and stats_file.exists():
            print("✓ TRANSLATION COMPLETED SUCCESSFULLY")
            print(f"✓ {row_count} rows translated")
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            print(f"✓ Success rate: {stats.get('success_rate', 0):.1%}")
        else:
            print("⚠️  TRANSLATION INCOMPLETE")
            print(f"   Only {row_count} rows found")
    else:
        print("❌ NO TRANSLATION OUTPUT - Run this command to start:")
        print("   python generates/vi_s1k_builder.py --max-samples 100")


def show_next_steps():
    """Show next steps to run full translation."""
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    
    print("""
1. Run test translation (5 samples):
   python generates/vi_s1k_builder.py --test-run

2. Run full translation (all samples):
   python generates/vi_s1k_builder.py --max-samples 10000

3. Use multiple API keys (if you have them):
   - Edit generates/.env and add:
     GEMINI_API_KEY_1=your_first_key
     GEMINI_API_KEY_2=your_second_key
   - Run: python generates/vi_s1k_builder.py --max-samples 10000

4. Check results:
   - View structure: python generates/check_translation_status.py
   - View statistics: Get-Content results/vi_s1k/statistics.json | ConvertFrom-Json
   - View sample data: Get-Content results/vi_s1k/vi_s1k_benchmark.json -Head 1 | ConvertFrom-Json

Note: First run translates all samples, subsequent runs use cache (much faster).
""")


if __name__ == "__main__":
    check_translation_output()
    show_next_steps()
