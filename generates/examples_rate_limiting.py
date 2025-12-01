#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples for using Vi-S1K with rate limit handling and API key rotation.
Run examples showing how to handle Gemini quota exhaustion gracefully.
"""

import sys
import io
from pathlib import Path

# Encoding fix for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "llm_evaluation"))


def example_1_single_key_with_retry():
    """Example 1: Single API key with automatic retry and backoff."""
    print("\n" + "=" * 70)
    print("Example 1: Single API Key with Exponential Backoff")
    print("=" * 70)
    print("""
The translator automatically handles rate limits (429 errors) with exponential backoff.
No configuration needed!

When you hit rate limit:
  1st attempt fails → Wait 1 second → Retry
  2nd attempt fails → Wait 2 seconds → Retry
  3rd attempt fails → Wait 4 seconds → Retry
  All retries fail  → Return original text

Setup:
  1. Set GEMINI_API_KEY in environment
  2. Run: python generates/vi_s1k_builder.py --test-run

The script will automatically retry with exponential backoff.
""")


def example_2_multiple_keys_setup():
    """Example 2: Setup and use multiple API keys for quota rotation."""
    print("\n" + "=" * 70)
    print("Example 2: Setup Multiple API Keys for Quota Rotation")
    print("=" * 70)
    print("""
With multiple keys, when one key hits quota, the translator automatically 
switches to the next key.

Step 1: Get multiple Gemini API keys
  - Go to https://ai.google.dev
  - Create 2-3 Google accounts (free tier = 15 req/min each)
  - Get API keys for each account

Step 2a: Set via .env file (RECOMMENDED)
  Create generates/.env:
    GEMINI_API_KEY_1=sk_gemini_xxxxxxx_account_1
    GEMINI_API_KEY_2=sk_gemini_yyyyyyy_account_2
    GEMINI_API_KEY_3=sk_gemini_zzzzzzz_account_3

Step 2b: Set via PowerShell (temporary for current session)
  $env:GEMINI_API_KEY_1 = "your_first_key"
  $env:GEMINI_API_KEY_2 = "your_second_key"
  $env:GEMINI_API_KEY_3 = "your_third_key"

Step 2c: Set via setx (persist across sessions, requires new terminal)
  setx GEMINI_API_KEY_1 "your_first_key"
  setx GEMINI_API_KEY_2 "your_second_key"
  setx GEMINI_API_KEY_3 "your_third_key"

Step 3: Run the builder (no changes needed)
  python generates/vi_s1k_builder.py --max-samples 10000

The translator will automatically:
  - Start with GEMINI_API_KEY_1
  - When quota exceeded, rotate to GEMINI_API_KEY_2
  - Then to GEMINI_API_KEY_3
  - Repeat until all translations complete

Watch the logs:
  INFO:Loaded 3 Gemini API key(s) for rotation
  WARNING:Rate limit error (attempt 1/3): 429 You exceeded quota...
  INFO:Rotated to API key 2/3
  INFO:Rotated to API key 3/3
""")


def example_3_check_current_key():
    """Example 3: Check which API keys are loaded."""
    print("\n" + "=" * 70)
    print("Example 3: Verify API Keys are Loaded Correctly")
    print("=" * 70)
    print("""
Check if your API keys were loaded by the translator:

Python code:
  import os
  
  # Check single key
  key1 = os.getenv("GEMINI_API_KEY")
  print(f"GEMINI_API_KEY: {'SET' if key1 else 'NOT SET'}")
  
  # Check multiple keys
  for i in range(1, 4):
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
      print(f"GEMINI_API_KEY_{i}: SET (last 4 chars: ...{key[-4:]})")
    else:
      print(f"GEMINI_API_KEY_{i}: NOT SET")

PowerShell:
  $env:GEMINI_API_KEY_1 | Select-Object -First 1
  $env:GEMINI_API_KEY_2 | Select-Object -First 1
  $env:GEMINI_API_KEY_3 | Select-Object -First 1

Expected output (if 3 keys set):
  GEMINI_API_KEY_1: SET (last 4 chars: ...aNw)
  GEMINI_API_KEY_2: SET (last 4 chars: ...Xyz)
  GEMINI_API_KEY_3: SET (last 4 chars: ...Abc)
""")


def example_4_monitor_progress():
    """Example 4: Monitor translation progress and quota usage."""
    print("\n" + "=" * 70)
    print("Example 4: Monitor Translation Progress")
    print("=" * 70)
    print("""
After running the builder, check the statistics:

Command:
  python -c "import json; s = json.load(open('results/vi_s1k/statistics.json')); \\
    print(f'Translated: {s[\\\"successful\\\"]} / {s[\\\"total_items\\\"]}'); \\
    print(f'Success rate: {s[\\\"success_rate\\\"]:.1%}'); \\
    print(f'Cache hit rate: {s[\\\"translator_stats\\\"][\\\"cache_hit_rate\\\"]:.1%}')"

Or PowerShell:
  $stats = Get-Content results/vi_s1k/statistics.json | ConvertFrom-Json
  $stats | Select-Object -Property successful, failed, total_items, success_rate | Format-Table

Output example:
  successful  failed  total_items  success_rate
  ----------  ------  -----------  -----------
  9800        200     10000        98.00%

Cache performance:
  translator_stats.total_translations: Total API calls made
  translator_stats.cache_hits: Times result was retrieved from cache (no API call)
  translator_stats.cache_hit_rate: Percentage of requests served from cache
""")


def example_5_resume_after_quota():
    """Example 5: Resume translation after hitting quota."""
    print("\n" + "=" * 70)
    print("Example 5: Resume After Hitting Quota Limit")
    print("=" * 70)
    print("""
If you hit quota on all keys before finishing:

1. Add more API keys to .env or environment
2. Clear the cache to re-translate (optional):
   - Remove translation_cache/ folder
   - Or modify cache_dir in vi_s1k_builder.py

3. Run builder again with different parameters:
   python generates/vi_s1k_builder.py --max-samples 5000 --translator gemini

The builder will:
  - Use newly added keys
  - Skip translations that are cached (if not cleared)
  - Continue from where it left off

For persistent cache across runs:
  - Cache is automatically saved in translation_cache/
  - Re-running with same samples will use cache (no API quota used)
  - Good for testing and development

For fresh translation:
  - Delete translation_cache/ folder first
  - Run builder again
""")


def example_6_custom_retry_settings():
    """Example 6: Customize retry and backoff settings."""
    print("\n" + "=" * 70)
    print("Example 6: Custom Retry and Backoff Settings")
    print("=" * 70)
    print("""
Edit generates/vi_s1k_builder.py to customize retry behavior.

Default settings (line ~56):
  self.translator = get_translator(
      backend=translator_backend,
      use_cache=True,
      cache_dir=cache_dir,
      max_retries=3,           # Try 3 times per translation
      retry_delay=1.0,         # Start with 1 second delay
      enable_key_rotation=True # Use multiple keys if available
  )

Aggressive retry (for unstable networks):
  max_retries=5,       # Try 5 times
  retry_delay=2.0      # Start with 2 second delay (2, 4, 8, 16, 32 sec)

Conservative retry (quick fail for testing):
  max_retries=1,       # Try only once
  retry_delay=0.5      # Start with 0.5 second delay

Key rotation disabled (use single key only):
  enable_key_rotation=False

Changes to GeminiTranslator:
  - max_retries: Number of attempts per item
  - retry_delay: Base delay (multiplied by 2^attempt)
  - enable_key_rotation: Auto-switch keys when quota exceeded
""")


def example_7_translate_programmatically():
    """Example 7: Use translator directly in your code."""
    print("\n" + "=" * 70)
    print("Example 7: Use Translator Directly in Your Code")
    print("=" * 70)
    print("""
You can import and use the translator directly:

Python code:
  from llm_evaluation.utils.translation_utils import get_translator
  
  # Get translator with multiple keys
  translator = get_translator(
      backend="gemini",
      use_cache=True,
      cache_dir="./translation_cache",
      max_retries=3,
      enable_key_rotation=True
  )
  
  # Translate a single item
  english_text = "What is 2 + 2?"
  vietnamese = translator.translate(
      text=english_text,
      source_lang="English",
      target_lang="Vietnamese",
      domain="mathematics"
  )
  print(f"English: {english_text}")
  print(f"Vietnamese: {vietnamese}")
  
  # Batch translation
  texts = ["Question 1", "Question 2", "Question 3"]
  translations = [
      translator.translate(text, domain="mathematics") 
      for text in texts
  ]
  
  # Check statistics
  print(f"Total translations: {translator.translation_count}")
  print(f"Cache hits: {translator.cache_hits}")
  print(f"Cache hit rate: {translator.cache_hit_rate:.1%}")

The translator automatically:
  - Caches translations
  - Retries on failure
  - Rotates keys when quota hit
  - Returns original text if all retries fail
""")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  VI-S1K: Rate Limiting & API Key Rotation Examples".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    examples = [
        example_1_single_key_with_retry,
        example_2_multiple_keys_setup,
        example_3_check_current_key,
        example_4_monitor_progress,
        example_5_resume_after_quota,
        example_6_custom_retry_settings,
        example_7_translate_programmatically,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\nError running example {i}: {e}")
    
    print("\n" + "=" * 70)
    print("Quick Start:")
    print("=" * 70)
    print("""
1. Setup API keys:
   Create generates/.env with:
     GEMINI_API_KEY_1=your_key_1
     GEMINI_API_KEY_2=your_key_2

2. Test with 5 samples:
   python generates/vi_s1k_builder.py --test-run

3. Build full dataset:
   python generates/vi_s1k_builder.py --max-samples 10000

4. Monitor progress:
   python -c "import json; print(json.load(open('results/vi_s1k/statistics.json')))"

Happy translating! 🚀
""")


if __name__ == "__main__":
    main()
