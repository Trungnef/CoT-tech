# Rate Limit & Key Rotation Guide for Vi-S1K

## Problem
The Gemini free tier has a quota limit of **15 requests/minute**. When building large datasets, you'll hit this limit and get `429 Quota exceeded` errors.

## Solutions

### Solution 1: Exponential Backoff (Built-in)
The updated `GeminiTranslator` now automatically handles rate limits with exponential backoff:
- **Attempt 1**: Waits 1 second before retry
- **Attempt 2**: Waits 2 seconds before retry  
- **Attempt 3**: Waits 4 seconds before retry

The script extracts the `retry-after` delay from API error messages when available and uses that instead.

**No configuration needed** — this is automatic.

### Solution 2: Multiple API Keys (Key Rotation)
If you have multiple Google accounts with free-tier Gemini access:

#### Setup (PowerShell)
1. Generate or get multiple Gemini API keys from [ai.google.dev](https://ai.google.dev)
2. Add them to `generates/.env` or create a root `.env`:

```env
GEMINI_API_KEY_1=your_first_key_here
GEMINI_API_KEY_2=your_second_key_here
GEMINI_API_KEY_3=your_third_key_here
```

Or set in PowerShell session:
```powershell
$env:GEMINI_API_KEY_1 = "your_first_key"
$env:GEMINI_API_KEY_2 = "your_second_key"
$env:GEMINI_API_KEY_3 = "your_third_key"
```

Or persist (requires new terminal):
```powershell
setx GEMINI_API_KEY_1 "your_first_key"
setx GEMINI_API_KEY_2 "your_second_key"
setx GEMINI_API_KEY_3 "your_third_key"
```

#### How It Works
When the translator hits a rate limit:
1. **Tries exponential backoff** on current key
2. **If still fails after 3 attempts**, automatically rotates to the next key
3. **Repeats** until all keys are exhausted

Logs show which key is active:
```
Loaded 3 Gemini API key(s) for rotation
Rotated to API key 2/3
```

### Solution 3: Use Paid Plans
- **Google Gemini Pro**: $5/1M tokens (much higher quota)
- **OpenAI GPT-4**: Switch `--translator openai` with paid key
- **Local models**: Use `--translator local` with Llama/Qwen (no API limits)

## Running with Rate Limit Handling

### Test run (5 samples)
```powershell
python generates/vi_s1k_builder.py --test-run
```

### Full benchmark build
```powershell
# With automatic retry and key rotation enabled
python generates/vi_s1k_builder.py --translator gemini --max-samples 1000
```

### With custom retry settings
```powershell
# Modify vi_s1k_builder.py to pass options:
# get_translator(backend="gemini", max_retries=5, retry_delay=2.0)
```

## Configuration

Edit `generates/vi_s1k_builder.py` line ~56 to customize:

```python
self.translator = get_translator(
    backend=translator_backend,
    use_cache=True,
    cache_dir=cache_dir,
    max_retries=3,        # Number of retry attempts
    retry_delay=1.0,      # Base delay in seconds (exponentially increased)
    enable_key_rotation=True  # Enable multi-key rotation
)
```

## Monitoring

Watch the logs for key rotation:
```
INFO:llm_evaluation.utils.translation_utils:Loaded 3 Gemini API key(s) for rotation
WARNING:llm_evaluation.utils.translation_utils:Rate limit error (attempt 1/3): 429 You exceeded your current quota...
INFO:llm_evaluation.utils.translation_utils:Waiting 1.00s before retry...
INFO:llm_evaluation.utils.translation_utils:Rotated to API key 2/3
```

## Translation Progress

After build completes, check:
```powershell
Get-Content results/vi_s1k/statistics.json | ConvertFrom-Json | Format-List

# Or Python:
python -c "import json; print(json.load(open('results/vi_s1k/statistics.json')), indent=2)"
```

Statistics show:
- `successful`: How many items translated successfully
- `failed`: How many failed after all retries
- `cache_hit_rate`: Percentage of translations from cache

## Troubleshooting

### Still hitting rate limits?
- Add more API keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
- Increase `retry_delay` to give more buffer between requests
- Use `--max-samples 100` and run in smaller batches

### Keys not being loaded?
- Ensure .env is in `generates/` folder or repo root
- Check: `python -c "import os; print(os.getenv('GEMINI_API_KEY_1'))"`
- Restart PowerShell after using `setx` to persist environment vars

### Still getting errors after retries?
- Check API key validity
- Check quota at [ai.google.dev/usage](https://ai.google.dev/usage?tab=rate-limit)
- Try switching to OpenAI or local models
