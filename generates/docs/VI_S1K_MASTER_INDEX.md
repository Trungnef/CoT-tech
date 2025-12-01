# 🌍 Vi-S1K Translation System - Master Documentation Index

> **Vietnamese S1K Benchmark: High-quality elementary mathematics dataset in Vietnamese**

## 📚 Complete Documentation

### 🚀 **START HERE** - Choose Your Path

```
Choose Your Role:
├─ 👤 I'm New to This System
│  └─→ Read: VI_S1K_SUMMARY.md (first section)
│  └─→ Run: python generates/quickstart_vi_s1k.py
│
├─ 👨‍💼 I'm a Manager/Stakeholder
│  └─→ Read: VI_S1K_SUMMARY.md (Overview + Features)
│  └─→ See: Output examples in README_VI_S1K.md
│
├─ 👨‍💻 I'm a Data Scientist
│  └─→ Run: python generates/examples_vi_s1k.py
│  └─→ Read: VI_S1K_SUMMARY.md (Usage Examples)
│  └─→ Follow: INTEGRATION_GUIDE.md
│
├─ 🛠️ I'm a Developer
│  └─→ Start: python generates/setup_vi_s1k.py
│  └─→ Study: s1k_translator.py & translation_utils.py
│  └─→ Extend: Modify classes as needed
│
└─ 🚀 I'm Ready to Deploy
   └─→ Setup: python generates/setup_vi_s1k.py
   └─→ Build: python generates/vi_s1k_builder.py
   └─→ Integrate: INTEGRATION_GUIDE.md
```

---

## 📖 Documentation Files (Read Order)

### 1️⃣ **Quick Reference** (5-10 minutes)
   - **File**: `VI_S1K_SUMMARY.md`
   - **Why**: Get overview of what Vi-S1K is and how to use it
   - **For**: Everyone
   - **Read**:
     - Overview (what it is)
     - Usage (how to use it)
     - Features (what it can do)

### 2️⃣ **File Index** (5 minutes)
   - **File**: `VI_S1K_FILE_INDEX.md`
   - **Why**: Understand what each file does
   - **For**: Developers, advanced users
   - **Read**: File-by-file breakdown and relationships

### 3️⃣ **Full Documentation** (20-30 minutes)
   - **File**: `README_VI_S1K.md`
   - **Why**: Complete guide with all details
   - **For**: Users who want to know everything
   - **Read**:
     - Installation
     - Usage (CLI & Python API)
     - All command-line options
     - Troubleshooting

### 4️⃣ **Integration Guide** (15-20 minutes)
   - **File**: `INTEGRATION_GUIDE.md`
   - **Why**: How to use with LLM Evaluator
   - **For**: Users integrating with existing systems
   - **Read**:
     - Architecture diagram
     - Step-by-step integration
     - Python API examples

### 5️⃣ **Implementation Status** (5 minutes)
   - **File**: `VI_S1K_IMPLEMENTATION_CHECKLIST.md`
   - **Why**: What's been built and tested
   - **For**: Project managers, developers
   - **Read**: Feature matrix and achievements

---

## 🐍 Interactive Helper Scripts

### 1. **quickstart_vi_s1k.py** - Get Started Quickly
```bash
python generates/quickstart_vi_s1k.py
```
**What it does**:
- Checks dependencies
- Verifies environment setup
- Runs quick demo
- Shows usage examples
- Guides next steps

**For**: First-time users

### 2. **setup_vi_s1k.py** - Automated Setup
```bash
python generates/setup_vi_s1k.py
```
**What it does**:
- Installs dependencies
- Creates .env file
- Sets up directories
- Verifies installation
- Tests translator

**For**: Setting up system

### 3. **examples_vi_s1k.py** - Learn by Examples
```bash
python generates/examples_vi_s1k.py
```
**Examples included**:
1. Basic usage (full workflow)
2. Streaming translation
3. Custom quality checking
4. Multilingual translation
5. Batch processing
6. Caching efficiency
7. Error handling
8. Integration notes

**For**: Learning different use cases

---

## 🔧 Core Implementation Files

### Translation Engine
**File**: `llm_evaluation/utils/translation_utils.py`
- Translation with Gemini, OpenAI, or Local models
- Automatic caching system
- Domain-specific prompts
- Error handling and retry logic

### Dataset Builder
**File**: `generates/s1k_translator.py`
- Load S1K from Hugging Face
- Translate to Vietnamese
- Quality checking
- Multi-format output (JSON, JSONL, CSV)

### Main CLI
**File**: `generates/vi_s1k_builder.py`
- Command-line interface
- Orchestrates translation workflow
- Statistics and reporting
- Checkpoint support

### Configuration
**File**: `generates/vi_s1k_config.py`
- Configuration dataclasses
- Preset configurations
- File-based persistence
- Environment variable support

---

## 🎯 Common Tasks & How to Do Them

### Task 1: Build Vi-S1K Benchmark (5-10 minutes)
```bash
# Test first (5 samples)
python generates/vi_s1k_builder.py --test-run

# Then build (100 samples)
python generates/vi_s1k_builder.py --translator gemini --max-samples 100

# Check results
ls results/vi_s1k/
cat results/vi_s1k/statistics.json
```
**Docs**: README_VI_S1K.md (Usage section)

### Task 2: Translate Custom Text (5 minutes)
```python
from llm_evaluation.utils.translation_utils import get_translator

translator = get_translator("gemini")
result = translator.translate(
    "What is 2 + 2?",
    "English",
    "Vietnamese",
    domain="mathematics"
)
print(result)
```
**Docs**: VI_S1K_SUMMARY.md (Examples)

### Task 3: Evaluate Models on Vi-S1K (10-15 minutes)
```bash
# Build Vi-S1K first
python generates/vi_s1k_builder.py

# Then evaluate
python llm_evaluation/main.py \
    --questions-file results/vi_s1k/vi_s1k_benchmark.json \
    --models gemini llama \
    --prompts zero_shot few_shot_3
```
**Docs**: INTEGRATION_GUIDE.md

### Task 4: Customize Translation Quality (10 minutes)
```python
from generates.s1k_translator import QualityChecker

# Create custom checker
def my_quality_check(item):
    if len(item.vietnamese_question) > 5:
        return 0.8
    return 0.3

# Use in builder
builder.build_benchmark(dataset, quality_checker=my_quality_check)
```
**Docs**: examples_vi_s1k.py (Example 3)

### Task 5: Process Large Dataset (30-60 minutes)
```bash
# Use batching
python generates/vi_s1k_builder.py \
    --max-samples 10000 \
    --batch-size 100
```
**Docs**: README_VI_S1K.md (Batch Processing)

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Input: S1K Dataset                     │
│            (from Hugging Face: simplescaling/s1K-1.1)   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  S1KDatasetLoader        │
        │  (Load & extract fields) │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  VietnameseBenchmarkBuilder      │
        │  (Translate & check quality)     │
        └──────────────┬────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
    ┌─────────────────┐  ┌─────────────────┐
    │ Translator      │  │ QualityChecker  │
    │ (Gemini/OpenAI/ │  │ (Simple/LLM)    │
    │  Local)         │  │                 │
    └────────┬────────┘  └────────┬────────┘
             │                    │
             └─────────┬──────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  Output: Vi-S1K          │
        │  JSON/JSONL/CSV          │
        │  + Statistics            │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  LLM Evaluator (Optional)│
        │  Evaluate models on Vi-S1K
        └──────────────────────────┘
```

---

## 🎓 Learning Path

### Beginner (0-1 hour)
1. Run: `python generates/quickstart_vi_s1k.py`
2. Read: `VI_S1K_SUMMARY.md` (first half)
3. Try: `python generates/vi_s1k_builder.py --test-run`
4. Understand: Output structure

### Intermediate (1-3 hours)
1. Read: Full `VI_S1K_SUMMARY.md`
2. Read: `README_VI_S1K.md`
3. Run: `python generates/examples_vi_s1k.py`
4. Try: Build with 100 samples
5. Check: Statistics and output formats

### Advanced (3-6 hours)
1. Study: Core implementation files
2. Read: `INTEGRATION_GUIDE.md`
3. Run: Examples 3-6 (custom quality, multilingual, etc.)
4. Integrate: With LLM Evaluator
5. Extend: Customize for your needs

---

## 🚨 Quick Troubleshooting

| Error | Solution | Docs |
|-------|----------|------|
| "Module not found" | `pip install datasets google-generativeai` | README_VI_S1K.md |
| "API key not found" | Create `.env` file | README_VI_S1K.md |
| "Translation too slow" | Check cache, first run slower | README_VI_S1K.md |
| "Out of memory" | Reduce batch size | README_VI_S1K.md |
| "Low quality scores" | Check translator, adjust prompts | examples_vi_s1k.py |

---

## 📋 File Quick Reference

| File | Type | Purpose | Read Time |
|------|------|---------|-----------|
| VI_S1K_SUMMARY.md | Doc | Quick reference | 10 min |
| VI_S1K_FILE_INDEX.md | Doc | File descriptions | 5 min |
| README_VI_S1K.md | Doc | Full manual | 30 min |
| INTEGRATION_GUIDE.md | Doc | Integration guide | 20 min |
| VI_S1K_IMPLEMENTATION_CHECKLIST.md | Doc | Status report | 5 min |
| quickstart_vi_s1k.py | Script | Get started | Interactive |
| setup_vi_s1k.py | Script | Automated setup | Interactive |
| examples_vi_s1k.py | Script | Learn examples | Interactive |
| vi_s1k_builder.py | Code | Main CLI | - |
| s1k_translator.py | Code | Dataset builder | - |
| translation_utils.py | Code | Translation engines | - |
| vi_s1k_config.py | Code | Configuration | - |

---

## 🎯 Success Criteria

You'll know you've succeeded when:

✅ **Setup Complete**
- Dependencies installed
- .env file created
- quickstart script runs without errors

✅ **Basic Usage Works**
- Can run `python generates/vi_s1k_builder.py --test-run` successfully
- Benchmark created in `results/vi_s1k/`
- Statistics show 100% success rate

✅ **Integration Complete**
- Vi-S1K benchmark integrated with LLM Evaluator
- Can run evaluations on Vietnamese questions
- Reports generated successfully

✅ **Customization Ready**
- Can modify translation prompts
- Can create custom quality checkers
- Can process different datasets

---

## 📞 Getting Help

### Where to Find Information
- **What is it?** → VI_S1K_SUMMARY.md
- **How to install?** → setup_vi_s1k.py or README_VI_S1K.md
- **How to use?** → examples_vi_s1k.py or quickstart_vi_s1k.py
- **How to integrate?** → INTEGRATION_GUIDE.md
- **Something not working?** → README_VI_S1K.md (Troubleshooting)
- **What's implemented?** → VI_S1K_IMPLEMENTATION_CHECKLIST.md
- **How are files organized?** → VI_S1K_FILE_INDEX.md

### Quick Links
- 🚀 **Quick Start**: `python generates/quickstart_vi_s1k.py`
- 🔧 **Setup**: `python generates/setup_vi_s1k.py`
- 📚 **Examples**: `python generates/examples_vi_s1k.py`
- 📖 **Documentation**: `generates/README_VI_S1K.md`

---

## ✨ Key Features at a Glance

✅ **Multi-Backend Support**: Gemini, OpenAI, Local Models
✅ **Intelligent Caching**: Save 50-90% API costs
✅ **Quality Control**: Automatic quality checking
✅ **Multiple Formats**: JSON, JSONL, CSV
✅ **Easy Integration**: Works with LLM Evaluator
✅ **Comprehensive Docs**: 5 detailed guides
✅ **Interactive Helpers**: quickstart, setup, examples
✅ **Production Ready**: Error handling, logging, monitoring

---

## 🚀 Ready to Start?

### Option 1: Guided (Recommended for First-time Users)
```bash
python generates/quickstart_vi_s1k.py
```

### Option 2: Automated Setup
```bash
python generates/setup_vi_s1k.py
python generates/vi_s1k_builder.py --test-run
```

### Option 3: Direct Usage
```bash
python generates/vi_s1k_builder.py --translator gemini --max-samples 100
```

### Option 4: Learn by Examples
```bash
python generates/examples_vi_s1k.py
```

---

## 📌 Remember

- **First run is slower** (no cache yet)
- **Second run is faster** (cache hits)
- **Test first** with `--test-run` before full build
- **Check statistics** after building
- **Read docs** relevant to your use case

---

**🎉 You're all set! Choose your path above and get started!**

---

*Vi-S1K: Vietnamese S1K Benchmark Translation System*
*Part of CoT-tech: Vietnamese Mathematics Evaluation Framework*
*For support, see relevant documentation files above*
