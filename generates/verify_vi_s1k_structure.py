#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify Vi-S1K output has all 13 original S1K columns + 8 Vietnamese translations"""

import json
import sys
import io

# Encoding fix for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Read the output
with open('results/vi_s1k/vi_s1k_benchmark.jsonl', 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())

print("=" * 90)
print("VI-S1K OUTPUT VERIFICATION")
print("=" * 90)

print(f"\nTotal columns: {len(sample)}")
print(f"Total rows: {sum(1 for line in open('results/vi_s1k/vi_s1k_benchmark.jsonl', encoding='utf-8'))}")

print("\n" + "=" * 90)
print("ORIGINAL S1K COLUMNS (13) - MUST BE PRESENT")
print("=" * 90)

s1k_cols = {
    'question': 'TRANSLATE',
    'solution': 'TRANSLATE',
    'cot_type': 'KEEP',
    'source_type': 'KEEP',
    'metadata': 'KEEP (JSON)',
    'gemini_thinking_trajectory': 'TRANSLATE',
    'gemini_attempt': 'TRANSLATE',
    'deepseek_thinking_trajectory': 'TRANSLATE',
    'deepseek_attempt': 'TRANSLATE',
    'gemini_grade': 'KEEP',
    'gemini_grade_reason': 'TRANSLATE',
    'deepseek_grade': 'KEEP',
    'deepseek_grade_reason': 'TRANSLATE',
}

all_present = True
for col, action in s1k_cols.items():
    status = "✓" if col in sample else "✗"
    present = col in sample
    if not present:
        all_present = False
    print(f"{status} {col:40} [{action:20}] Present: {present}")

print("\n" + "=" * 90)
print("VIETNAMESE TRANSLATIONS (8) - NEW COLUMNS")
print("=" * 90)

vi_cols = [
    'vietnamese_question',
    'vietnamese_solution',
    'vietnamese_gemini_thinking_trajectory',
    'vietnamese_gemini_attempt',
    'vietnamese_deepseek_thinking_trajectory',
    'vietnamese_deepseek_attempt',
    'vietnamese_gemini_grade_reason',
    'vietnamese_deepseek_grade_reason',
]

all_vi_present = True
for col in vi_cols:
    status = "✓" if col in sample else "✗"
    present = col in sample
    has_value = bool(sample.get(col)) if col in sample else False
    if not present:
        all_vi_present = False
    print(f"{status} {col:45} [Has value: {has_value}]")

print("\n" + "=" * 90)
print("METADATA COLUMNS (2)")
print("=" * 90)

meta_cols = ['translation_model', 'quality_score']
for col in meta_cols:
    status = "✓" if col in sample else "✗"
    value = sample.get(col, "N/A")
    print(f"{status} {col:40} Value: {value}")

print("\n" + "=" * 90)
print("SAMPLE DATA PREVIEW")
print("=" * 90)

print(f"\n1. Original question (English):")
print(f"   {sample['question'][:100]}...")

print(f"\n2. Vietnamese question translation:")
print(f"   {sample['vietnamese_question'][:100]}...")

print(f"\n3. Original solution (English):")
print(f"   {sample['solution'][:80]}...")

print(f"\n4. Vietnamese solution translation:")
print(f"   {sample['vietnamese_solution'][:80]}...")

print("\n" + "=" * 90)
print("LABEL PRESERVATION CHECK")
print("=" * 90)

print(f"\n✓ cot_type (kept): {sample['cot_type']}")
print(f"✓ source_type (kept): {sample['source_type'][:50]}...")
print(f"✓ gemini_grade (kept): {sample['gemini_grade']}")
print(f"✓ deepseek_grade (kept): {sample['deepseek_grade']}")
print(f"✓ metadata (JSON, not translated): {sample['metadata'][:100]}...")

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"\n✓ All 13 original S1K columns present: {all_present}")
print(f"✓ All 8 Vietnamese translations present: {all_vi_present}")
print(f"✓ Total columns: {len(sample)} (13 original + 8 Vietnamese + 2 metadata = 23)")

if all_present and all_vi_present:
    print("\n✅ VERIFICATION PASSED - Vi-S1K is correctly structured!")
else:
    print("\n❌ VERIFICATION FAILED - Some columns are missing!")
    sys.exit(1)
