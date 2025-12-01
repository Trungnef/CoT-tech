#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check actual S1K dataset structure"""

from datasets import load_dataset
import json

# Load a single sample
dataset = load_dataset("simplescaling/s1K-1.1", "default")
print("=" * 90)
print("ACTUAL S1K-1.1 DATASET STRUCTURE")
print("=" * 90)

print("\nAvailable splits:", list(dataset.keys()))

first_item = dataset['train'][0]
print(f"\nTotal columns in S1K: {len(first_item)}")

print("\nColumn List:")
for i, (key, value) in enumerate(first_item.items(), 1):
    value_type = type(value).__name__
    if isinstance(value, str):
        value_preview = value[:70].replace('\n', ' ')
    else:
        value_preview = str(value)[:70]
    print(f"{i:2}. {key:40} ({value_type:15}) {value_preview}...")

print("\n" + "=" * 90)
print("FULL COLUMN NAMES:")
print("=" * 90)
for key in first_item.keys():
    print(f"  - {key}")
