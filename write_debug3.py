import os

with open('researchbench/evaluation/comparison.py', 'r', encoding='utf-8') as f:
    comp = f.read()

import re
# Just fix the indentation manually
with open('researchbench/evaluation/comparison.py', 'w', encoding='utf-8') as f:
    lines = comp.split('\\n')
    for i, line in enumerate(lines):
        if 'grid = config.get("models", {}).get(name, {})' in line:
            print("Found line:", repr(line))