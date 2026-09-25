import os

with open('researchbench/evaluation/preprocessing.py', 'r', encoding='utf-8') as f:
    prep = f.read()

import re
# Look for explicit_cols = set() and fix indentation
# We can just run autopep8 or fix it by string split
lines = prep.split('\n')
for i, line in enumerate(lines):
    if 'explicit_cols = set()' in line:
        lines[i] = '    explicit_cols = set()'
    if 'num_conf = preproc_config.get(' in line:
        lines[i] = '    num_conf = preproc_config.get("numerical", {})'
        
with open('researchbench/evaluation/preprocessing.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))