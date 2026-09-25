import os

with open('researchbench/evaluation/preprocessing.py', 'r', encoding='utf-8') as f:
    prep = f.read()

prep = prep.replace('\ntransformers = []\n    explicit_cols = set()', '\n    transformers = []\n    explicit_cols = set()')

with open('researchbench/evaluation/preprocessing.py', 'w', encoding='utf-8') as f:
    f.write(prep)