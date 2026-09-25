import os

with open('researchbench/config.py', 'r', encoding='utf-8') as f:
    config = f.read()
config = config.replace("'task': None,", "'dataset': None,\n    'task': None,")
with open('researchbench/config.py', 'w', encoding='utf-8') as f:
    f.write(config)

with open('examples/researchbench.yml', 'r', encoding='utf-8') as f:
    yml = f.read()
yml = "dataset: examples/classification.csv\n" + yml
with open('examples/researchbench.yml', 'w', encoding='utf-8') as f:
    f.write(yml)