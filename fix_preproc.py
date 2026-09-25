import os

with open('researchbench/evaluation/preprocessing.py', 'r', encoding='utf-8') as f:
    code = f.read()
    
# config is the full config, so we need to extract preprocessing
code = code.replace(
'''    if config is None:
        from researchbench.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG['preprocessing']''',
'''    if config is None:
        from researchbench.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    config = config.get("preprocessing", {})'''
)

with open('researchbench/evaluation/preprocessing.py', 'w', encoding='utf-8') as f:
    f.write(code)