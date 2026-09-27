import os

with open('researchbench/dataset/profiler.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('except Exception:\\n        pass', 'except Exception as e:\\n        import traceback\\n        traceback.print_exc()\\n        pass')

with open('researchbench/dataset/profiler.py', 'w', encoding='utf-8') as f:
    f.write(text)