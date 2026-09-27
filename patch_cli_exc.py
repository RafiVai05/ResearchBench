import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('except Exception:\\n                pass', 'except Exception as e:\\n                print("EXCEPTION:", str(e))')

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(text)