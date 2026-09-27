import os

with open('tests/test_v108.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('open("researchbench-report.html", "r")', 'open("researchbench-report.html", "r", encoding="utf-8")')

with open('tests/test_v108.py', 'w', encoding='utf-8') as f:
    f.write(text)