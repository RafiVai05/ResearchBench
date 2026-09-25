import os

with open('researchbench/audit/audit.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    "def perform_research_audit(df, target, task):",
    "def perform_research_audit(df, target, task, config=None):"
)
with open('researchbench/audit/audit.py', 'w', encoding='utf-8') as f:
    f.write(content)