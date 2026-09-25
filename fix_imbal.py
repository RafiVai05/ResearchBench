import os

with open('researchbench/audit/imbalance.py', 'r', encoding='utf-8') as f:
    imbal = f.read()
    
imbal = imbal.replace('def check_imbalance(df: pd.DataFrame, target: str, task: str) -> list:', 'def check_imbalance(df: pd.DataFrame, target: str, task: str, config: dict = None) -> list:')

with open('researchbench/audit/imbalance.py', 'w', encoding='utf-8') as f:
    f.write(imbal)