import os

with open('researchbench/audit/leakage.py', 'r', encoding='utf-8') as f:
    leakage = f.read()

replacement = '''
    if num_duplicates > 0:
        concerns.append(f"Found {num_duplicates} identical rows across splits. This is a severe leakage risk.")
        
    strategy = config.get("evaluation", {}).get("strategy", "stratified") if config else "stratified"
    time_col = config.get("evaluation", {}).get("time_column") if config else None
    
    if time_col and time_col in X_train.columns:
        if strategy != "time_series":
            concerns.append("POTENTIAL TEMPORAL LEAKAGE RISK: Time column detected but evaluation strategy is not 'time_series'. Randomized splitting may cause future data to predict the past.")
            
    group_col = config.get("evaluation", {}).get("group_column") if config else None
    if group_col and group_col in X_train.columns:
        if strategy != "group_kfold":
            concerns.append("POTENTIAL GROUP LEAKAGE RISK: Group column detected but evaluation strategy is not 'group_kfold'. Samples from the same group might leak across splits.")
'''

import re
leakage = re.sub(r'    if num_duplicates > 0:.*?This is a severe leakage risk\."\)', replacement.strip(), leakage, flags=re.DOTALL)

with open('researchbench/audit/leakage.py', 'w', encoding='utf-8') as f:
    f.write(leakage)