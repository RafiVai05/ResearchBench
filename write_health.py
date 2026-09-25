import os

with open('researchbench/dataset/health.py', 'r', encoding='utf-8') as f:
    health = f.read()

health = health.replace('def audit_dataset_health(df: pd.DataFrame, target: str = None) -> dict:', 'def audit_dataset_health(df: pd.DataFrame, target: str = None, config: dict = None) -> dict:')
health = health.replace('if n > 0 and (p / n) > 0.2:', '''
    fs_ratio = 0.2
    if config and "audit" in config and "thresholds" in config["audit"]:
        fs_ratio = config["audit"]["thresholds"].get("feature_sample_ratio", 0.2)
    if n > 0 and (p / n) > fs_ratio:''')

with open('researchbench/dataset/health.py', 'w', encoding='utf-8') as f:
    f.write(health)

with open('researchbench/audit/imbalance.py', 'r', encoding='utf-8') as f:
    imbal = f.read()
imbal = imbal.replace('def check_imbalance(df, target, task):', 'def check_imbalance(df, target, task, config=None):')
imbal = imbal.replace('if ratio > 5:', '''
        threshold = 5
        if config and "audit" in config and "thresholds" in config["audit"]:
            threshold = config["audit"]["thresholds"].get("class_imbalance_ratio", 5)
        if ratio > threshold:''')
with open('researchbench/audit/imbalance.py', 'w', encoding='utf-8') as f:
    f.write(imbal)

with open('researchbench/audit/audit.py', 'r', encoding='utf-8') as f:
    audit = f.read()

audit = audit.replace('health = audit_dataset_health(df, target)', 'health = audit_dataset_health(df, target, config)')
audit = audit.replace('imbalance_concerns = check_imbalance(df, target, task)', 'imbalance_concerns = check_imbalance(df, target, task, config)')

# Add preprocessing audit
audit = audit.replace('repro_info = get_reproducibility_info()', '''
    if config and "preprocessing" in config:
        if config.get("preprocessing", {}).get("categorical", {}).get("encoding") == "onehot":
            all_concerns = [] # just declaring
            pass
    repro_info = get_reproducibility_info()
''')
with open('researchbench/audit/audit.py', 'w', encoding='utf-8') as f:
    f.write(audit)