import re

with open('researchbench/evaluation/comparison.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the Auto-balancing section
new_balancing = '''
    # Auto-balancing (v1.1.0 Fix)
    balancing_info = {}
    if task == "classification":
        class_counts = pd.Series(y).value_counts(normalize=True)
        if len(class_counts) > 0 and class_counts.min() < 0.15:
            for name, model in models_to_run.items():
                applied = False
                # Direct estimator
                if hasattr(model, "class_weight"):
                    try:
                        setattr(model, "class_weight", "balanced")
                        applied = True
                    except Exception:
                        pass
                
                balancing_info[name] = "Applied" if applied else "Unsupported"
'''
text = re.sub(
    r'# Auto-balancing\s+if task == "classification":\s+class_counts = pd\.Series\(y\)\.value_counts\(normalize=True\)\s+if len\(class_counts\) > 0 and class_counts\.min\(\) < 0\.15:\s+for name, model in models_to_run\.items\(\):\s+if hasattr\(model, "class_weight"\):\s+setattr\(model, "class_weight", "balanced"\)',
    new_balancing.strip('\\n'),
    text,
    flags=re.DOTALL
)

with open('researchbench/evaluation/comparison.py', 'w', encoding='utf-8') as f:
    f.write(text)