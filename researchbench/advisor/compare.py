import json

def compare_experiments(exp1_path: str, exp2_path: str):
    with open(exp1_path, 'r', encoding='utf-8') as f:
        exp1 = json.load(f)
    with open(exp2_path, 'r', encoding='utf-8') as f:
        exp2 = json.load(f)
        
    changes = []
    
    # Check dataset
    if exp1.get("dataset_name") != exp2.get("dataset_name"):
        changes.append(f"Dataset changed from {exp1.get('dataset_name')} to {exp2.get('dataset_name')}")
        
    # Check models
    m1 = set(exp1.get("models", {}).keys())
    m2 = set(exp2.get("models", {}).keys())
    if m1 != m2:
        changes.append(f"Models changed. Exp1: {m1}, Exp2: {m2}")
        
    # Check if onehot encoding or preprocessing changed
    # Actually, we can just say preprocessing changed if there are diffs in the config.
    
    return {
        "exp1_id": exp1.get("id"),
        "exp2_id": exp2.get("id"),
        "changes": changes,
        "exp1": exp1,
        "exp2": exp2
    }