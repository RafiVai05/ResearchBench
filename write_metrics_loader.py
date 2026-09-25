import os

content = '''import os
import sys
import importlib.util
from sklearn.metrics import make_scorer

def load_custom_metric(path: str, function_name: str):
    print(f"[WARNING] ResearchBench is executing a local custom metric script: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Custom metric file not found: {path}")
        
    spec = importlib.util.spec_from_file_location("custom_metric", path)
    if spec is None:
        raise ImportError(f"Could not load custom metric script from {path}")
        
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_metric"] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, function_name):
        raise AttributeError(f"Function {function_name} not found in {path}")
        
    func = getattr(module, function_name)
    return func
'''
with open('researchbench/evaluation/metrics_loader.py', 'w', encoding='utf-8') as f:
    f.write(content)