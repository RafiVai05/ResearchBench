import time
import sys
import pickle
import numpy as np

def profile_compute(model, X_test):
    """
    Measures the computational complexity and efficiency of a fitted model.
    """
    if len(X_test) == 0:
        return {}
        
    X_single = X_test.iloc[[0]] if hasattr(X_test, "iloc") else X_test[0:1]
    
    # Batch Latency
    start_time = time.perf_counter()
    model.predict(X_test)
    end_time = time.perf_counter()
    batch_latency = (end_time - start_time) / len(X_test)
    
    # Single Instance Latency
    start_time = time.perf_counter()
    model.predict(X_single)
    end_time = time.perf_counter()
    single_latency = (end_time - start_time)
    
    # Memory Footprint Estimation (via Pickling)
    try:
        memory_bytes = len(pickle.dumps(model))
    except Exception:
        memory_bytes = sys.getsizeof(model)
        
    # Model Complexity
    complexity = {}
    actual_model = model.steps[-1][1] if hasattr(model, "steps") else model
    
    if hasattr(actual_model, "coef_"):
        complexity["parameters"] = np.prod(actual_model.coef_.shape)
    elif hasattr(actual_model, "tree_"):
        complexity["nodes"] = actual_model.tree_.node_count
        complexity["depth"] = actual_model.tree_.max_depth
    elif hasattr(actual_model, "estimators_"):
        complexity["trees"] = len(actual_model.estimators_)
        if hasattr(actual_model.estimators_[0], "tree_"):
            complexity["avg_depth"] = np.mean([t.tree_.max_depth for t in actual_model.estimators_])
            complexity["total_nodes"] = sum([t.tree_.node_count for t in actual_model.estimators_])
            
    return {
        "batch_latency_ms": float(batch_latency * 1000),
        "single_latency_ms": float(single_latency * 1000),
        "memory_bytes": int(memory_bytes),
        "complexity": complexity
    }
