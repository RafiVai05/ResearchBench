import os
import re

with open('researchbench/evaluation/classification.py', 'r', encoding='utf-8') as f:
    clf = f.read()

clf = clf.replace('def evaluate_classification_metrics(y_true, y_pred, y_prob=None):', 'def evaluate_classification_metrics(y_true, y_pred, y_prob=None, config=None):')
clf = clf.replace('    return metrics, cm', '''
    if config and "metrics" in config and "custom" in config["metrics"]:
        from researchbench.evaluation.metrics_loader import load_custom_metric
        for c_metric in config["metrics"]["custom"]:
            try:
                func = load_custom_metric(c_metric["path"], c_metric["function"])
                name = c_metric.get("name", c_metric["function"])
                metrics[f"USER-DEFINED METRIC ({name})"] = func(y_true, y_pred)
            except Exception as e:
                metrics[f"USER-DEFINED METRIC (Error)"] = str(e)
                
    return metrics, cm
''')

with open('researchbench/evaluation/classification.py', 'w', encoding='utf-8') as f:
    f.write(clf)

with open('researchbench/evaluation/regression.py', 'r', encoding='utf-8') as f:
    reg = f.read()

reg = reg.replace('def evaluate_regression_metrics(y_true, y_pred):', 'def evaluate_regression_metrics(y_true, y_pred, config=None):')
reg = reg.replace('    return metrics', '''
    if config and "metrics" in config and "custom" in config["metrics"]:
        from researchbench.evaluation.metrics_loader import load_custom_metric
        for c_metric in config["metrics"]["custom"]:
            try:
                func = load_custom_metric(c_metric["path"], c_metric["function"])
                name = c_metric.get("name", c_metric["function"])
                metrics[f"USER-DEFINED METRIC ({name})"] = func(y_true, y_pred)
            except Exception as e:
                metrics[f"USER-DEFINED METRIC (Error)"] = str(e)
                
    return metrics
''')

with open('researchbench/evaluation/regression.py', 'w', encoding='utf-8') as f:
    f.write(reg)

with open('researchbench/evaluation/cross_validation.py', 'r', encoding='utf-8') as f:
    cv = f.read()
    
cv = cv.replace('metrics, _ = evaluate_classification_metrics(y_test, preds, probs)', 'metrics, _ = evaluate_classification_metrics(y_test, preds, probs, config)')
cv = cv.replace('metrics = evaluate_regression_metrics(y_test, preds)', 'metrics = evaluate_regression_metrics(y_test, preds, config)')

# Also pass config to get_split call context
cv = cv.replace('delayed(_eval_fold)(model, *get_split(train_idx, test_idx), task, main_metric_name)', 'delayed(_eval_fold)(model, *get_split(train_idx, test_idx), task, main_metric_name, config)')
cv = cv.replace('def _eval_fold(model, X_train, X_test, y_train, y_test, task, main_metric_name):', 'def _eval_fold(model, X_train, X_test, y_train, y_test, task, main_metric_name, config=None):')

with open('researchbench/evaluation/cross_validation.py', 'w', encoding='utf-8') as f:
    f.write(cv)
    
with open('researchbench/evaluation/stability.py', 'r', encoding='utf-8') as f:
    stb = f.read()
    
stb = stb.replace('metrics, _ = evaluate_classification_metrics(y_test, preds, probs)', 'metrics, _ = evaluate_classification_metrics(y_test, preds, probs, config)')
stb = stb.replace('metrics = evaluate_regression_metrics(y_test, preds)', 'metrics = evaluate_regression_metrics(y_test, preds, config)')
stb = stb.replace('delayed(_eval_seed)(model, X, y, task, s, main_metric_name)', 'delayed(_eval_seed)(model, X, y, task, s, main_metric_name, config)')
stb = stb.replace('def _eval_seed(model, X, y, task, seed, main_metric_name):', 'def _eval_seed(model, X, y, task, seed, main_metric_name, config=None):')

with open('researchbench/evaluation/stability.py', 'w', encoding='utf-8') as f:
    f.write(stb)