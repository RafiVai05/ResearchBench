import re

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Patch conformal
replacement_conformal = '''
            # Conformal Prediction Intervals (v1.1.0)
            from researchbench.evaluation.conformal import calculate_conformal_bounds
            if task == "regression" and config.get("conformal", {}).get("enabled", False):
                processed_models = getattr(evaluate_models, "last_processed", {})
                for m_name, model_pipe in processed_models.items():
                    res = calculate_conformal_bounds(model_pipe, X, y, confidence_level=0.90)
                    if res:
                        model_res[m_name]["conformal"] = res
'''

text = re.sub(
    r'# Conformal Prediction Intervals\s*from researchbench\.evaluation\.conformal import calculate_conformal_bounds\s*if task == "regression" and config\.get\("conformal", \{\}\)\.get\("enabled", False\):\s*for m_name, m_data in model_res\.items\(\):\s*oof_y = m_data\.get\("cv", \{\}\)\.get\("oof_y"\)\s*oof_preds = m_data\.get\("cv", \{\}\)\.get\("oof_preds"\)\s*if oof_y and oof_preds:\s*model_res\[m_name\]\["conformal"\] = calculate_conformal_bounds\(oof_y, oof_preds, confidence_level=0\.90\)',
    replacement_conformal.strip('\\n'),
    text,
    flags=re.DOTALL
)

# Patch calibration
replacement_calib = '''
            # Calibration (v1.1.0)
            from researchbench.evaluation.calibration import calculate_probability_calibration
            if config.get("calibration", {}).get("enabled", False) and task == "classification":
                processed_models = getattr(evaluate_models, "last_processed", {})
                for m_name, model_pipe in processed_models.items():
                    res = calculate_probability_calibration(model_pipe, X, y)
                    if res:
                        model_res[m_name]["calibration"] = res
'''

text = re.sub(
    r'# Calibration\s*from researchbench\.evaluation\.calibration import calculate_calibration\s*if config\.get\("calibration", \{\}\)\.get\("enabled", False\) and task == "classification":\s*for m_name, m_data in model_res\.items\(\):\s*oof_y = m_data\.get\("cv", \{\}\)\.get\("oof_y"\)\s*oof_probs = m_data\.get\("cv", \{\}\)\.get\("oof_probs"\)\s*if oof_y and oof_probs:\s*m_data\["calibration"\] = calculate_calibration\(oof_y, oof_probs, n_bins=config\["calibration"\]\.get\("bins", 10\)\)',
    replacement_calib.strip('\\n'),
    text,
    flags=re.DOTALL
)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(text)