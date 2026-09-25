import os

with open('researchbench/evaluation/residuals.py', 'r', encoding='utf-8') as f:
    resid = f.read()

replacement = '''
    # Breusch-Pagan test
    bp_stat = None
    bp_pvalue = None
    try:
        import statsmodels.stats.api as sms
        import statsmodels.api as sm
        # Fit OLS on residuals to test heteroscedasticity
        exog = sm.add_constant(preds)
        bp_test = sms.het_breuschpagan(residuals, exog)
        bp_stat = float(bp_test[0])
        bp_pvalue = float(bp_test[1])
    except ImportError:
        pass
        
    return {
        "mean_residual": float(mean_res),
        "std_residual": float(std_res),
        "large_residual_ratio": float(large_ratio),
        "variance_ratio_high_low": float(variance_ratio),
        "flags": flags,
        "breusch_pagan_stat": bp_stat,
        "breusch_pagan_pvalue": bp_pvalue,
        "raw_preds": preds.tolist(),
        "raw_residuals": residuals.tolist()
    }
'''

resid = resid.replace('''
    return {
        "mean_residual": float(mean_res),
        "std_residual": float(std_res),
        "large_residual_ratio": float(large_ratio),
        "variance_ratio_high_low": float(variance_ratio),
        "flags": flags
    }''', replacement)

with open('researchbench/evaluation/residuals.py', 'w', encoding='utf-8') as f:
    f.write(resid)