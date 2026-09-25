import os
import json
import pandas as pd

def export_results(history_path, formats, outdir):
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
        
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
        
    if not history:
        raise ValueError("History is empty")
        
    latest = history[-1]
    os.makedirs(outdir, exist_ok=True)
    
    # 1. Model comparison table
    models = latest.get('models', {})
    if models:
        records = []
        for m_name, m_data in models.items():
            record = {"Model": m_name}
            if 'cv' in m_data:
                record["Metric"] = m_data['cv'].get('metric', '')
                record["Mean"] = m_data['cv'].get('mean', 0)
                record["Std"] = m_data['cv'].get('std', 0)
                if m_data['cv'].get('ci_lower') is not None:
                    record["CI_Lower"] = m_data['cv']['ci_lower']
                    record["CI_Upper"] = m_data['cv']['ci_upper']
            records.append(record)
            
        df_models = pd.DataFrame(records)
        
        if "csv" in formats:
            df_models.to_csv(os.path.join(outdir, "models_comparison.csv"), index=False)
        if "latex" in formats:
            with open(os.path.join(outdir, "models_comparison.tex"), "w", encoding='utf-8') as f:
                f.write(df_models.style.format(precision=4).to_latex(hrules=True))
                
    # 2. Feature Attribution
    for m_name, m_data in models.items():
        attr = m_data.get('cv', {}).get('attribution')
        if attr:
            df_attr = pd.DataFrame({
                "Feature": attr["features"],
                "Importance_Mean": attr["importances_mean"],
                "Importance_Std": attr["importances_std"]
            }).sort_values("Importance_Mean", ascending=False)
            
            if "csv" in formats:
                df_attr.to_csv(os.path.join(outdir, f"{m_name}_attribution.csv"), index=False)
            if "latex" in formats:
                with open(os.path.join(outdir, f"{m_name}_attribution.tex"), "w", encoding='utf-8') as f:
                    f.write(df_attr.style.format(precision=4).to_latex(hrules=True))

    # 3. Statistical comparisons
    stats = latest.get('statistics', {})
    if stats and stats.get('comparisons'):
        df_stats = pd.DataFrame(stats['comparisons'])
        if "csv" in formats:
            df_stats.to_csv(os.path.join(outdir, "statistics.csv"), index=False)
        if "latex" in formats:
            with open(os.path.join(outdir, "statistics.tex"), "w", encoding='utf-8') as f:
                f.write(df_stats.style.format(precision=4).to_latex(hrules=True))
                
    # 4. Drift detection
    drift = latest.get('drift', {})
    if drift and drift.get('details'):
        df_drift = pd.DataFrame(drift['details'])
        if "csv" in formats:
            df_drift.to_csv(os.path.join(outdir, "drift_analysis.csv"), index=False)
        if "latex" in formats:
            with open(os.path.join(outdir, "drift_analysis.tex"), "w", encoding='utf-8') as f:
                f.write(df_drift.style.format(precision=4).to_latex(hrules=True))
                
    return f"Exported results to {outdir}"