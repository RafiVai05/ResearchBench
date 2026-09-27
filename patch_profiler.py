import os

with open('researchbench/dataset/profiler.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
    target_distribution = None
    if target and target in df.columns:
        counts = df[target].value_counts(dropna=False)
        percentages = (counts / num_rows) * 100
        target_distribution = {
            "counts": counts.to_dict(),
            "percentages": percentages.to_dict()
        }
        
    # Feature Correlation Heatmap (v1.0.8)
    correlation_heatmap = None
    try:
        from researchbench.evaluation.correlation_plot import generate_correlation_heatmap
        correlation_heatmap = generate_correlation_heatmap(df)
    except Exception:
        pass

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_rows,
        "constant_cols": constant_cols,
        "id_like_cols": id_like_cols,
        "target_distribution": target_distribution,
        "correlation_heatmap": correlation_heatmap,
        "target": target
    }
'''

text = text.replace(
    '    target_distribution = None\\n    if target and target in df.columns:\\n        counts = df[target].value_counts(dropna=False)\\n        percentages = (counts / num_rows) * 100\\n        target_distribution = {\\n            "counts": counts.to_dict(),\\n            "percentages": percentages.to_dict()\\n        }\\n\\n    return {\\n        "num_rows": num_rows,\\n        "num_cols": num_cols,\\n        "numeric_cols": numeric_cols,\\n        "categorical_cols": categorical_cols,\\n        "missing_count": missing_count,\\n        "missing_percentage": missing_percentage,\\n        "duplicate_rows": duplicate_rows,\\n        "constant_cols": constant_cols,\\n        "id_like_cols": id_like_cols,\\n        "target_distribution": target_distribution,\\n        "target": target\\n    }',
    replacement.strip('\\n')
)

with open('researchbench/dataset/profiler.py', 'w', encoding='utf-8') as f:
    f.write(text)