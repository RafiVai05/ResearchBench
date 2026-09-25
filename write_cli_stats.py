import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

replacement = '''
        # Statistical comparisons
        from researchbench.audit.statistics import perform_statistical_comparison
        stats_results = perform_statistical_comparison(model_results, config)
        
        # Dataset Fingerprint
        from researchbench.dataset.fingerprint import generate_dataset_fingerprint
        fingerprint = generate_dataset_fingerprint(df, target)
        
        # HTML Report
        from researchbench.reporting.html import generate_html_report
        
        repro_info = get_reproducibility_info()
        repro_info["dataset_shape"] = [health["profile"]["num_rows"], health["profile"]["num_cols"]]
        repro_info["dataset_fingerprint"] = fingerprint
        repro_info["config"] = config
        
        # Model Comparison JSON
        comparison_record = {
            "dataset": fingerprint,
            "preprocessing": config.get("preprocessing", {}),
            "models": model_results,
            "statistics": stats_results,
            "residuals": residuals_data,
            "audit": {
                "health": health,
                "leakage": leakage,
                "reproducibility": repro_info
            },
            "advisor": advisor_report
        }
        
        if args.save:
            save_experiment(comparison_record)
            
        report_html = generate_html_report(comparison_record, config=config)
'''

import re
cli = re.sub(r'# HTML Report.*?report_html = generate_html_report\(comparison_record\)', replacement.strip(), cli, flags=re.DOTALL)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)