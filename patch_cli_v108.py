import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
            adv_res = run_advisor(audit_res, model_res, config.get("task", args.task if hasattr(args, "task") else "classification"))
            
            # Model Stability Plot (v1.0.8)
            try:
                from researchbench.evaluation.stability_plot import generate_stability_plot
                stability_plot = generate_stability_plot(model_res)
                if stability_plot:
                    if adv_res is None:
                        adv_res = {}
                    adv_res["global_stability_plot"] = stability_plot
            except Exception:
                pass
'''

text = text.replace('            adv_res = run_advisor(audit_res, model_res, task)', replacement.strip('\\n'))
text = text.replace('            adv_res = run_advisor(audit_res, model_res, args.task)', replacement.strip('\\n'))

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(text)