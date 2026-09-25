import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

replacement = '''
            if not exp1_id or not exp2_id:
                print("Must provide two experiment IDs to compare.")
                return
                
            p1 = f".researchbench/experiments/{exp1_id}.json"
            p2 = f".researchbench/experiments/{exp2_id}.json"
            
            if not os.path.exists(p1) or not os.path.exists(p2):
                print("Experiment records not found.")
                return
                
            from researchbench.advisor.compare import compare_experiments
            comp = compare_experiments(p1, p2)
            print_section(f"Comparing EXPERIMENT {exp1_id} vs {exp2_id}")
            if comp["changes"]:
                print("Changes detected:")
                for c in comp["changes"]:
                    print(f"+ {c}")
            else:
                print("No major configuration changes detected.")
                
            # Advisor on comparison
            e1 = comp["exp1"]
            e2 = comp["exp2"]
            m1 = e1.get("models", {})
            m2 = e2.get("models", {})
            
            common_models = set(m1.keys()).intersection(set(m2.keys()))
            if common_models and comp["changes"]:
                print("\\nPossible interpretation:")
                print("The performance difference coincides with the observed changes. This association does not establish causality.")
                print("\\nSuggested investigation:")
                print("Repeat the experiment with identical random seeds and folds while changing only one configuration component at a time.")
            return
'''

import re
cli = re.sub(r'if not exp1_id or not exp2_id:.*?return', replacement.strip(), cli, flags=re.DOTALL)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)