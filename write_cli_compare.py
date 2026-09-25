import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

replacement = '''        if args.command == "compare":
            history_path = ".researchbench/history.json"
            if not os.path.exists(history_path):
                print("No history found.")
                return
            with open(history_path, "r") as f:
                history = json.load(f)
            
            exp1_id, exp2_id = args.exp1, args.exp2
            if args.latest:
                if len(history) >= 2:
                    exp1_id = history[-2]["id"]
                    exp2_id = history[-1]["id"]
            
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
                
            print(\"\nPossible interpretation:\")
            print(\"The performance difference coincides with the observed changes. This association does not establish causality.\")
            return'''

cli = cli.replace('''        if args.command == "compare":
            print("Comparison feature executed.")
            # Implementation for compare will go here...
            return''', replacement)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)