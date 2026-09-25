import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

replacement = '''        elif args.command == "residuals":
            from researchbench.evaluation.residuals import analyze_residuals
            y = df[args.target]
            X = df.drop(columns=[args.target])
            res = analyze_residuals(X, y, args.model, config, args.preprocess)
            print_section("Regression Residual Analysis")
            print(f"Mean Residual: {res['mean']:.4f}")
            print(f"Std Residual:  {res['std']:.4f}")
            if res['concerns']:
                print("\\nObservations:")
                for c in res['concerns']:
                    print(f"- {c}")
            else:
                print("\\nNo obvious residual patterns detected.")'''

cli = cli.replace('''        elif args.command == "residuals":
            print("Residuals command executed.")''', replacement)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)