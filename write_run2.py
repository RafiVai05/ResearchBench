import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

replacement = '''
        elif args.command == "run":
            if not config.get("dataset"):
                print("Dataset not specified in config.")
                sys.exit(1)
            if not config.get("target"):
                print("Target not specified in config.")
                sys.exit(1)
                
            task = config.get("task", "classification")
            df = load_dataset(config["dataset"])
            y = df[config["target"]]
            X = df.drop(columns=[config["target"]])
            
            models = list(config.get("models", {}).keys())
            if not models:
                models = ["logistic_regression", "random_forest"]
                
            audit_res = perform_research_audit(df, config["target"], task, config)
            model_res = evaluate_models(X, y, task, models, config=config, preprocess_mode="auto")
            adv_res = run_advisor(audit_res, model_res, task)
            
            print_section("RUN CONFIGURATION RESULTS")
            for m, vals in model_res.items():
                print(f"{m}: {vals['metrics']}")
                if 'best_params' in vals:
                    print(f"  Best Params: {vals['best_params']}")
            
            out_html = "researchbench-report.html"
            generate_report(audit_res, model_res, adv_res, out_html, config["dataset"])
            print(f"Report generated at: {out_html}")
            
            if args.save:
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "task": task,
                    "dataset_name": config["dataset"],
                    "models": model_res,
                    "audit": audit_res,
                    "advisor": adv_res
                }
                save_experiment(record)
                print("Experiment saved.")
                
        elif args.command == "residuals":'''

cli = cli.replace('''        elif args.command == "residuals":''', replacement)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)