import argparse
import sys
import os
import json
import uuid
from datetime import datetime

from researchbench.dataset.loader import load_dataset
from researchbench.dataset.profiler import profile_dataset
from researchbench.dataset.health import audit_dataset_health
from researchbench.evaluation.comparison import evaluate_models
from researchbench.evaluation.cross_validation import run_cross_validation
from researchbench.evaluation.stability import run_stability_analysis
from researchbench.audit.audit import perform_research_audit
from researchbench.advisor.advisor import run_advisor
from researchbench.reporting.html import generate_report
from researchbench.reporting.json import export_json
from researchbench.utils.formatting import print_section, print_audit_concerns
from researchbench.config import load_config
from researchbench.evaluation.preprocessing import build_model_pipeline


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def save_experiment(record, filename=None):
    os.makedirs(".researchbench/experiments", exist_ok=True)
    if not filename:
        record["id"] = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        filename = f".researchbench/experiments/{record['id']}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4, cls=NpEncoder)

        
    # Update history index
    history_path = ".researchbench/history.json"
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
            
    history.append({
        "id": record["id"],
        "date": record.get("timestamp", ""),
        "task": record.get("task", ""),
        "dataset": record.get("dataset_name", ""),
        "models": list(record.get("models", {}).keys())
    })
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    return record["id"]

def get_common_parser():
    parser = argparse.ArgumentParser(description="ResearchBench - Quality control for ML experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Profile
    prof_parser = subparsers.add_parser("profile", help="Profile dataset")
    prof_parser.add_argument("dataset")
    prof_parser.add_argument("--target")
    
    # Audit
    audit_parser = subparsers.add_parser("audit", help="Audit dataset health")
    audit_parser.add_argument("dataset")
    audit_parser.add_argument("--target", required=True)
    audit_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    audit_parser.add_argument("--config")
    
    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("dataset")
    eval_parser.add_argument("--target", required=True)
    eval_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    eval_parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest"])
    eval_parser.add_argument("--preprocess", default="auto")
    eval_parser.add_argument("--config")
    eval_parser.add_argument("--save", action="store_true")
    
    # CV
    cv_parser = subparsers.add_parser("cv", help="Run cross validation")
    cv_parser.add_argument("dataset")
    cv_parser.add_argument("--target", required=True)
    cv_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    cv_parser.add_argument("--model", required=True)
    cv_parser.add_argument("--folds", type=int, default=5)
    cv_parser.add_argument("--preprocess", default="auto")
    cv_parser.add_argument("--config")
    
    # Stability
    stab_parser = subparsers.add_parser("stability", help="Run seed stability analysis")
    stab_parser.add_argument("dataset")
    stab_parser.add_argument("--target", required=True)
    stab_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    stab_parser.add_argument("--model", required=True)
    stab_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    stab_parser.add_argument("--preprocess", default="auto")
    stab_parser.add_argument("--config")
    
    # Report
    rep_parser = subparsers.add_parser("report", help="Generate HTML report")
    rep_parser.add_argument("dataset")
    rep_parser.add_argument("--target", required=True)
    rep_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    rep_parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest"])
    rep_parser.add_argument("--output", default="researchbench-report.html")
    rep_parser.add_argument("--json", default=None)
    rep_parser.add_argument("--preprocess", default="auto")
    rep_parser.add_argument("--config")
    rep_parser.add_argument("--save", action="store_true")

    # History
    hist_parser = subparsers.add_parser("history", help="View experiment history")
    hist_parser.add_argument("--show", help="Show specific experiment ID")
    hist_parser.add_argument("--clear", action="store_true", help="Clear all history")

    # Compare
    comp_parser = subparsers.add_parser("compare", help="Compare two experiments")
    comp_parser.add_argument("exp1", nargs="?", help="First experiment ID")
    comp_parser.add_argument("exp2", nargs="?", help="Second experiment ID")
    comp_parser.add_argument("--latest", type=int, help="Compare latest N experiments")

    # Run
    run_parser = subparsers.add_parser("run", help="Run experiment from config")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--folds", type=int)
    run_parser.add_argument("--save", action="store_true")

    # Residuals
    resid_parser = subparsers.add_parser("residuals", help="Analyze regression residuals")
    resid_parser.add_argument("dataset")
    resid_parser.add_argument("--target", required=True)
    resid_parser.add_argument("--model", required=True)
    resid_parser.add_argument("--preprocess", default="auto")
    resid_parser.add_argument("--config")
    
# Export parser
    parser_export = subparsers.add_parser("export", help="Export results to CSV/LaTeX")
    parser_export.add_argument("history_file", type=str, help="Path to history JSON file (e.g., .researchbench/history.json)")
    parser_export.add_argument("--formats", type=str, default="csv,latex", help="Comma-separated formats (e.g. csv,latex)")
    parser_export.add_argument("--outdir", type=str, default="exports", help="Output directory")
    
    # Audit artifacts parser
    parser_audit_art = subparsers.add_parser("audit-artifacts", help="Audit research artifacts (figures, tables, manuscripts)")
    parser_audit_art.add_argument("--project-dir", type=str, default=".", help="Project directory to scan")
    parser_audit_art.add_argument("--run-file", type=str, default=".researchbench/history.json", help="Path to ResearchBench history JSON")
    
    return parser

def execute_cli():
    parser = get_common_parser()
    args = parser.parse_args()
    
    try:
        config = load_config(getattr(args, 'config', None))
        
        if args.command == "history":
            history_path = ".researchbench/history.json"
            if args.clear:
                confirm = input("Are you sure you want to clear history? (y/N): ")
                if confirm.lower() == "y":
                    if os.path.exists(history_path):
                        os.remove(history_path)
                    import shutil
                    if os.path.exists(".researchbench/experiments"):
                        shutil.rmtree(".researchbench/experiments")
                    print("History cleared.")
                return
            if args.show:
                exp_file = f".researchbench/experiments/{args.show}.json"
                if os.path.exists(exp_file):
                    with open(exp_file, "r") as f:
                        print(json.dumps(json.load(f), indent=2))
                else:
                    print("Experiment not found.")
                return
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
                print_section("ResearchBench Experiment History")
                print(f"{'ID':<30} {'Date':<20} {'Task':<15} {'Models'}")
                for h in history:
                    models_str = ", ".join(h.get("models", []))
                    print(f"{h['id']:<30} {h['date']:<20} {h['task']:<15} {models_str}")
            else:
                print("No history found.")
            return


        if args.command == "export":
            from researchbench.reporting.export import export_results
            if not os.path.exists(args.history_file):
                print(f"History file not found: {args.history_file}")
                return
            export_results(args.history_file, args.formats.split(","), args.outdir)
            print(f"Exported results to {args.outdir}")
            return
            
        if args.command == "audit-artifacts":
            from researchbench.audit.artifacts import audit_artifacts
            history = []
            if os.path.exists(args.run_file):
                import json
                with open(args.run_file, "r") as f:
                    history = json.load(f)
            report = audit_artifacts(args.project_dir, history)
            import json
            print(json.dumps(report, indent=2))
            return
            
        if args.command == "compare":

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
                


        df = None
        if hasattr(args, "dataset") and args.dataset:
            df = load_dataset(args.dataset)
            
        if args.command == "profile":
            prof = profile_dataset(df, args.target)
            print_section("ResearchBench Dataset Profile")
            print(f"Rows: {prof['num_rows']}\nColumns: {prof['num_cols']}\n")
            print(f"Missing values: {prof['missing_percentage']:.2f}%")
            print(f"Duplicate rows: {prof['duplicate_rows']}")
            
        elif args.command == "audit":
            audit_res = perform_research_audit(df, args.target, args.task, config)
            print_section("RESEARCH QUALITY AUDIT")
            print_audit_concerns(audit_res["all_concerns"])
            
        elif args.command == "evaluate":
            y = df[args.target]
            X = df.drop(columns=[args.target])
            res = evaluate_models(X, y, args.task, args.models, config=config, preprocess_mode=args.preprocess)
            print_section("Model Comparison")
            for m, vals in res.items():
                print(f"{m}: {vals['metrics']}")
                
            if args.save:
                import researchbench
                save_experiment({
                    "timestamp": datetime.now().isoformat(),
                    "researchbench_version": researchbench.__version__,
                    "task": args.task,
                    "models": res
                })
                print("Experiment saved.")
                
        elif args.command == "cv":
            y = df[args.target]
            X = df.drop(columns=[args.target])
            from researchbench.evaluation.classification import get_classification_models
            from researchbench.evaluation.regression import get_regression_models
            available = get_classification_models() if args.task == "classification" else get_regression_models()
            model = available[args.model]
            pipeline = build_model_pipeline(model, X, config, args.preprocess)
            res = run_cross_validation(pipeline, X, y, args.task, args.folds)
            print_section("Cross Validation")
            print(f"Mean: {res['mean']:.4f} | Std: {res['std']:.4f}")
            
        elif args.command == "stability":
            y = df[args.target]
            X = df.drop(columns=[args.target])
            from researchbench.evaluation.classification import get_classification_models
            from researchbench.evaluation.regression import get_regression_models
            available = get_classification_models() if args.task == "classification" else get_regression_models()
            model = available[args.model]
            pipeline = build_model_pipeline(model, X, config, args.preprocess)
            res = run_stability_analysis(pipeline, X, y, args.task, args.seeds)
            print_section("Stability Analysis")
            print(f"Seeds: {args.seeds}")
            print(f"Mean: {res['mean']:.4f} | Std: {res['std']:.4f}")
            
        elif args.command == "report":
            y = df[args.target]
            X = df.drop(columns=[args.target])
            audit_res = perform_research_audit(df, args.target, args.task, config)
            model_res = evaluate_models(X, y, args.task, args.models, config=config, preprocess_mode=args.preprocess)
            adv_res = run_advisor(audit_res, model_res, args.task)
            
            generate_report(audit_res, model_res, adv_res, args.output, args.dataset)
            print(f"Report generated at: {args.output}")
            
            record = {
                "audit": audit_res,
                "models": model_res,
                "advisor": adv_res
            }
            if args.json:
                export_json(record, args.json)
                print(f"JSON generated at: {args.json}")
            
            if args.save:
                record["timestamp"] = datetime.now().isoformat()
                record["task"] = args.task
                save_experiment(record)
                

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
            
            # v0.5 Additions
            # Calibration
            from researchbench.evaluation.calibration import calculate_calibration
            if config.get("calibration", {}).get("enabled", False) and task == "classification":
                for m_name, m_data in model_res.items():
                    oof_y = m_data.get("cv", {}).get("oof_y")
                    oof_probs = m_data.get("cv", {}).get("oof_probs")
                    if oof_y and oof_probs:
                        m_data["calibration"] = calculate_calibration(oof_y, oof_probs, n_bins=config["calibration"].get("bins", 10))
            
            # Sensitivity
            from researchbench.evaluation.sensitivity import run_sensitivity_analysis
            sensitivity_results = {}
            if config.get("sensitivity_analysis", {}).get("enabled", False):
                processed_models = getattr(evaluate_models, "last_processed", {})
                for m_name, model_pipe in processed_models.items():
                    res = run_sensitivity_analysis(model_pipe, X, y, task, config)
                    if res: sensitivity_results[m_name] = res
            audit_res["sensitivity"] = sensitivity_results
            
            # Drift
            from researchbench.audit.drift import detect_drift
            audit_res["drift"] = detect_drift(X, config)
                        # Artifact Audit
            audit_res["artifact_audit"] = {}
            if config.get("artifact_audit", {}).get("enabled", False):
                from researchbench.audit.artifacts import audit_artifacts
                history_mock = [{"models": model_res, "config": config}]
                audit_res["artifact_audit"] = audit_artifacts(".", history_mock)
                
            # v0.6 Additions
            # Compute Profile
            from researchbench.evaluation.compute import profile_compute
            if config.get("compute_profiling", {}).get("enabled", False):
                processed_models = getattr(evaluate_models, "last_processed", {})
                for m_name, model_pipe in processed_models.items():
                    model_res[m_name]["compute"] = profile_compute(model_pipe, X)
                    
            # Robustness
            from researchbench.evaluation.robustness import evaluate_robustness
            if config.get("robustness", {}).get("enabled", False):
                processed_models = getattr(evaluate_models, "last_processed", {})
                for m_name, model_pipe in processed_models.items():
                    model_res[m_name]["robustness"] = evaluate_robustness(model_pipe, X, y, task, config)
                    
            # Hard Examples
            from researchbench.evaluation.typology import mine_hard_examples
            if config.get("hard_examples", {}).get("enabled", False):
                for m_name, m_data in model_res.items():
                    oof_y = m_data.get("cv", {}).get("oof_y")
                    oof_preds = m_data.get("cv", {}).get("oof_preds")
                    oof_probs = m_data.get("cv", {}).get("oof_probs")
                    if oof_y and oof_preds:
                        model_res[m_name]["hard_examples"] = mine_hard_examples(oof_y, oof_preds, oof_probs, X, task)
                        
            # Model Agreement
            from researchbench.evaluation.agreement import evaluate_agreement
            if config.get("agreement", {}).get("enabled", False) and len(model_res) > 1:
                preds_dict = {}
                true_y = None
                for m_name, m_data in model_res.items():
                    oof_y = m_data.get("cv", {}).get("oof_y")
                    oof_preds = m_data.get("cv", {}).get("oof_preds")
                    if oof_preds:
                        preds_dict[m_name] = oof_preds
                        if true_y is None: true_y = oof_y
                if true_y is not None:
                    audit_res["agreement"] = evaluate_agreement(preds_dict, true_y, task)
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
                
        elif args.command == "residuals":
            from researchbench.evaluation.residuals import analyze_residuals
            y = df[args.target]
            X = df.drop(columns=[args.target])
            res = analyze_residuals(X, y, args.model, config, args.preprocess)
            print_section("Regression Residual Analysis")
            print(f"Mean Residual: {res['mean']:.4f}")
            print(f"Std Residual:  {res['std']:.4f}")
            if res['concerns']:
                print("\nObservations:")
                for c in res['concerns']:
                    print(f"- {c}")
            else:
                print("\nNo obvious residual patterns detected.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    execute_cli()
