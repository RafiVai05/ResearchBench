import argparse
import sys
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

def main():
    parser = argparse.ArgumentParser(description="ResearchBench - Quality control for ML experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Profile
    prof_parser = subparsers.add_parser("profile", help="Profile dataset")
    prof_parser.add_argument("dataset", help="Path to dataset CSV")
    prof_parser.add_argument("--target", help="Target column name", default=None)
    
    # Audit
    audit_parser = subparsers.add_parser("audit", help="Audit dataset health")
    audit_parser.add_argument("dataset")
    audit_parser.add_argument("--target", required=True)
    audit_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    
    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("dataset")
    eval_parser.add_argument("--target", required=True)
    eval_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    eval_parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest"])
    
    # CV
    cv_parser = subparsers.add_parser("cv", help="Run cross validation")
    cv_parser.add_argument("dataset")
    cv_parser.add_argument("--target", required=True)
    cv_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    cv_parser.add_argument("--model", required=True)
    cv_parser.add_argument("--folds", type=int, default=5)
    
    # Stability
    stab_parser = subparsers.add_parser("stability", help="Run seed stability analysis")
    stab_parser.add_argument("dataset")
    stab_parser.add_argument("--target", required=True)
    stab_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    stab_parser.add_argument("--model", required=True)
    stab_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    
    # Report
    rep_parser = subparsers.add_parser("report", help="Generate HTML report")
    rep_parser.add_argument("dataset")
    rep_parser.add_argument("--target", required=True)
    rep_parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    rep_parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest"])
    rep_parser.add_argument("--output", default="researchbench-report.html")
    rep_parser.add_argument("--json", default=None)
    
    args = parser.parse_args()
    
    try:
        df = load_dataset(args.dataset)
        
        if args.command == "profile":
            prof = profile_dataset(df, args.target)
            print_section("ResearchBench Dataset Profile")
            print(f"Rows: {prof['num_rows']}\nColumns: {prof['num_cols']}\n")
            print(f"Missing values: {prof['missing_percentage']:.2f}%")
            print(f"Duplicate rows: {prof['duplicate_rows']}")
            
        elif args.command == "audit":
            audit_res = perform_research_audit(df, args.target, args.task)
            print_section("RESEARCH QUALITY AUDIT")
            print_audit_concerns(audit_res["all_concerns"])
            
        elif args.command == "evaluate":
            y = df[args.target]
            import numpy as np
            X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
            # Impute missing values for basic evaluation safety
            X = X.fillna(X.mean())
            res = evaluate_models(X, y, args.task, args.models)
            print_section("Model Comparison")
            for m, vals in res.items():
                print(f"{m}: {vals['metrics']}")
                
        elif args.command == "cv":
            y = df[args.target]
            import numpy as np
            X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
            X = X.fillna(X.mean())
            from researchbench.evaluation.classification import get_classification_models
            from researchbench.evaluation.regression import get_regression_models
            available = get_classification_models() if args.task == "classification" else get_regression_models()
            model = available[args.model]
            res = run_cross_validation(model, X, y, args.task, args.folds)
            print_section("Cross Validation")
            print(f"Mean: {res['mean']:.4f} | Std: {res['std']:.4f}")
            
        elif args.command == "stability":
            y = df[args.target]
            import numpy as np
            X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
            X = X.fillna(X.mean())
            from researchbench.evaluation.classification import get_classification_models
            from researchbench.evaluation.regression import get_regression_models
            available = get_classification_models() if args.task == "classification" else get_regression_models()
            model = available[args.model]
            res = run_stability_analysis(model, X, y, args.task, args.seeds)
            print_section("Stability Analysis")
            print(f"Seeds: {args.seeds}")
            print(f"Mean: {res['mean']:.4f} | Std: {res['std']:.4f}")
            
        elif args.command == "report":
            y = df[args.target]
            import numpy as np
            X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
            X = X.fillna(X.mean())
            audit_res = perform_research_audit(df, args.target, args.task)
            model_res = evaluate_models(X, y, args.task, args.models)
            adv_res = run_advisor(audit_res, model_res, args.task)
            
            generate_report(audit_res, model_res, adv_res, args.output, args.dataset)
            print(f"Report generated at: {args.output}")
            
            if args.json:
                export_json({
                    "audit": audit_res,
                    "models": model_res,
                    "advisor": adv_res
                }, args.json)
                print(f"JSON generated at: {args.json}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()