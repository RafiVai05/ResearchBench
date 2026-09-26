import pandas as pd
from researchbench.audit.audit import perform_research_audit
from researchbench.evaluation.comparison import evaluate_models
from researchbench.reporting.html import generate_report
from researchbench.advisor.advisor import run_advisor
import hashlib

class ResearchBenchProject:
    """
    Programmatic entrypoint for ResearchBench 1.0.0.
    Allows Jupyter Notebook users and backend services to interact with ResearchBench without the CLI.
    """
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        self.config = config
        self.target = config.get("target")
        self.task = config.get("task", "classification")
        self.audit_results = None
        self.model_results = None
        self.advice_results = None
        
    def hash_dataframe(self):
        """Hashes the DataFrame securely for lineage tracking."""
        return hashlib.sha256(pd.util.hash_pandas_object(self.df, index=True).values).hexdigest()

    def run(self):
        # 1. Audit
        self.audit_results = perform_research_audit(self.df, self.target, self.task, self.config)
        self.audit_results.setdefault("reproducibility", {})
        self.audit_results["reproducibility"]["dataset_hash"] = self.hash_dataframe()
        
        # Fairness
        from researchbench.audit.fairness import audit_fairness
        self.audit_results["fairness"] = audit_fairness(self.df, self.target, self.config, {})
        
        # 2. Evaluate
        models_list = list(self.config.get("models", {}).keys())
        self.model_results = evaluate_models(self.df.drop(columns=[self.target]), self.df[self.target], self.task, models_list, self.config)
        
        # Fairness (Post-model)
        oof_preds_dict = {m_name: m_data.get("cv", {}).get("oof_preds") for m_name, m_data in self.model_results.items() if m_data.get("cv")}
        self.audit_results["fairness"] = audit_fairness(self.df, self.target, self.config, oof_preds_dict)
        
        # Executive Insights
        from researchbench.reporting.insights import generate_executive_summary
        self.audit_results["executive_summary"] = generate_executive_summary(self.audit_results, self.model_results)
        
        # 3. Advice
        self.advice_results = run_advisor(self.audit_results, self.model_results, self.task)
        
        return {
            "audit": self.audit_results,
            "models": self.model_results,
            "advice": self.advice_results
        }
        
    def generate_report(self, output_path: str = "researchbench_programmatic_report.html"):
        if not self.audit_results:
            raise ValueError("Run the project first using .run()")
        dataset_name = self.config.get("dataset", "DataFrame")
        generate_report(self.audit_results, self.model_results, self.advice_results, output_path, dataset_name)
        return output_path
