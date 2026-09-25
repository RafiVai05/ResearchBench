from .rules import apply_rules

def run_advisor(audit_results: dict, model_results: dict, task: str) -> dict:
    """
    Run the deterministic, rule-based Research Advisor.
    It produces:
    - observations
    - potential_concerns
    - suggested_investigations
    - reproducibility_notes
    - limitations
    """
    rules_out = apply_rules(audit_results, model_results, task)
    
    repro_notes = [
        f"Results were evaluated using {audit_results['reproducibility']['python_version']} on {audit_results['reproducibility']['os']}."
    ]
    
    limitations = [
        "ResearchBench provides automated heuristics and cannot substitute domain expertise.",
        "A lack of warnings does not guarantee that a research methodology is perfectly robust.",
        "Model evaluation on a single dataset does not guarantee out-of-distribution generalization."
    ]
    
    return {
        "observations": rules_out["observations"],
        "potential_concerns": rules_out["concerns"],
        "suggested_investigations": rules_out["suggestions"],
        "reproducibility_notes": repro_notes,
        "limitations": limitations
    }