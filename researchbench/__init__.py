"""
ResearchBench - An open-source research quality-control laboratory for machine-learning experiments.
"""

__version__ = '0.5.0'

from .dataset.profiler import profile_dataset
from .dataset.health import audit_dataset_health
from .evaluation.comparison import evaluate_models
from .evaluation.cross_validation import run_cross_validation
from .evaluation.stability import run_stability_analysis
from .audit.audit import perform_research_audit
from .reporting.html import generate_report

__all__ = [
    'profile_dataset',
    'audit_dataset_health',
    'evaluate_models',
    'run_cross_validation',
    'run_stability_analysis',
    'perform_research_audit',
    'generate_report'
]
