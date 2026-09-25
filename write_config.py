import os

with open('researchbench/config.py', 'r', encoding='utf-8') as f:
    config = f.read()

# I will update DEFAULT_CONFIG
replacement = '''DEFAULT_CONFIG = {
    'dataset': None,
    'task': None,
    'target': None,
    'preprocessing': {
        'numerical': {
            'imputation': 'median',
            'scaling': 'standard'
        },
        'categorical': {
            'imputation': 'most_frequent',
            'encoding': 'onehot'
        }
    },
    'evaluation': {
        'cv': {
            'folds': 5
        },
        'seeds': [42, 123, 456],
        'n_jobs': 1
    },
    'models': {
        'logistic_regression': {},
        'random_forest': {}
    },
    'statistics': {
        'enabled': False,
        'method': 'wilcoxon',
        'correction': 'holm',
        'confidence_intervals': {
            'enabled': False,
            'level': 0.95,
            'bootstrap_samples': 1000
        }
    },
    'metrics': {
        'builtin': [],
        'custom': []
    },
    'audit': {
        'thresholds': {
            'missing_percentage': 20,
            'class_imbalance_ratio': 4,
            'feature_sample_ratio': 0.1,
            'high_cardinality_threshold': 50,
            'seed_count_warning': 3,
            'cv_variability_warning': 0.05
        }
    },
    'report': {
        'title': 'ResearchBench Report',
        'sections': [
            'overview', 'dataset', 'preprocessing', 'baseline', 
            'comparison', 'statistics', 'residuals', 'audit', 
            'advisor', 'reproducibility'
        ]
    }
}'''

import re
config = re.sub(r'DEFAULT_CONFIG = \{.*?\}\n\ndef', replacement + '\\n\\ndef', config, flags=re.DOTALL)

with open('researchbench/config.py', 'w', encoding='utf-8') as f:
    f.write(config)