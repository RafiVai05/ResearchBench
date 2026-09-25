import os
import re

with open('researchbench/config.py', 'r', encoding='utf-8') as f:
    config = f.read()

replacement = '''DEFAULT_CONFIG = {
    'dataset': None,
    'task': None,
    'target': None,
    'preprocessing': {
        'text': {},
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
        'strategy': 'stratified',
        'group_column': None,
        'time_column': None,
        'cv': {
            'folds': 5
        },
        'seeds': [42, 123, 456],
        'n_jobs': 1,
        'statistical_tests': {
            'enabled': False,
            'method': '5x2cv'
        }
    },
    'subgroups': [],
    'models': {
        'logistic_regression': {},
        'random_forest': {}
    },
    'deep_learning': {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': 'auto'
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
            'comparison', 'statistics', 'subgroup_analysis', 'residuals', 'audit', 
            'advisor', 'reproducibility'
        ]
    }
}'''

config = re.sub(r'DEFAULT_CONFIG = \{.*?\}\n\ndef', replacement + '\\n\\ndef', config, flags=re.DOTALL)

with open('researchbench/config.py', 'w', encoding='utf-8') as f:
    f.write(config)