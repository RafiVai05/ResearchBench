import yaml
import os

DEFAULT_CONFIG = {
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

    'calibration': {
        'enabled': True,
        'bins': 10
    },
    'feature_attribution': {
        'enabled': True,
        'method': 'permutation',
        'n_repeats': 10
    },
    'sensitivity_analysis': {
        'enabled': False,
        'parameters': {}
    },
    'drift_detection': {
        'enabled': False,
        'external_dataset': None
    },
    'artifact_audit': {
        'enabled': False,
        'figures_dir': 'figures/',
        'tables_dir': 'tables/',
        'methodology': 'methodology.pdf',
        'results': 'results.pdf'
    },
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
}

def deep_merge(target, source):
    for k, v in source.items():
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            deep_merge(target[k], v)
        else:
            target[k] = v
    return target

def load_config(path: str = None) -> dict:
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
            if user_config:
                deep_merge(config, user_config)
    return config