import yaml
import os

DEFAULT_CONFIG = {
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
        'seeds': [42, 123, 456]
    },
    'models': {
        'logistic_regression': {},
        'random_forest': {}
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