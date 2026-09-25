import os

with open('researchbench/evaluation/preprocessing.py', 'r', encoding='utf-8') as f:
    prep = f.read()

replacement_get = '''
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_transformer(name: str):
    name = name.lower()
    if name == 'standard':
        return StandardScaler()
    elif name == 'minmax':
        return MinMaxScaler()
    elif name == 'robust':
        return RobustScaler()
    elif name == 'power':
        return PowerTransformer()
    elif name == 'polynomial':
        return PolynomialFeatures()
    elif name == 'log1p':
        return FunctionTransformer(np.log1p, validate=False)
    elif name == 'median':
        return SimpleImputer(strategy='median')
    elif name == 'mean':
        return SimpleImputer(strategy='mean')
    elif name == 'most_frequent':
        return SimpleImputer(strategy='most_frequent')
    elif name == 'onehot':
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    elif name == 'tfidf':
        return TfidfVectorizer()
    elif name == 'count':
        return CountVectorizer()
    else:
        raise ValueError(f"Unknown transformer: {name}")
'''

import re
prep = re.sub(r'def get_transformer\(name: str\):.*?\n\n', replacement_get.strip() + '\n\n', prep, flags=re.DOTALL)

replacement_build = '''
    transformers = []
    explicit_cols = set()
    
    text_conf = preproc_config.get('text', {})
    if isinstance(text_conf, dict) and 'columns' in text_conf:
        text_cols = text_conf['columns']
        vectorizer_name = text_conf.get('vectorizer', 'tfidf')
        
        for c in text_cols:
            if c in X.columns:
                transformers.append((f"text_{c}", get_transformer(vectorizer_name), c))
                explicit_cols.add(c)
                
    num_conf = preproc_config.get('numerical', {})
'''
prep = re.sub(r'    transformers = \[\]\s*# Track which columns have explicit transformers\s*explicit_cols = set\(\)\s*num_conf = preproc_config.get\(\'numerical\', \{\}\)', replacement_build.strip(), prep, flags=re.DOTALL)

with open('researchbench/evaluation/preprocessing.py', 'w', encoding='utf-8') as f:
    f.write(prep)