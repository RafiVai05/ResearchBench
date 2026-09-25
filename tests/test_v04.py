import pytest
import pandas as pd
import numpy as np

def test_subgroups():
    from researchbench.evaluation.subgroups import analyze_subgroups
    from sklearn.linear_model import LogisticRegression
    from researchbench.evaluation.preprocessing import build_model_pipeline
    
    df = pd.DataFrame({
        'feat1': np.random.randn(50),
        'sub': np.random.choice(['A', 'B'], 50),
        'target': np.random.randint(0, 2, 50)
    })
    X = df[['feat1', 'sub']]
    y = df['target']
    
    config = {"subgroups": ["sub"]}
    model = LogisticRegression()
    pipeline = build_model_pipeline(model, X, config)
    pipeline.fit(X, y)
    
    processed = {"lr": pipeline}
    
    res = analyze_subgroups({}, processed, X, y, "classification", config)
    assert "sub" in res
    assert "lr" in res["sub"]
    assert "A" in res["sub"]["lr"]
    assert "B" in res["sub"]["lr"]
    
def test_pytorch_wrapper():
    try:
        import torch
        import torch.nn as nn
        from researchbench.models.deep_learning import PyTorchWrapper
    except ImportError:
        pytest.skip("PyTorch not installed")
        
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
        def forward(self, x):
            return self.linear(x)
            
    model = SimpleModel()
    wrapper = PyTorchWrapper(model, config={"deep_learning": {"epochs": 1, "device": "cpu"}}, task="classification")
    
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = pd.Series([0, 1])
    
    wrapper.fit(X, y)
    preds = wrapper.predict(X)
    assert len(preds) == 2