import pytest
import pandas as pd
from researchbench.audit.leakage import check_leakage

def test_check_leakage():
    df = pd.DataFrame({
        "feat": [1, 2, 3],
        "target": [1, 2, 3]
    })
    concerns = check_leakage(df, "target")
    assert any("exact copy" in c for c in concerns)