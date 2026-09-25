import pytest
import pandas as pd
import numpy as np
from researchbench.dataset.profiler import profile_dataset

def test_profile_dataset():
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": ["x", "x", "x", "x"],
        "id": ["i1", "i2", "i3", "i4"]
    })
    prof = profile_dataset(df)
    assert prof["num_rows"] == 4
    assert prof["num_cols"] == 3
    assert prof["missing_count"] == 1
    assert "b" in prof["constant_cols"]
    assert "id" in prof["id_like_cols"]