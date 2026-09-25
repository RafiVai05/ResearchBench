import pandas as pd
from researchbench.evaluation.preprocessing import build_model_pipeline
from researchbench.config import load_config
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('dummy.csv')
X = df.drop('target', axis=1)

config = load_config('examples/researchbench_v03.yml')
model = LogisticRegression()
pipeline = build_model_pipeline(model, X, config)

print("Pipeline steps:", getattr(pipeline, "steps", "Not a pipeline"))