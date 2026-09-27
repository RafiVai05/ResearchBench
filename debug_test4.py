import pandas as pd
from researchbench.evaluation.correlation_plot import generate_correlation_heatmap

df = pd.read_csv('examples/v108_data.csv')
res = generate_correlation_heatmap(df)
print("TYPE:", type(res))
if res is None:
    print("IT RETURNED NONE!")