from researchbench.evaluation.stability_plot import generate_stability_plot
model_res = {
    "model_a": {"cv": {"folds": [0.8, 0.85, 0.82]}},
    "model_b": {"cv": {"folds": [0.9, 0.91, 0.89]}}
}
print(generate_stability_plot(model_res)[:50] if generate_stability_plot(model_res) else "NONE!")