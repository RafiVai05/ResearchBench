import os
from jinja2 import Environment, FileSystemLoader
from .visualizer import plot_target_distribution, plot_cv_boxplot

def generate_report(audit_results: dict, model_results: dict, advisor_results: dict, output_path: str, dataset_name: str):
    # Generate plots
    target = audit_results["reproducibility"]["target"]
    target_dist = audit_results["distribution"]
    
    plots = {
        "target_dist": plot_target_distribution(audit_results["health"]["profile"].get("target_distribution", {}), target),
        "cv_boxplot": plot_cv_boxplot(model_results)
    }
    
    from .visualizer import generate_residual_plots
    if audit_results["reproducibility"]["task"] == "regression":
        pass
        
    from .visualizer import plot_calibration_curve, plot_permutation_importance
    for m_name, m_data in model_results.items():
        if m_data.get("calibration"):
            cal_plot = plot_calibration_curve(m_data["calibration"], m_name, None)
            m_data["calibration"]["plot"] = cal_plot
            
        if m_data.get("cv", {}).get("attribution"):
            attr_plot = plot_permutation_importance(m_data["cv"]["attribution"], m_name, None)
            m_data["cv"]["attribution"]["plot"] = attr_plot

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html")
    
    html_content = template.render(
        dataset_name=dataset_name,
        audit=audit_results,
        models=model_results,
        advisor=advisor_results,
        plots=plots
    )
    
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(html_content)
