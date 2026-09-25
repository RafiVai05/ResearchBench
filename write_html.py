import os

with open('researchbench/reporting/html.py', 'r', encoding='utf-8') as f:
    html = f.read()

html = html.replace('def generate_html_report(experiment_data: dict) -> str:', 'def generate_html_report(experiment_data: dict, config: dict = None) -> str:')
html = html.replace('report_html = template.render(', '''
    report_config = config.get("report", {}) if config else {}
    report_title = report_config.get("title", "ResearchBench Quality Control Report")
    visible_sections = report_config.get("sections", [
        'overview', 'dataset', 'preprocessing', 'baseline', 
        'comparison', 'statistics', 'residuals', 'audit', 
        'advisor', 'reproducibility'
    ])
    
    report_html = template.render(
        report_title=report_title,
        visible_sections=visible_sections,
''')

with open('researchbench/reporting/html.py', 'w', encoding='utf-8') as f:
    f.write(html)