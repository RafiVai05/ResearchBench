import os

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('models._global_stability_plot', 'advisor.global_stability_plot')

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(text)