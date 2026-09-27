import os

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
    {% if advisor.global_stability_plot %}
    <h2>Cross-Validation Stability (All Models)</h2>
    <div class="card">
        <p><em>Comparison of score variance across all cross-validation folds.</em></p>
        <div class="plot">
            <img src="{{ advisor.global_stability_plot }}" alt="CV Stability Plot" style="max-width: 800px;">
        </div>
    </div>
    {% endif %}

    <h2>2. Model Comparison</h2>
'''

text = text.replace('    <h2>2. Model Comparison</h2>', replacement.strip('\\n'))

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(text)