import os

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
    <div class="section">
        <h2>Model Performance</h2>
        
        {% if models._global_stability_plot %}
        <div class="plot" style="margin-bottom: 30px;">
            <h3>Cross-Validation Stability (All Models)</h3>
            <img src="{{ models._global_stability_plot }}" alt="Model Stability Comparison" style="max-width: 800px;">
            <p style="font-size: 0.9em; color: #666;">This box plot shows the variance and distribution of the model's accuracy across the 5 cross-validation folds. Tighter boxes mean more stable, consistent performance.</p>
        </div>
        {% endif %}
'''

text = text.replace('    <div class="section">\n        <h2>Model Performance</h2>', replacement.strip('\\n'))

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(text)