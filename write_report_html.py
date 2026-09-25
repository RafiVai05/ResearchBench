import os

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    html = f.read()

# I will append new sections before the Reproducibility section
insertion = '''
    {% if audit.preproc_concerns %}
    <h2>Preprocessing Audit</h2>
    <div class="card">
        {% for p in audit.preproc_concerns %}
            <div class="concern">⚠️ {{ p }}</div>
        {% endfor %}
    </div>
    {% endif %}

    <h2>Hyperparameters</h2>
    <div class="card">
        <ul>
        {% for m_name, res in models.items() %}
            {% if res.best_params %}
                <li><strong>{{ m_name }}</strong>: {{ res.best_params }}</li>
            {% endif %}
        {% endfor %}
        </ul>
        <p><em>Highest observed CV score under the configured search.</em></p>
    </div>
    
    {% if plots.residuals %}
    <h2>Regression Residuals</h2>
    <div class="card">
        <div class="plot">
            <img src="data:image/png;base64,{{ plots.residuals }}" alt="Residuals">
        </div>
    </div>
    {% endif %}
'''

html = html.replace('<h2>4. Reproducibility & Limitations</h2>', insertion + '\n    <h2>Reproducibility & Limitations</h2>')

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(html)