import os

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Replace title
html = html.replace('<title>ResearchBench Quality Control Report</title>', '<title>{{ report_title }}</title>')
html = html.replace('<h1>ResearchBench Quality Control Report</h1>', '<h1>{{ report_title }}</h1>')

# Add visibility checks (simplified)
# Actually, I'll just insert the new fields.
# For fingerprint, in the dataset overview:
insert_fingerprint = '''
        {% if dataset.fingerprint %}
        <h3>Dataset Fingerprint</h3>
        <p><strong>Hash:</strong> {{ dataset.fingerprint.hash }}</p>
        {% endif %}
'''
html = html.replace('<h2>Dataset Profile</h2>', '<h2>Dataset Profile</h2>' + insert_fingerprint)

# Add statistics section
insert_stats = '''
    {% if 'statistics' in visible_sections and statistics %}
    <div class="card">
        <h2>Statistical Comparison</h2>
        <p><em>Exploratory statistical comparison on CV folds. Does not establish generalized scientific supremacy.</em></p>
        <p><strong>Method:</strong> {{ statistics.method }} (Correction: {{ statistics.correction }})</p>
        <table>
            <tr>
                <th>Model A</th>
                <th>Model B</th>
                <th>Statistic</th>
                <th>Raw p-value</th>
                <th>Adjusted p-value</th>
                <th>Mean Difference</th>
            </tr>
            {% for comp in statistics.comparisons %}
            <tr>
                <td>{{ comp.model_a }}</td>
                <td>{{ comp.model_b }}</td>
                <td>{{ "%.4f"|format(comp.stat) }}</td>
                <td>{{ "%.4f"|format(comp.p_value_raw) }}</td>
                <td>{{ "%.4f"|format(comp.p_value_adj) if comp.p_value_adj is not none else "N/A" }}</td>
                <td>{{ "%.4f"|format(comp.diff_mean) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
'''
html = html.replace('<!-- ADVISOR -->', insert_stats + '\n<!-- ADVISOR -->')

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(html)