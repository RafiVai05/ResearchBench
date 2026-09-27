import os

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
    <h2>3. Dataset Health Profile</h2>
    <div class="card">
        <ul>
            <li>Rows: {{ audit.health.profile.num_rows }}</li>
            <li>Columns: {{ audit.health.profile.num_cols }}</li>
            <li>Missing: {{ audit.health.profile.missing_percentage }}%</li>
            <li>Duplicates: {{ audit.health.profile.duplicate_rows }}</li>
        </ul>
        {% if plots.target_dist %}
        <div class="plot">
            <img src="data:image/png;base64,{{ plots.target_dist }}" alt="Target Distribution">
        </div>
        {% endif %}
        
        {% if audit.health.profile.correlation_heatmap %}
        <h3>Feature Correlation Matrix</h3>
        <p><em>Highlights collinearity and data leakage directly against the target variable.</em></p>
        <div class="plot">
            <img src="{{ audit.health.profile.correlation_heatmap }}" alt="Correlation Heatmap" style="max-width: 600px;">
        </div>
        {% endif %}
    </div>
'''

text = text.replace(
    '''    <h2>3. Dataset Health Profile</h2>
    <div class="card">
        <ul>
            <li>Rows: {{ audit.health.profile.num_rows }}</li>
            <li>Columns: {{ audit.health.profile.num_cols }}</li>
            <li>Missing: {{ audit.health.profile.missing_percentage }}%</li>
            <li>Duplicates: {{ audit.health.profile.duplicate_rows }}</li>
        </ul>
        {% if plots.target_dist %}
        <div class="plot">
            <img src="data:image/png;base64,{{ plots.target_dist }}" alt="Target Distribution">
        </div>
        {% endif %}
    </div>''',
    replacement.strip('\\n')
)

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(text)