import os

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
    <h2>3. Dataset Health Profile</h2>
    <div class="card">
        <h3>Overview</h3>
        <p><strong>Rows:</strong> {{ audit.health.profile.num_rows }} | <strong>Columns:</strong> {{ audit.health.profile.num_cols }}</p>
        <p><strong>Numeric Features:</strong> {{ audit.health.profile.numeric_cols|length }} | <strong>Categorical:</strong> {{ audit.health.profile.categorical_cols|length }}</p>
        <p><strong>Missing Values:</strong> {{ audit.health.profile.missing_count }} ({{ "%.2f"|format(audit.health.profile.missing_percentage) }}%)</p>
        
        {% if plots.target_dist %}
        <h3>Target Distribution</h3>
        <div class="plot">
            <img src="{{ plots.target_dist }}" alt="Target Distribution">
        </div>
        {% endif %}
        
        {% if audit.health.profile.correlation_heatmap %}
        <h3>Feature Correlation Matrix</h3>
        <p><em>Highlights collinearity and data leakage directly against the target variable.</em></p>
        <div class="plot">
            <img src="{{ audit.health.profile.correlation_heatmap }}" alt="Correlation Heatmap">
        </div>
        {% endif %}
'''

# Find the exact string in report.html
text = text.replace(
    '''    <h2>3. Dataset Health Profile</h2>
    <div class="card">
        <h3>Overview</h3>
        <p><strong>Rows:</strong> {{ audit.health.profile.num_rows }} | <strong>Columns:</strong> {{ audit.health.profile.num_cols }}</p>
        <p><strong>Numeric Features:</strong> {{ audit.health.profile.numeric_cols|length }} | <strong>Categorical:</strong> {{ audit.health.profile.categorical_cols|length }}</p>
        <p><strong>Missing Values:</strong> {{ audit.health.profile.missing_count }} ({{ "%.2f"|format(audit.health.profile.missing_percentage) }}%)</p>
        
        {% if plots.target_dist %}
        <h3>Target Distribution</h3>
        <div class="plot">
            <img src="{{ plots.target_dist }}" alt="Target Distribution">
        </div>
        {% endif %}''',
    replacement.strip('\\n')
)

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(text)