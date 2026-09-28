import re

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
    {% if audit.artifact_audit %}
    <div class="card">
        <h2>Research Artifact Audit (v1.1.0)</h2>
        <p><em>Status: {{ audit.artifact_audit.status }}</em></p>
        
        <h3>Artifact Inventory</h3>
        <ul>
            <li>Methodology PDF: {{ "Found" if audit.artifact_audit.inventory.methodology_found else "Not found" }}</li>
            <li>Results PDF: {{ "Found" if audit.artifact_audit.inventory.results_found else "Not found" }}</li>
        </ul>
        
        {% if audit.artifact_audit.semantic_claims %}
        <h3>Extracted Semantic Claims</h3>
        <ul>
            {% for claim in audit.artifact_audit.semantic_claims %}
            <li><strong>{{ claim.claim_type }}</strong>: {{ claim }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        
        {% if audit.artifact_audit.consistency %}
        <h3>Consistency Engine Findings</h3>
        <ul>
            {% for chk in audit.artifact_audit.consistency %}
            <li><strong>[{{ chk.severity }}]</strong> {{ chk.issue }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endif %}
'''

text = re.sub(
    r'\{% if audit\.artifact_audit %\}.*?<h3>Methodology / Results Consistency</h3>.*?</ul>\s*\{% endif %\}\s*</div>\s*\{% endif %\}',
    replacement.strip('\\n'),
    text,
    flags=re.DOTALL
)

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(text)