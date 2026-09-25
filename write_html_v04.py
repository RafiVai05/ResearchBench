import os

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Add Subgroups right before audit
insert_subgroups = '''
    {% if 'subgroup_analysis' in visible_sections and subgroups %}
    <div class="card">
        <h2>Subgroup Performance Disparity</h2>
        {% for subgroup, models_data in subgroups.items() %}
            <h3>Subgroup: {{ subgroup }}</h3>
            {% for model_name, slice_data in models_data.items() %}
                <h4>Model: {{ model_name }}</h4>
                <table>
                    <tr>
                        <th>Group Value</th>
                        <th>N Samples</th>
                        <th>Metrics</th>
                    </tr>
                    {% for val, details in slice_data.items() %}
                    <tr>
                        <td>{{ val }}
                        {% if details.warning %}
                        <br><span style="color:red; font-size: 0.9em;">(POTENTIAL INSTABILITY: {{ details.warning }})</span>
                        {% endif %}
                        </td>
                        <td>{{ details.n_samples }}</td>
                        <td>
                            <ul>
                            {% for m_name, m_val in details.metrics.items() %}
                                {% if m_val is not none %}
                                <li>{{ m_name }}: {{ "%.4f"|format(m_val) }}</li>
                                {% endif %}
                            {% endfor %}
                            </ul>
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            {% endfor %}
        {% endfor %}
    </div>
    {% endif %}
'''

html = html.replace('<!-- ADVISOR -->', insert_subgroups + '\\n<!-- ADVISOR -->')

# Update Statistics warning to be cleaner for 5x2cv/McNemar
html = html.replace('''<p><em>Exploratory statistical comparison on CV folds. Does not establish generalized scientific supremacy.</em></p>''', '''<p><em>{{ statistics.warning }}</em></p>''')

# Add Contingency table for McNemar
html = html.replace('''<th>Adjusted p-value</th>
                <th>Mean Difference</th>
            </tr>''', '''<th>Adjusted p-value</th>
                <th>Mean Difference</th>
                <th>Contingency (McNemar)</th>
            </tr>''')

html = html.replace('''<td>{{ "%.4f"|format(comp.diff_mean) }}</td>
            </tr>''', '''<td>{{ "%.4f"|format(comp.diff_mean) }}</td>
                <td>
                    {% if comp.contingency %}
                        11:{{ comp.contingency.both_correct }} | 10:{{ comp.contingency.a_correct_b_wrong }} | 01:{{ comp.contingency.a_wrong_b_correct }} | 00:{{ comp.contingency.both_wrong }}
                    {% else %}
                        N/A
                    {% endif %}
                </td>
            </tr>''')

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(html)