import os

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    html = f.read()

replacement_th = '''
            <tr>
                <th>Model</th>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
                <th>95% CI</th>
            </tr>
'''

replacement_td = '''
            <tr>
                <td>{{ name }}</td>
                <td>{{ res.cv.metric }}</td>
                <td>{{ "%.4f"|format(res.cv.mean) }}</td>
                <td>{{ "%.4f"|format(res.cv.std) }}</td>
                <td>{{ "%.4f"|format(res.cv.min) }}</td>
                <td>{{ "%.4f"|format(res.cv.max) }}</td>
                <td>
                    {% if res.cv.ci_lower %}
                    [{{ "%.4f"|format(res.cv.ci_lower) }}, {{ "%.4f"|format(res.cv.ci_upper) }}]
                    {% else %}
                    N/A
                    {% endif %}
                </td>
            </tr>
'''

import re
html = re.sub(r'<tr>\s*<th>Model</th>\s*<th>Metric</th>.*?</tr>', replacement_th.strip(), html, flags=re.DOTALL)
html = re.sub(r'<tr>\s*<td>{{ name }}</td>\s*<td>{{ res.cv.metric }}</td>.*?</tr>', replacement_td.strip(), html, flags=re.DOTALL)

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(html)