import re

with open('researchbench/reporting/templates/report.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Add Plotly.js to <head>
text = text.replace('</head>', '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\\n</head>')

# Replace img tags with divs and plot logic
def replace_img_with_plotly(match):
    var_name = match.group(1)
    div_id = f"plot_{abs(hash(var_name))}"
    return f'''<div id="{div_id}" class="plotly-graph"></div>
    <script>
      var figure = {{{{ {var_name} | safe }}}};
      Plotly.newPlot("{div_id}", figure.data, figure.layout);
    </script>'''

text = re.sub(r'<img src="\{\{\s*(.*?)\s*\}\}"[^>]*>', replace_img_with_plotly, text)

with open('researchbench/reporting/templates/report.html', 'w', encoding='utf-8') as f:
    f.write(text)