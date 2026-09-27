import os
import re

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
            generate_report(audit_res, model_res, adv_res, out_html, config["dataset"])
            print(f"Report generated at: {out_html}")
            
            # Markdown Report (v1.0.5)
            try:
                from researchbench.reporting.markdown_report import generate_markdown_report
                out_md = out_html.replace(".html", ".md")
                generate_markdown_report(audit_res, model_res, adv_res, out_md, config["dataset"])
                print(f"Markdown artifact generated at: {out_md}")
            except Exception as e:
                pass
'''

text = text.replace(
    '            generate_report(audit_res, model_res, adv_res, out_html, config["dataset"])\n            print(f"Report generated at: {out_html}")',
    replacement.strip('\\n')
)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(text)