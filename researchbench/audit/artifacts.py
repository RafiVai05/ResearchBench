import os
import glob
import re
import logging
import pandas as pd

def audit_artifacts(project_dir, history_data=None):
    report = {
        "inventory": [],
        "figure_checks": [],
        "table_checks": [],
        "consistency": []
    }
    
    # 1. Inventory
    figures = glob.glob(os.path.join(project_dir, "figures", "*.*"))
    tables = glob.glob(os.path.join(project_dir, "tables", "*.*"))
    methodology = os.path.join(project_dir, "methodology.pdf")
    results_pdf = os.path.join(project_dir, "results.pdf")
    
    report["inventory"] = {
        "figures_found": len(figures),
        "tables_found": len(tables),
        "methodology_found": os.path.exists(methodology),
        "results_found": os.path.exists(results_pdf)
    }
    
    # Extract latest run data for consistency checks
    latest = history_data[-1] if history_data else {}
    models_info = latest.get("models", {})
    
    # 2. Figure Checks
    try:
        import cv2
        import pytesseract
        HAS_VISION = True
    except ImportError:
        HAS_VISION = False
        report["figure_checks"].append({"warning": "opencv-python and pytesseract not installed. Install researchbench[artifacts] for automated figure text inspection."})
        
    if HAS_VISION:
        for fig in figures:
            if not fig.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            try:
                img = cv2.imread(fig)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray).lower()
                
                # Basic checks
                if "x" not in text and "y" not in text and "0." not in text:
                    report["figure_checks"].append({
                        "figure": os.path.basename(fig),
                        "severity": "WARNING",
                        "issue": "Possible missing axis labels or unreadable text detected via OCR."
                    })
                
                # Compare against metric names
                if models_info:
                    for m_name, m_data in models_info.items():
                        metric_name = m_data.get("cv", {}).get("metric", "").lower()
                        if metric_name and metric_name in text:
                            report["figure_checks"].append({
                                "figure": os.path.basename(fig),
                                "severity": "INFO",
                                "issue": f"Detected reference to metric '{metric_name}' in figure text."
                            })
            except Exception as e:
                pass
                
    # 3. Table checks
    for tab in tables:
        if tab.endswith(".csv"):
            try:
                df = pd.read_csv(tab)
                # Check for metric consistency
                if models_info:
                    for col in df.columns:
                        for m_name, m_data in models_info.items():
                            mean_val = m_data.get("cv", {}).get("mean")
                            if mean_val is not None:
                                # See if any cell is close to mean_val
                                for val in df[col]:
                                    try:
                                        if abs(float(val) - float(mean_val)) < 0.001:
                                            report["table_checks"].append({
                                                "table": os.path.basename(tab),
                                                "severity": "INFO",
                                                "issue": f"Verified value {val} matches ResearchBench {m_name} CV mean."
                                            })
                                    except:
                                        pass
            except:
                pass
                
    # 4. Methodology / Results consistency
    try:
        import fitz # PyMuPDF
        HAS_PDF = True
    except ImportError:
        HAS_PDF = False
        report["consistency"].append({"warning": "PyMuPDF (fitz) not installed. Cannot parse PDF text for consistency."})
        
    if HAS_PDF:
        for pdf_file in [methodology, results_pdf]:
            if os.path.exists(pdf_file):
                try:
                    doc = fitz.open(pdf_file)
                    text = ""
                    for page in doc:
                        text += page.get_text().lower() + "\n"
                        
                    # Check folds
                    if latest:
                        folds = latest.get("config", {}).get("evaluation", {}).get("cv", {}).get("folds", 5)
                        if f"{folds}-fold" not in text and f"{folds} fold" not in text:
                            report["consistency"].append({
                                "document": os.path.basename(pdf_file),
                                "severity": "WARNING",
                                "issue": f"Methodology describes a CV strategy, but {folds}-fold (used in experiment) was not clearly detected."
                            })
                except:
                    pass
                    
    return report