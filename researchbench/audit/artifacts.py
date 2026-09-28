import os
import glob
import json
import logging
import urllib.request
import urllib.parse
from urllib.error import URLError

def _call_ollama(prompt, model="llama2"):
    """
    Calls local Ollama API to extract structured claims.
    """
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode('utf-8'))
            return res.get("response", "{}")
    except Exception as e:
        return None

def audit_artifacts(project_dir, history_data=None):
    report = {
        "status": "deterministic_fallback",
        "inventory": [],
        "semantic_claims": [],
        "consistency": []
    }
    
    methodology = os.path.join(project_dir, "methodology.pdf")
    results_pdf = os.path.join(project_dir, "results.pdf")
    
    report["inventory"] = {
        "methodology_found": os.path.exists(methodology),
        "results_found": os.path.exists(results_pdf)
    }
    
    latest = history_data[-1] if history_data else {}
    config = latest.get("config", {})
    cv_folds = config.get("evaluation", {}).get("cv", {}).get("folds", 5)
    
    try:
        import fitz
        HAS_PDF = True
    except ImportError:
        HAS_PDF = False
        
    extracted_text = ""
    if HAS_PDF and os.path.exists(methodology):
        try:
            doc = fitz.open(methodology)
            for page in doc:
                extracted_text += page.get_text() + "\n"
        except Exception:
            pass

    if extracted_text:
        # Attempt LLM Structured Extraction
        prompt = f'''Extract validation strategy claims from the following text. 
Return ONLY a JSON object with this exact schema:
{{"claim_type": "cross_validation", "folds": <int>, "stratified": <bool>}}
Text: {extracted_text[:2000]}
'''
        llm_res = _call_ollama(prompt)
        if llm_res:
            try:
                claim = json.loads(llm_res)
                report["status"] = "semantic_verified"
                report["semantic_claims"].append(claim)
                
                # Consistency check
                if claim.get("folds") != cv_folds:
                    report["consistency"].append({
                        "severity": "POSSIBLE_DISCREPANCY",
                        "issue": f"Methodology claims {claim.get('folds')}-fold CV, but {cv_folds}-fold was used in configuration."
                    })
                else:
                    report["consistency"].append({
                        "severity": "VERIFIED",
                        "issue": f"Methodology claims {claim.get('folds')}-fold CV, matching configuration."
                    })
            except Exception:
                pass
                
        # Deterministic Fallback if LLM failed or not available
        if report["status"] == "deterministic_fallback":
            report["consistency"].append({
                "severity": "INFO",
                "issue": "Semantic verification unavailable (no local LLM). Performing deterministic artifact checks."
            })
            if f"{cv_folds}-fold" in extracted_text.lower():
                report["consistency"].append({
                    "severity": "INFO",
                    "issue": f"Direct textual evidence found: '{cv_folds}-fold' appears in methodology text. (Note: Not semantically verified)."
                })
            else:
                report["consistency"].append({
                    "severity": "INSUFFICIENT_EVIDENCE",
                    "issue": f"Unable to verify: '{cv_folds}-fold' not found via direct string matching."
                })
                
    return report