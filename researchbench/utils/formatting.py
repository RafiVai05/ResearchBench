def print_section(title: str):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

def print_audit_concerns(concerns: list):
    if not concerns:
        print("[OK] No obvious issues detected")
    else:
        for c in concerns:
            if "leakage" in c.lower() or "severe" in c.lower() or "missing" in c.lower():
                print(f"[WARNING] {c}")
            else:
                print(f"[INFO] {c}")