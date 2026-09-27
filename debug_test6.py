with open("researchbench-report.html", "r", encoding="utf-8") as f:
    content = f.read()
    print("Len:", len(content))
    if "correlation_heatmap" in content:
        print("YES HEATMAP")
    print("Contains Matrix:", "Feature Correlation Matrix" in content)
    idx = content.find("Dataset Health Profile")
    print(content[idx:idx+500])