# ResearchBench

> *"An open-source research quality-control laboratory for machine-learning experiments."*

ResearchBench helps researchers answer: **"Can I actually trust the conclusions I'm drawing from this ML experiment?"**

It is NOT intended to be another basic ML metrics calculator or leaderboard. It examines an ML experiment from multiple perspectives, providing:
1. Raw measurements
2. Visual comparisons
3. Research-quality warnings
4. Evidence-based observations
5. Suggested investigations or improvements

## Features
- **Dataset Profiling**: Deep summary of missing values, constants, duplicates, and class imbalance.
- **Health & Leakage Audit**: Detect target copies, ID-like fields, and unrealistic train/test bleed.
- **Evaluation & Baselines**: Contextualizes logistic regression, random forests, and other models against simple baselines.
- **Stability & CV**: Evaluates standard deviation across seeds and cross-validation folds.
- **Research Advisor**: A deterministic, rule-based system generating factual observations and warnings.
- **HTML Reporting**: Generates standalone, publication-grade reports embedded with visual diagnostics.

## Quick Start
```bash
pip install researchbench
`

Run a dataset profile:
```bash
researchbench profile data.csv --target diagnosis
`

Audit the research quality of a dataset:
```bash
researchbench audit data.csv --target diagnosis --task classification
`

Evaluate models with stability and baselines:
```bash
researchbench evaluate data.csv --target diagnosis --task classification --models logistic_regression random_forest
`

Generate a comprehensive HTML report:
```bash
researchbench report data.csv --target diagnosis --task classification --models logistic_regression random_forest
`

## Contributing
See \CONTRIBUTING.md\.

## License
MIT