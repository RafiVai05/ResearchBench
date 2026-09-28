#!/bin/bash
echo "Installing ResearchBench dependencies..."
pip install -r requirements.txt
pip install -e .
echo "ResearchBench setup complete."