#!/bin/bash

CONDA_ENV="ppm_llm"
# Use the user-provided activation command
CONDA_ACTIVATE="source ~/anaconda3/bin/activate $CONDA_ENV"

echo "========================================="
echo "Starting evaluation and report generation in conda env: $CONDA_ENV"
echo "========================================="

# --- Run Evaluation on Test Set ---
echo -e "\n--- Evaluating all trained models ---"
eval "$CONDA_ACTIVATE" && python scripts/evaluate_models.py

# --- Generate Final Report ---
echo -e "\n--- Generating final performance report ---"
eval "$CONDA_ACTIVATE" && python reports/generate_report.py

echo "========================================="
echo "Evaluation and reporting finished."
echo "Find summary at: results/evaluation_summary.csv"
echo "Find plot at: reports/performance_comparison.png"
echo "=========================================" 