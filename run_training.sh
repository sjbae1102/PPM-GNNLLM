#!/bin/bash

CONDA_ENV="ppm_llm"
# Use the user-provided activation command
CONDA_ACTIVATE="source ~/anaconda3/bin/activate $CONDA_ENV"

echo "========================================="
echo "Starting Experiment Pipeline in conda env: $CONDA_ENV"
echo "========================================="

# --- 1. Preprocess Data (Run Once) ---
echo -e "\n--- Step 1: Preprocessing Data ---"
eval "$CONDA_ACTIVATE" && python scripts/preprocess_data.py

# --- 2. Train All Models ---
echo -e "\n--- Step 2: Training Models ---"

# Train Baseline Models (Lightest)
echo -e "\n--- Training LSTM Model ---"
eval "$CONDA_ACTIVATE" && python scripts/train_base_models.py --model lstm

echo -e "\n--- Training GRU Model ---"
eval "$CONDA_ACTIVATE" && python scripts/train_base_models.py --model gru

echo -e "\n--- Training Transformer Model ---"
eval "$CONDA_ACTIVATE" && python scripts/train_base_models.py --model transformer

# Train LLM-based Models (Heavier)
echo -e "\n--- Training Frozen LLM (Feature Extractor) Model ---"
eval "$CONDA_ACTIVATE" && python scripts/train_llm_models.py --model llm_frozen

echo -e "\n--- Training Hybrid GNN-LLM Model ---"
eval "$CONDA_ACTIVATE" && python scripts/train_hybrid_model.py

echo -e "\n--- Training Fine-tuned LLM Model ---"
eval "$CONDA_ACTIVATE" && python scripts/train_llm_models.py --model llm_finetune


echo "========================================="
echo "All training scripts finished."
echo "=========================================" 