# GNN-LLM Hybrid Model for Next Activity Prediction in Business Processes

## 1. Overview

This project tackles the **Next Activity Prediction** task, a core challenge in the field of Business Process Management (BPM). It aims to implement and comparatively analyze various approaches, ranging from traditional RNN-based models to modern Large Language Model (LLM) based methods, and culminating in a novel GNN-LLM hybrid model that combines the strengths of both.

Based on event log data (e.g., BPI Challenge 2012, 2017), the model predicts the next likely activity given a sequence of previous activities in a process case.

## 2. Key Features

- **Implementation of Diverse Models:**
    - **Baseline Models:** LSTM, GRU, Transformer
    - **LLM-based Models:**
        - **Frozen LLM:** Uses a pre-trained LLM solely as a feature extractor.
        - **Fine-tuned LLM:** Fine-tunes the entire LLM on the specific dataset.
    - **Core Proposed Model:**
        - **GNN-LLM Hybrid Model:** A hybrid model that leverages a GNN to process structural process information and an LLM for sequential text information, creating synergy.

- **Flexible Experimentation Environment:**
    - Supports `BPI_Challenge_2012` and `BPI_Challenge_2017` datasets.
    - Easily switch datasets and training devices (GPUs) via command-line arguments.

- **Systematic Project Structure:**
    - Separate pipelines for data preprocessing, training, and evaluation.
    - Centralized configuration management through `src/config.py`.

## 3. Project Structure

```
PPM/
├── data/
│   ├── BPI_Challenge_2012.xes.gz
│   └── BPI_Challenge_2017.xes.gz
├── scripts/
│   ├── preprocess_data.py      # Data preprocessing script
│   ├── train_base_models.py    # Train LSTM, GRU, Transformer
│   ├── train_llm_models.py     # Train LLM (Frozen/Finetune)
│   ├── train_hybrid_model.py   # Train GNN-LLM Hybrid Model
│   └── evaluate_models.py      # Evaluate all trained models
├── src/
│   ├── config.py               # Core project configurations
│   ├── data_processing.py      # Data loading and processing logic
│   ├── engine.py               # Model training and validation loop
│   └── models/                 # Model architecture definitions
│       ├── base_models.py
│       ├── llm_models.py
│       └── hybrid_model.py
├── results/
│   ├── checkpoints/            # Trained model weights (.pt)
│   └── predictions/            # Model predictions (.csv)
├── .gitignore
├── README.md
├── requirements.txt
├── run_training.sh             # Shell script to run training for all models
└── run_evaluation.sh           # Shell script to run evaluation for all models
```


## 4. Getting Started

### 4.1. Environment Setup

1.  **Create and Activate Conda Environment**
    ```bash
    conda create -n ppm_llm python=3.10
    conda activate ppm_llm
    ```

2.  **Install Required Libraries**
    ```bash
    pip install -r requirements.txt
    ```

### 4.2. Experiment Pipeline

**Step 1: Data Preprocessing**

Before starting training, you must convert the original `.xes.gz` log files into `.pkl` format for training.

```bash
# Preprocess BPI 2012 dataset
python scripts/preprocess_data.py --dataset BPI_Challenge_2012.xes.gz

# Preprocess BPI 2017 dataset
python scripts/preprocess_data.py --dataset BPI_Challenge_2017.xes.gz
```

**Step 2: Model Training**

You can run the `run_training.sh` script to train all models on the default dataset specified in `config.py` (BPI 2012).

```bash
bash run_training.sh
```

Alternatively, you can run individual training scripts. For example, to train the hybrid model on the BPI 2017 dataset for 50 epochs on GPU 1:

```bash
# 1. Modify NUM_EPOCHS = 50 in the src/config.py file
# 2. Run the command below
python scripts/train_hybrid_model.py --dataset BPI_Challenge_2017.xes.gz --device cuda:1
```

**Step 3: Model Evaluation**

Run the `run_evaluation.sh` script to evaluate the performance of all models saved in `results/checkpoints` and generate prediction files.

```bash
bash run_evaluation.sh
```

## 5. Main Configurations

-   All core hyperparameters and paths can be modified in the `src/config.py` file.
-   **`LLM_MODEL_NAME`**: Specifies the pre-trained model to be used in LLM and hybrid models (e.g., `"facebook/opt-350m"`).
-   **`NUM_EPOCHS`**: Determines the number of training epochs.
-   **`BATCH_SIZE`**, **`LLM_BATCH_SIZE`**: Sets the batch size for different models.