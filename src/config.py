import torch

# --- Dirs ---
DATA_DIR = "data/"
RESULTS_DIR = "results/"
CHECKPOINT_DIR = f"{RESULTS_DIR}checkpoints/"

# --- Data ---
DATASET_NAME = "BPI_Challenge_2012.xes.gz"
N_SUBSET_TRACES = 5000
TEST_SPLIT = 0.2 # 20% for testing
VAL_SPLIT = 0.1 # 10% for validation from the training set

# --- Model ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MODEL_NAME = "facebook/opt-350m" # Use a single LLM for all related models
FINETUNE_LLM = True
FROZEN_LLM = False

# --- Model Hyperparameters ---
MODEL_CONFIGS = {
    'lstm': {
        'embedding_dim': 128,
        'hidden_dim': 128,
    },
    'gru': {
        'embedding_dim': 128,
        'hidden_dim': 128,
    },
    'transformer': {
        'embedding_dim': 128,
        'nhead': 4,
        'nlayers': 2,
        'dropout': 0.1,
    },
    'llm_finetune': {
        'llm_model_name': LLM_MODEL_NAME,
    },
    'llm_frozen': {
        'llm_model_name': LLM_MODEL_NAME,
    },
    'hybrid': {
        'gnn_hidden_dim': 128,
        'llm_model_name': LLM_MODEL_NAME,
    }
}

# --- Training ---
BATCH_SIZE = 32
LLM_BATCH_SIZE = 16 # Smaller batch size for LLM-based models
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50 