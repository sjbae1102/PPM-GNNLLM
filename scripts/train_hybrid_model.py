import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import pickle
from torch.utils.data import DataLoader

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import HybridModelDataset, hybrid_collate_fn
from src.models.hybrid_model import HybridGNNLLMModel
from src.engine import train_and_evaluate
from src import config

def get_trainable_parameters(model):
    """Gets all parameters that have requires_grad set to True."""
    return filter(lambda p: p.requires_grad, model.parameters())

def main():
    parser = argparse.ArgumentParser(description="Train the Hybrid GNN-LLM model.")
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='BPI_Challenge_2012.xes.gz', 
        help='Name of the dataset file (e.g., BPI_Challenge_2017.xes.gz)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda:0', 
        help='Device to train on (e.g., "cuda:0" or "cuda:1")'
    )
    args = parser.parse_args()

    # --- Override config values ---
    config.DATASET_NAME = args.dataset
    config.DEVICE = args.device

    print("--- Starting training for Hybrid GNN-LLM model ---")
    
    # 1. Load preprocessed data
    data_path = os.path.join(config.DATA_DIR, "processed", f"{config.DATASET_NAME.split('.')[0]}_processed.pkl")
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_sequences = data['train']
    val_sequences = data['val']
    metadata = data['metadata']
    
    # 2. Create Datasets and DataLoaders
    train_dataset = HybridModelDataset(train_sequences, metadata)
    val_dataset = HybridModelDataset(val_sequences, metadata)

    train_loader = DataLoader(train_dataset, batch_size=config.LLM_BATCH_SIZE, shuffle=True, collate_fn=hybrid_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.LLM_BATCH_SIZE, shuffle=False, collate_fn=hybrid_collate_fn)

    # 3. Initialize model
    print("Initializing model...")
    model_config = config.MODEL_CONFIGS['hybrid']
    model = HybridGNNLLMModel(metadata=metadata, **model_config)
    model.to(config.DEVICE)

    # 4. Define Loss and Optimizer (using only trainable parameters for PEFT)
    print("Setting up optimizer for PEFT...")
    trainable_params = get_trainable_parameters(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=config.LEARNING_RATE
    )

    # 5. Train model
    print(f"Training on {config.DEVICE}...")

    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        model_name='hybrid'
    )
    
    print(f"--- Finished training for Hybrid GNN-LLM model ---")

if __name__ == '__main__':
    main() 