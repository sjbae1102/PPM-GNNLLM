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

from src.data_processing import TraditionalModelDataset, base_collate_fn
from src.models.base_models import LSTMModel, GRUModel, TransformerModel
from src.engine import train_and_evaluate
from src import config

def main():
    parser = argparse.ArgumentParser(description="Train a baseline model for next activity prediction.")
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'gru', 'transformer'],
                        help='The type of baseline model to train.')
    args = parser.parse_args()

    print(f"--- Starting training for {args.model.upper()} model ---")
    
    # 1. Load preprocessed data
    data_path = os.path.join(config.DATA_DIR, "processed", f"{config.DATASET_NAME.split('.')[0]}_processed.pkl")
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_sequences = data['train']
    val_sequences = data['val']
    metadata = data['metadata']
    vocab_size = metadata['num_unique_activities']
    
    # 2. Create Datasets and DataLoaders
    train_dataset = TraditionalModelDataset(train_sequences, metadata)
    val_dataset = TraditionalModelDataset(val_sequences, metadata)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=base_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=base_collate_fn)

    # 3. Initialize model
    print("Initializing model...")
    model_config = config.MODEL_CONFIGS[args.model]
    if args.model == 'lstm':
        model = LSTMModel(vocab_size=vocab_size, **model_config)
    elif args.model == 'gru':
        model = GRUModel(vocab_size=vocab_size, **model_config)
    else: # transformer
        model = TransformerModel(vocab_size=vocab_size, **model_config)
    model.to(config.DEVICE)

    # 4. Define Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=0) # Padding index
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 5. Train model
    print(f"Training on {config.DEVICE}...")
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        model_name=args.model
    )
    print(f"--- Finished training for {args.model.upper()} model ---")

if __name__ == '__main__':
    main() 