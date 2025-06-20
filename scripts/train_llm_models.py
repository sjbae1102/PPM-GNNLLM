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

from src.data_processing import LLMClassifierDataset, llm_collate_fn
from src.models.llm_models import LLMClassifier
from src.engine import train_and_evaluate
from src import config

def main():
    parser = argparse.ArgumentParser(description="Train an LLM-based model for next activity prediction.")
    parser.add_argument('--model', type=str, required=True, choices=['llm_finetune', 'llm_frozen'],
                        help='The type of LLM model to train.')
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
    train_dataset = LLMClassifierDataset(train_sequences, metadata)
    val_dataset = LLMClassifierDataset(val_sequences, metadata)

    train_loader = DataLoader(train_dataset, batch_size=config.LLM_BATCH_SIZE, shuffle=True, collate_fn=llm_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.LLM_BATCH_SIZE, shuffle=False, collate_fn=llm_collate_fn)

    # 3. Initialize model
    print("Initializing model...")
    model_config = config.MODEL_CONFIGS[args.model]
    finetune_llm = True if args.model == 'llm_finetune' else False
    
    model = LLMClassifier(
        vocab_size=vocab_size,
        llm_model_name=model_config['llm_model_name'],
        finetune_llm=finetune_llm
    )
    model.to(config.DEVICE)

    # 4. Define Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)

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