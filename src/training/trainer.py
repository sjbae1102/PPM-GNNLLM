import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DataParallel as TorchDataParallel
from torch_geometric.nn import DataParallel as GeometricDataParallel
from tqdm import tqdm
import os
import logging
from ..evaluation.evaluator import evaluate_model_on_validation_set
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)

def train_model(model, data_loader, vocab, device, checkpoint_dir, model_name, data_file, epochs=10, patience=3):
    """
    Trains a traditional model (LSTM, GRU, Transformer) with early stopping.
    """
    logger.info(f"Starting training for {model_name} on {data_file}...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Create a validation set from the training data (e.g., last 10%)
    # Note: This is a simplistic split. A more robust approach would be a separate validation file or pre-split indices.
    train_size = int(len(data_loader.dataset) * 0.9)
    val_size = len(data_loader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data_loader.dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_loader.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=data_loader.batch_size, shuffle=False)

    model_filename = f"{os.path.splitext(data_file)[0]}_{model_name}_best.pt"
    model_path = os.path.join(checkpoint_dir, model_filename)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = batch['features'].to(device)
            targets = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Reshape for CrossEntropyLoss: (N, C) and (N)
            loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation step
        val_loss, val_accuracy = evaluate_model_on_validation_set(model, val_loader, criterion, device, vocab)
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            logger.info(f"Validation loss improved. Saved model to {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
                
    logger.info("Training finished.")
    return model_path


def train_llm(model, data_loader, device, checkpoint_dir, data_file, epochs=3, patience=1):
    """
    Fine-tunes an LLM with early stopping.
    """
    # (Implementation for LLM training...)
    logger.info("LLM training starts...")
    # For now, just save the initial model as a placeholder for demonstration
    model_filename = f"{os.path.splitext(data_file)[0]}_llm_finetune_best.pt"
    model_path = os.path.join(checkpoint_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    logger.info(f"LLM fine-tuning placeholder finished. Model saved to {model_path}")
    return model_path


def hybrid_collate_fn(batch_list, pad_token_id):
    """
    Custom collate function for the HybridDataset.
    Batches GNN data and pads LLM data.
    """
    gnn_data_list = [item['gnn_data'] for item in batch_list]
    input_ids_list = [item['input_ids'] for item in batch_list]
    attention_mask_list = [item['attention_mask'] for item in batch_list]
    labels_list = [item['labels'] for item in batch_list]

    # Batch GNN data
    gnn_batch = Batch.from_data_list(gnn_data_list)
    
    # Pad LLM data
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    
    # Stack labels
    labels = torch.stack(labels_list)
    
    return {
        'gnn_data': gnn_batch,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels
    }

def train_gnn_hybrid(model, train_loader, val_loader, device, checkpoint_dir, data_file, epochs=10, patience=3):
    """
    Trains the GNN-LLM hybrid model using the PEFT strategy.
    """
    logger.info(f"Starting PEFT training for GNN-LLM Hybrid on {data_file}...")

    # --- PEFT: Freeze most of LLM, unfreeze specific parts ---
    for name, param in model.llm.named_parameters():
        param.requires_grad = False
        
    # Unfreeze input embeddings
    for param in model.llm.get_input_embeddings().parameters():
        param.requires_grad = True

    # Ensure GNN, projection, and classification head are trainable
    for param in model.gnn.parameters():
        param.requires_grad = True
    for param in model.projection.parameters():
        param.requires_grad = True
    for param in model.classification_head.parameters():
        param.requires_grad = True
        
    # Collect parameters that require gradients for the optimizer
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(optimizer_params, lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_path = os.path.join(checkpoint_dir, f"{os.path.splitext(data_file)[0]}_gnn_llm_hybrid_best.pt")

    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            gnn_data = batch['gnn_data'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(gnn_data, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                gnn_data = batch['gnn_data'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(gnn_data, input_ids, attention_mask)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            logger.info(f"Validation loss improved. Saved model to {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs.")
                break
                
    logger.info("Training finished.")
    return model_path 