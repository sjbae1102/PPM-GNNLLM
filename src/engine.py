import torch
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, f1_score

from . import config

def train_step(model, data_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        # Move all tensor data to the configured device
        batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch.pop('labels')
        
        optimizer.zero_grad()
        
        outputs = model(**batch)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def val_step(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop('labels')

            outputs = model(**batch)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, model_name):
    """
    Main function to run the training and validation loop.
    """
    best_val_accuracy = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    model_save_path = os.path.join(config.CHECKPOINT_DIR, f"{config.DATASET_NAME.split('.')[0]}_{model_name}_best.pt")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        
        train_loss = train_step(model, train_loader, optimizer, loss_fn)
        val_loss, val_accuracy, val_f1 = val_step(model, val_loader, loss_fn)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best validation accuracy! Saving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
            
    print("\nTraining finished.")
    return model_save_path 