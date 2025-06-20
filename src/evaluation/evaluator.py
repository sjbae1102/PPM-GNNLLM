import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from ..data.datasets import LLMDataset, HybridDataset 

def evaluate_model(model, data_loader, vocab, device):
    """
    Evaluates a traditional model (LSTM, GRU, Transformer) and returns predictions.
    """
    model.eval()
    predictions = []
    actuals = []
    case_ids = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Model"):
            inputs = batch['features'].to(device)
            targets = batch['labels']
            cases = batch['case_id']

            outputs = model(inputs)
            _, predicted_indices = torch.max(outputs, -1)

            for i in range(len(cases)):
                case_id = cases[i]
                # We take the last non-pad token for prediction/actual
                seq_len = (inputs[i].sum(dim=1) != 0).sum().item()
                
                # Get the last actual label before padding
                actual_label_idx = targets[i][seq_len-1].item()
                
                # Get the last predicted label
                predicted_label_idx = predicted_indices[i][seq_len-1].item()

                if actual_label_idx != vocab['<pad>']:
                    actual_activity = vocab.get(actual_label_idx, '<unk>')
                    predicted_activity = vocab.get(predicted_label_idx, '<unk>')

                    predictions.append(predicted_activity)
                    actuals.append(actual_activity)
                    case_ids.append(case_id)

    return pd.DataFrame({
        'Case ID': case_ids,
        'Actual': actuals,
        'Predicted': predictions
    })


def evaluate_llm(model, dataset, tokenizer, device):
    """
    Evaluates a fine-tuned LLM.
    """
    model.eval()
    predictions = []
    actuals = []
    case_ids = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating LLM"):
            sample = dataset[i]
            case_id = sample['case_id']
            input_text = sample['text']
            actual_activity = sample['label']

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
            
            # Generate output; max_length should be chosen carefully
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=len(inputs['input_ids'][0]) + 5  # Allow for a few new tokens
            )
            
            # Decode the prediction, skipping special tokens
            predicted_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            # Extract the last activity from the generated text
            # This logic assumes the model appends the next activity at the end
            predicted_activity = predicted_text.split(" -> ")[-1].strip()

            predictions.append(predicted_activity)
            actuals.append(actual_activity)
            case_ids.append(case_id)
            
    return pd.DataFrame({
        'Case ID': case_ids,
        'Actual': actuals,
        'Predicted': predictions
    })


def evaluate_gnn_hybrid(model, data_loader, device, vocab):
    """
    Evaluates the GNN-LLM Hybrid model.
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating GNN-Hybrid Model"):
            gnn_data = batch['gnn_data'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # Labels are already indexed

            # Forward pass
            logits = model(gnn_data, input_ids, attention_mask)
            _, predicted_indices = torch.max(logits, 1)
            
            # Convert indices to activity names
            predicted_activities = [list(vocab.keys())[list(vocab.values()).index(i)] for i in predicted_indices.cpu().numpy()]
            actual_activities = [list(vocab.keys())[list(vocab.values()).index(i)] for i in labels.cpu().numpy()]

            predictions.extend(predicted_activities)
            actuals.extend(actual_activities)

    return pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions
    })


def evaluate_model_on_validation_set(model, val_loader, criterion, device, vocab):
    """Evaluates the model on a validation set for traditional models."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['features'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(inputs)
            # Flatten outputs and targets for loss calculation
            outputs_flat = outputs.view(-1, len(vocab))
            targets_flat = targets.view(-1)

            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, -1)
            
            # Create a mask to ignore padding tokens in accuracy calculation
            mask = targets_flat != vocab['<pad>']
            correct_predictions += ((predicted.view(-1) == targets_flat) & mask).sum().item()
            total_predictions += mask.sum().item()
            
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return avg_loss, accuracy 