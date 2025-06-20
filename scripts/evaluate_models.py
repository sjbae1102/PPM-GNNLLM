import torch
import pandas as pd
from tqdm import tqdm
import os
import sys
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import (
    TraditionalModelDataset, base_collate_fn,
    LLMClassifierDataset, llm_collate_fn,
    HybridModelDataset, hybrid_collate_fn
)
from src.models.base_models import LSTMModel, GRUModel, TransformerModel
from src.models.hybrid_model import HybridGNNLLMModel
from src.models.llm_models import LLMClassifier
from src import config

def evaluate(model, data_loader):
    """Evaluates a model on a given data loader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating"):
            batch = {k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop('labels')
            outputs = model(**batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

def main():
    print("--- Starting Model Evaluation ---")
    
    data_path = os.path.join(config.DATA_DIR, "processed", f"{config.DATASET_NAME.split('.')[0]}_processed.pkl")
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    test_sequences = data['test']
    metadata = data['metadata']
    vocab_size = metadata['num_unique_activities']

    # Define all models to evaluate
    models_to_init = {
        'lstm': lambda: LSTMModel(vocab_size=vocab_size, **config.MODEL_CONFIGS['lstm']),
        'gru': lambda: GRUModel(vocab_size=vocab_size, **config.MODEL_CONFIGS['gru']),
        'transformer': lambda: TransformerModel(vocab_size=vocab_size, **config.MODEL_CONFIGS['transformer']),
        'llm_frozen': lambda: LLMClassifier(vocab_size=vocab_size, finetune_llm=False, **config.MODEL_CONFIGS['llm_frozen']),
        'llm_finetune': lambda: LLMClassifier(vocab_size=vocab_size, finetune_llm=True, **config.MODEL_CONFIGS['llm_finetune']),
        'hybrid': lambda: HybridGNNLLMModel(metadata=metadata, **config.MODEL_CONFIGS['hybrid'])
    }
    
    # Define datasets and collate functions for each model type
    dataset_map = {
        'base': (TraditionalModelDataset, base_collate_fn),
        'llm': (LLMClassifierDataset, llm_collate_fn),
        'hybrid': (HybridModelDataset, hybrid_collate_fn)
    }
    model_type_map = {
        'lstm': 'base', 'gru': 'base', 'transformer': 'base',
        'llm_frozen': 'llm', 'llm_finetune': 'llm',
        'hybrid': 'hybrid'
    }

    evaluation_results = {}
    pred_dir = os.path.join(config.RESULTS_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for name, init_fn in models_to_init.items():
        print(f"\n--- Evaluating {name.upper()} ---")
        model_path = os.path.join(config.CHECKPOINT_DIR, f"{config.DATASET_NAME.split('.')[0]}_{name}_best.pt")
        
        if not os.path.exists(model_path):
            print(f"Checkpoint for {name} not found. Skipping.")
            continue
            
        model = init_fn()
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.to(config.DEVICE)
        
        # Create appropriate dataset and dataloader
        m_type = model_type_map[name]
        DatasetClass, collate_fn = dataset_map[m_type]
        test_dataset = DatasetClass(test_sequences, metadata)
        # Use largest batch size for evaluation for speed
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        labels, preds = evaluate(model, test_loader)
        
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        evaluation_results[name] = {'accuracy': accuracy, 'f1_score': f1}
        print(f"Test Accuracy: {accuracy:.4f} | Test F1-score: {f1:.4f}")
        
        df = pd.DataFrame({'true_label': labels, 'predicted_label': preds})
        df.to_csv(os.path.join(pred_dir, f"{config.DATASET_NAME.split('.')[0]}_{name}_predictions.csv"), index=False)

    print("\n--- Evaluation Summary ---")
    results_df = pd.DataFrame.from_dict(evaluation_results, orient='index')
    print(results_df)
    results_df.to_csv(os.path.join(config.RESULTS_DIR, "evaluation_summary.csv"))
    print(f"\n✅ Evaluation finished. Summary saved to {config.RESULTS_DIR}evaluation_summary.csv")

if __name__ == '__main__':
    main() 