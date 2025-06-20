#!/usr/bin/env python3
"""
PPM Next Activity Prediction 모델 비교 실험의 메인 실행 스크립트
"""
import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import argparse
from transformers import AutoConfig
import gc

# 내부 모듈 임포트
from src.data_loader import (
    ProcessDataProcessor, TraditionalModelDataset, 
    HybridDataset, hybrid_collate_fn,
    LLMDataset, llm_collate_fn
)
from src.models.lstm_model import LSTMModel
from src.models.traditional_models import GRUModel, TransformerModel
from src.models.gnn_hybrid_model import TEAGLM_Hybrid_Model
from src.models.llm_models import LLMForClassification
from src.trainer import train_traditional, train_gnn_hybrid, train_llm
from src.evaluator import evaluate_model, plot_performance_comparison

def set_seed(seed: int):
    """재현성을 위해 시드를 고정합니다."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="PPM Next Activity Prediction")
    
    # 실행 모드 및 경로 설정
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'], help='Run mode: train or eval.')
    parser.add_argument('--result-dir', type=str, default='results', help='Directory to save evaluation results.')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory for model checkpoints.')
    
    # 실험 설정
    parser.add_argument('--datasets', nargs='+', default=['BPI_2012', 'BPI_2017'], help='Datasets to run on.')
    parser.add_argument('--models', nargs='+', default=['lstm', 'gnn_llm_hybrid'], help='Models to run.')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of process instances to use.')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing the XES data files.')
    
    # 모델 파라미터
    parser.add_argument('--max-length', type=int, default=1024, help='Max sequence length for padding/truncation.')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs for Traditional models.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for Traditional models.')
    parser.add_argument('--llm-model', type=str, default='facebook/opt-350m', help='Base LLM for hybrid model.')
    parser.add_argument('--llm-epochs', type=int, default=100, help='Epochs for GNN-LLM hybrid model.')
    parser.add_argument('--llm-batch-size', type=int, default=16, help='Batch size for GNN-LLM hybrid model.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    
    args = parser.parse_args()
    
    # Ensure checkpoint and result directories exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        run_training(args, device)
    elif args.mode == 'eval':
        run_eval(args, device)

def run_training(args, device):
    """모델 훈련 파이프라인"""
    print("\nStarting Training Mode...")
    for dataset_name in args.datasets:
        print(f"\n{'='*25} PROCESSING DATASET: {dataset_name.upper()} {'='*25}")
        
        processor = ProcessDataProcessor(dataset_name, llm_model_name=args.llm_model, data_dir=args.data_dir)
        sequences, metadata = processor.get_sequences_and_metadata(args.num_samples)
        
        if not sequences or not metadata:
            print(f"Skipping {dataset_name} due to lack of data.")
            continue
            
        train_val_seq, _ = train_test_split(sequences, test_size=0.2, random_state=42)
        train_seq, val_seq = train_test_split(train_val_seq, test_size=0.25, random_state=42)
        print(f"Data split for training: Train={len(train_seq)}, Val={len(val_seq)}")

        for model_name in args.models:
            print(f"\n--- Training Model: {model_name.upper()} on {dataset_name.upper()} ---")
            
            if model_name.lower() in ['lstm', 'gru', 'transformer']:
                train_dataset = TraditionalModelDataset(train_seq, metadata['activity_encoder'], args.max_length)
                val_dataset = TraditionalModelDataset(val_seq, metadata['activity_encoder'], args.max_length)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
                
                model_class = None
                if model_name.lower() == 'lstm':
                    model_class = LSTMModel
                elif model_name.lower() == 'gru':
                    model_class = GRUModel
                elif model_name.lower() == 'transformer':
                    model_class = TransformerModel

                train_traditional(model_class, train_loader, val_loader, metadata['num_unique_activities'], device, args, dataset_name, model_name)

            elif model_name.lower() == 'gnn_llm_hybrid':
                tokenizer = processor.tokenizer
                train_dataset = HybridDataset(train_seq, metadata, tokenizer, args.max_length)
                val_dataset = HybridDataset(val_seq, metadata, tokenizer, args.max_length)
                train_loader = DataLoader(train_dataset, batch_size=args.llm_batch_size, shuffle=True, collate_fn=lambda b: hybrid_collate_fn(b, tokenizer.pad_token_id))
                val_loader = DataLoader(val_dataset, batch_size=args.llm_batch_size, collate_fn=lambda b: hybrid_collate_fn(b, tokenizer.pad_token_id), shuffle=False)
                llm_config = AutoConfig.from_pretrained(args.llm_model)
                train_gnn_hybrid(TEAGLM_Hybrid_Model, llm_config, train_loader, val_loader, metadata, device, args, dataset_name)
            elif model_name.lower() == 'llm_finetune':
                tokenizer = processor.tokenizer
                train_dataset = LLMDataset(train_seq, metadata, tokenizer, args.max_length)
                val_dataset = LLMDataset(val_seq, metadata, tokenizer, args.max_length)
                train_loader = DataLoader(train_dataset, batch_size=args.llm_batch_size, shuffle=True, collate_fn=lambda b: llm_collate_fn(b, tokenizer.pad_token_id))
                val_loader = DataLoader(val_dataset, batch_size=args.llm_batch_size, collate_fn=lambda b: llm_collate_fn(b, tokenizer.pad_token_id))
                
                train_llm(LLMForClassification, train_loader, val_loader, metadata['num_unique_activities'], device, args, dataset_name)
            
            elif model_name.lower() == 'llm_zero_shot':
                print("Zero-shot model does not require training. Skipping.")
                continue

            else:
                print(f"Model '{model_name}' not recognized for training. Skipping.")
        
        # 루프 마지막에 메모리 정리
        print(f"Finished processing {dataset_name}. Cleaning up memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_eval(args, device):
    """최적 모델 평가 파이프라인"""
    print("\nStarting Evaluation Mode...")
    final_results = {}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*25} EVALUATING ON DATASET: {dataset_name.upper()} {'='*25}")

        processor = ProcessDataProcessor(dataset_name, llm_model_name=args.llm_model, data_dir=args.data_dir)
        sequences, metadata = processor.get_sequences_and_metadata(args.num_samples)
        
        if not sequences or not metadata:
            print(f"Skipping {dataset_name} due to lack of data.")
            continue

        _, test_seq = train_test_split(sequences, test_size=0.2, random_state=42)
        print(f"Test set size: {len(test_seq)}")

        for model_name in args.models:
            print(f"\n--- Evaluating Model: {model_name.upper()} on {dataset_name.upper()} ---")
            
            model_instance = None
            best_model_path = os.path.join(args.checkpoint_dir, f'{dataset_name}_{model_name.lower()}_best.pt')
            
            if model_name.lower() in ['lstm', 'gru', 'transformer']:
                if not os.path.exists(best_model_path):
                    print(f"Checkpoint not found for {model_name.upper()} on {dataset_name}. Skipping.")
                    continue
                
                test_dataset = TraditionalModelDataset(test_seq, metadata['activity_encoder'], args.max_length)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
                
                if model_name.lower() == 'lstm':
                    model_instance = LSTMModel(num_activities=metadata['num_unique_activities'], embedding_dim=128, hidden_dim=128).to(device)
                elif model_name.lower() == 'gru':
                    model_instance = GRUModel(num_activities=metadata['num_unique_activities'], embedding_dim=128, hidden_dim=128).to(device)
                elif model_name.lower() == 'transformer':
                    model_instance = TransformerModel(num_activities=metadata['num_unique_activities'], embedding_dim=128, nhead=8, nhid=256, nlayers=2).to(device)

                model_instance.load_state_dict(torch.load(best_model_path))
                metrics, preds_df = evaluate_model(model_instance, test_loader, device, 'lstm') # 'lstm' type can be used for all traditional models

            elif model_name.lower() == 'gnn_llm_hybrid':
                best_model_path = os.path.join(args.checkpoint_dir, f'{dataset_name}_gnn_llm_hybrid_best.pt')
                if not os.path.exists(best_model_path):
                    print(f"Checkpoint not found for GNN-Hybrid on {dataset_name}. Skipping.")
                    continue
                tokenizer = processor.tokenizer
                test_dataset = HybridDataset(test_seq, metadata, tokenizer, args.max_length)
                test_loader = DataLoader(test_dataset, batch_size=args.llm_batch_size, collate_fn=lambda b: hybrid_collate_fn(b, tokenizer.pad_token_id), shuffle=False)
                llm_config = AutoConfig.from_pretrained(args.llm_model)
                model_instance = TEAGLM_Hybrid_Model(llm_name=args.llm_model, llm_config=llm_config, num_nodes=metadata['num_unique_activities'], num_classes=metadata['num_unique_activities']).to(device)
                model_instance.load_state_dict(torch.load(best_model_path))

                metrics, preds_df = evaluate_model(model_instance, test_loader, device, 'gnn_hybrid')
            
            elif model_name.lower() in ['llm_finetune', 'llm_zero_shot']:
                if model_name.lower() == 'llm_finetune' and not os.path.exists(best_model_path):
                    print(f"Checkpoint not found for {model_name.upper()} on {dataset_name}. Skipping.")
                    continue

                tokenizer = processor.tokenizer
                test_dataset = LLMDataset(test_seq, metadata, tokenizer, args.max_length)
                test_loader = DataLoader(test_dataset, batch_size=args.llm_batch_size, collate_fn=lambda b: llm_collate_fn(b, tokenizer.pad_token_id))
                
                model_instance = LLMForClassification(llm_name=args.llm_model, num_classes=metadata['num_unique_activities']).to(device)
                
                if model_name.lower() == 'llm_finetune':
                    model_instance.load_state_dict(torch.load(best_model_path))
                
                # For zero-shot, we use the model as is
                metrics, preds_df = evaluate_model(model_instance, test_loader, device, 'llm')

            else:
                print(f"Model '{model_name}' not recognized. Skipping.")
                continue

            if not preds_df.empty:
                preds_csv_path = os.path.join(args.result_dir, f"{dataset_name}_{model_name}_predictions.csv")
                preds_df.to_csv(preds_csv_path, index=False)
                print(f"📊 Predictions saved to {preds_csv_path}")

            final_results.setdefault(model_name, {})[dataset_name] = metrics
            print(f"Final Test Metrics for {model_name.upper()}: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1_score']:.4f}")
    
    if final_results:
        plot_performance_comparison(final_results, args.result_dir)
        
if __name__ == '__main__':
    main()