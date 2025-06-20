import argparse
import os
import torch
import logging
from torch.utils.data import DataLoader
from functools import partial

from .data.data_loader import load_and_preprocess_data
from .data.datasets import (
    TraditionalModelDataset,
    LLMDataset,
    HybridDataset
)
from .models.traditional_models import (
    LSTM_model,
    GRU_model,
    Transformer_model
)
from .models.llm_models import (
    get_llm_model_and_tokenizer
)
from .models.gnn_hybrid_model import (
    TEAGLM_Hybrid_Model,
    GNNConfig
)
from .training.trainer import (
    train_model,
    train_llm,
    train_gnn_hybrid,
    hybrid_collate_fn
)
from .evaluation.evaluator import (
    evaluate_model,
    evaluate_llm,
    evaluate_gnn_hybrid
)


def setup_logging():
    """Set up logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main(args):
    """Main function to run the training and evaluation of models."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    data_path = os.path.join(args.data_dir, args.data_file)
    logger.info(f"Loading data from: {data_path}")
    processed_data = load_and_preprocess_data(data_path)
    train_df = processed_data['train']
    test_df = processed_data['test']

    # --- Model Specific Execution ---

    # Traditional Models (LSTM, GRU, Transformer)
    if args.model_name in ["lstm", "gru", "transformer"]:
        logger.info(f"Initializing {args.model_name.upper()} Model components...")
        dataset = TraditionalModelDataset(train_df)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        model_map = {
            "lstm": LSTM_model,
            "gru": GRU_model,
            "transformer": Transformer_model
        }
        ModelClass = model_map[args.model_name]
        model = ModelClass(input_dim=len(dataset.vocab), hidden_dim=128, output_dim=len(dataset.vocab)).to(device)

        logger.info(f"Starting {args.model_name.upper()} training...")
        best_model_path = train_model(
            model, data_loader, dataset.vocab, device, args.checkpoint_dir, args.model_name, args.data_file
        )
        
        logger.info(f"Evaluating {args.model_name.upper()} model...")
        model.load_state_dict(torch.load(best_model_path))
        eval_dataset = TraditionalModelDataset(test_df)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        predictions_df = evaluate_model(model, eval_loader, dataset.vocab, device)
        
        result_path = os.path.join(args.result_dir, f"{os.path.splitext(args.data_file)[0]}_{args.model_name}_predictions.csv")
        predictions_df.to_csv(result_path, index=False)
        logger.info(f"Results saved to {result_path}")

    # LLM Fine-tuning Model
    elif args.model_name == "llm_finetune":
        logger.info("Initializing LLM Fine-tune Model components...")
        llm_model, llm_tokenizer = get_llm_model_and_tokenizer()
        llm_model.to(device)
        
        dataset = LLMDataset(processed_data['train'], llm_tokenizer)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

        logger.info("Starting LLM fine-tuning...")
        best_model_path = train_llm(llm_model, data_loader, device, args.checkpoint_dir, args.data_file)
        
        logger.info("Evaluating fine-tuned LLM...")
        llm_model.load_state_dict(torch.load(best_model_path))
        eval_dataset = LLMDataset(processed_data['test'], llm_tokenizer)
        predictions_df = evaluate_llm(llm_model, eval_dataset, llm_tokenizer, device)

        result_path = os.path.join(args.result_dir, f"{os.path.splitext(args.data_file)[0]}_llm_finetune_predictions.csv")
        predictions_df.to_csv(result_path, index=False)
        logger.info(f"Results saved to {result_path}")

    # LLM Zero-shot Model
    elif args.model_name == "llm_zero_shot":
        logger.info("Running LLM Zero-shot evaluation...")
        llm_model, llm_tokenizer = get_llm_model_and_tokenizer()
        llm_model.to(device)
        
        eval_dataset = LLMDataset(processed_data['test'], llm_tokenizer)
        predictions_df = get_zero_shot_predictions(llm_model, eval_dataset, llm_tokenizer, device)

        result_path = os.path.join(args.result_dir, f"{os.path.splitext(args.data_file)[0]}_llm_zero_shot_predictions.csv")
        predictions_df.to_csv(result_path, index=False)
        logger.info(f"Results saved to {result_path}")

    # GNN-LLM Hybrid Model
    elif args.model_name == "gnn_llm_hybrid":
        logger.info("Initializing GNN-LLM Hybrid Model components...")
        
        # --- Tokenizer ---
        llm_model_name = "gpt2" # or from args
        _, llm_tokenizer = get_llm_model_and_tokenizer(model_name=llm_model_name)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        # --- Datasets and DataLoaders ---
        logger.info("Creating Hybrid datasets...")
        train_dataset = HybridDataset(train_df, llm_tokenizer, device)
        val_dataset = HybridDataset(test_df, llm_tokenizer, device) # Using test set for validation

        collate_with_pad = partial(hybrid_collate_fn, pad_token_id=llm_tokenizer.pad_token_id)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_pad)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_pad)

        # --- Model Initialization ---
        logger.info("Initializing TEAGLM Hybrid Model...")
        vocab_size = len(train_dataset.vocab)
        gnn_config = GNNConfig(input_dim=vocab_size, output_dim=args.gnn_hidden_dim)
        
        model = TEAGLM_Hybrid_Model(
            gnn_config=gnn_config,
            llm_model_name=llm_model_name,
            num_classes=vocab_size
        ).to(device)

        # --- Training ---
        logger.info("Starting GNN-LLM Hybrid Model training...")
        best_model_path = train_gnn_hybrid(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            device=device,
            checkpoint_dir=args.checkpoint_dir, 
            data_file=args.data_file,
            epochs=args.epochs, 
            patience=args.patience
        )
        
        # --- Evaluation ---
        logger.info("Starting GNN-LLM Hybrid Model evaluation...")
        model.load_state_dict(torch.load(best_model_path))
        
        predictions_df = evaluate_gnn_hybrid(
            model=model, 
            data_loader=val_loader, # Evaluate on the validation/test set
            device=device,
            vocab=val_dataset.vocab
        )

        result_path = os.path.join(args.result_dir, f"{os.path.splitext(args.data_file)[0]}_{args.model_name}_predictions.csv")
        predictions_df.to_csv(result_path, index=False)
        logger.info(f"Results saved to {result_path}")

    else:
        logger.error(f"Unknown model name: {args.model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Next Activity Prediction in Process Mining")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to run")
    parser.add_argument("--data-file", type=str, required=True, help="Name of the data file (e.g., BPI_Challenge_2012.xes.gz)")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the data files")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--result-dir", type=str, required=True, help="Directory to save prediction results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--gnn_hidden_dim", type=int, default=128, help="Hidden dimension for the GNN")
    
    cli_args = parser.parse_args()
    main(cli_args) 