import os
import sys
import pickle
from sklearn.model_selection import train_test_split
import argparse

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import ProcessDataProcessor, LogDataProcessor
from src import config

def main(dataset_name):
    """Preprocesses the specified event log dataset."""
    # Override the dataset name in the config
    config.DATASET_NAME = dataset_name
    
    print("--- Starting Data Preprocessing ---")
    
    # Define output directory
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Initialize the processor
    processor = ProcessDataProcessor(
        dataset_name=config.DATASET_NAME,
        llm_model_name=config.LLM_MODEL_NAME,
        data_dir=config.DATA_DIR
    )
    
    # Get all sequences and metadata
    print("Creating sequences from log...")
    all_sequences, metadata = processor.get_sequences_and_metadata(num_samples=config.N_SUBSET_TRACES)
    
    if not all_sequences:
        print("No sequences were generated. Exiting.")
        return

    # Split data
    print("Splitting data into train, validation, and test sets...")
    train_val_sequences, test_sequences = train_test_split(all_sequences, test_size=config.TEST_SPLIT, random_state=42)
    train_sequences, val_sequences = train_test_split(train_val_sequences, test_size=config.VAL_SPLIT / (1 - config.TEST_SPLIT), random_state=42)
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    
    # Save data
    data_to_save = {
        'train': train_sequences,
        'val': val_sequences,
        'test': test_sequences,
        'metadata': metadata
    }
    
    output_path = os.path.join(processed_dir, f"{config.DATASET_NAME.split('.')[0]}_processed.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)
        
    print(f"\n✅ Successfully preprocessed and saved data to {output_path}")
    print("--- Data Preprocessing Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess event log data.")
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='BPI_Challenge_2012.xes.gz', 
        help='Name of the dataset file in the data directory (e.g., BPI_Challenge_2017.xes.gz)'
    )
    args = parser.parse_args()
    main(args.dataset) 