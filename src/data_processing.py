import pandas as pd
import pm4py
import numpy as np
from typing import List, Tuple, Dict
import random
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import os
import pickle
from . import config

# Provided by user - this will be the core of data processing
class ProcessDataProcessor:
    """Process Mining 데이터를 Next Activity Prediction용으로 변환하는 클래스"""
    
    def __init__(self, dataset_name: str, llm_model_name: str, data_dir: str):
        self.data_dir = data_dir
        self.log = self._load_log(dataset_name)
        raw_df = pm4py.convert_to_dataframe(self.log)
        self.df = self.preprocess_dataframe(raw_df)
        
        self.activity_encoder = LabelEncoder()
        self.resource_encoder = LabelEncoder()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model_max_length = self.tokenizer.model_max_length
            if not self.model_max_length or self.model_max_length > 2048:
                self.model_max_length = 1024
        except Exception as e:
            print(f"Warning: Could not load tokenizer '{llm_model_name}'. Using 'gpt2' as default. Error: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.model_max_length = 1024

    def _load_log(self, dataset_name: str):
        file_path = os.path.join(self.data_dir, dataset_name)
        print(f"Loading log from {file_path}...")
        return pm4py.read_xes(file_path)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Original columns: {df.columns.tolist()}")
        column_mapping = {
            'case:concept:name': 'case_id', 'concept:name': 'activity',
            'time:timestamp': 'timestamp', 'org:resource': 'resource'
        }
        df = df.rename(columns=column_mapping)
        if 'resource' not in df.columns:
            df['resource'] = 'Unknown'
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(['case_id', 'timestamp'])
        print(f"Final columns: {df.columns.tolist()}")
        return df

    def create_sequences(self, all_activities_list: List[str]) -> List[Dict]:
        sequences = []
        for case_id, case_df in self.df.groupby('case_id'):
            case_df = case_df.sort_values('timestamp')
            activities = case_df['activity'].tolist()
            resources = case_df['resource'].tolist()
            timestamps = case_df['timestamp'].tolist()
            
            if len(activities) < 3: continue
            
            for i in range(1, len(activities) - 1):
                if i + 1 > 50: continue # Limit prefix length

                input_activities = activities[:i+1]
                next_activity = activities[i+1]
                
                llm_input = self._create_enhanced_llm_prompt(input_activities, timestamps[:i+1])
                
                sequences.append({
                    'case_id': case_id,
                    'input_activities': input_activities,
                    'next_activity_text': next_activity,
                    'llm_input': llm_input
                })
        return sequences

    def filter_and_sample_sequences(self, sequences: List[Dict], num_samples: int) -> List[Dict]:
        if not sequences or num_samples <= 0 or num_samples >= len(sequences):
            return sequences
        random.seed(42)
        random.shuffle(sequences)
        return sequences[:num_samples]

    def get_sequences_and_metadata(self, num_samples: int) -> Tuple[List[Dict], Dict]:
        all_activities_list = self.df['activity'].unique().tolist()
        all_sequences = self.create_sequences(all_activities_list)
        sampled_sequences = self.filter_and_sample_sequences(all_sequences, num_samples)
        
        if not sampled_sequences: return [], {}
        
        metadata = self.get_metadata(sampled_sequences)
        return sampled_sequences, metadata

    def get_metadata(self, sequences: List[Dict]) -> Dict:
        all_activities = sorted(list(self.df['activity'].unique()))
        self.activity_encoder.fit(all_activities)
        
        return {
            'num_unique_activities': len(self.activity_encoder.classes_),
            'activity_encoder': self.activity_encoder,
            'tokenizer': self.tokenizer,
            'model_max_length': self.model_max_length
        }

    def _format_timedelta(self, td: pd.Timedelta) -> str:
        seconds = td.total_seconds()
        if seconds < 60: return f"{seconds:.0f}s"
        if seconds < 3600: return f"{seconds/60:.1f}m"
        if seconds < 86400: return f"{seconds/3600:.1f}h"
        return f"{seconds/86400:.1f}d" 

    def _create_enhanced_llm_prompt(self, activities: List[str], timestamps: List[pd.Timestamp]) -> str:
        prompt = "Case History:\n"
        for i, (activity, ts) in enumerate(zip(activities, timestamps)):
            duration_str = ""
            if i > 0:
                duration = ts - timestamps[i-1]
                duration_str = f" (Duration: {self._format_timedelta(duration)})"
            prompt += f"- {activity} at {ts.strftime('%Y-%m-%d %H:%M')}{duration_str}\n"
        prompt += "\nPredict the next activity:"
        return prompt

# --- Dataset Classes ---
class TraditionalModelDataset(Dataset):
    def __init__(self, sequences: List[Dict], metadata: Dict):
        self.sequences = sequences
        self.activity_encoder = metadata['activity_encoder']
        self.max_len = 50 # Max prefix length

    def __len__(self): return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        activities = seq['input_activities']
        activity_indices = self.activity_encoder.transform(activities)
        target_idx = self.activity_encoder.transform([seq['next_activity_text']])[0]
        
        padded_sequence = np.zeros(self.max_len, dtype=np.int64)
        padded_sequence[-len(activity_indices):] = activity_indices
        
        return {
            'prefix_indices': torch.tensor(padded_sequence, dtype=torch.long),
            'labels': torch.tensor(target_idx, dtype=torch.long)
        }

class HybridModelDataset(Dataset):
    def __init__(self, sequences: List[Dict], metadata: Dict):
        self.sequences = sequences
        self.activity_encoder = metadata['activity_encoder']
        self.tokenizer = metadata['tokenizer']
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = metadata['model_max_length']

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Text
        tokenized = self.tokenizer(seq['llm_input'], padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].squeeze()
        attention_mask = tokenized['attention_mask'].squeeze()
        
        # Label
        label = torch.tensor(self.activity_encoder.transform([seq['next_activity_text']])[0], dtype=torch.long)
        
        # Graph
        nodes = self.activity_encoder.transform(seq['input_activities'])
        # Add self-loops to every node
        edges = [[i, i] for i in range(len(nodes))]
        # Add sequential edges
        edges.extend([[i, i + 1] for i in range(len(nodes) - 1)])
        graph_data = Data(x=torch.tensor(nodes, dtype=torch.long), edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label, 'graph_data': graph_data}

class LLMClassifierDataset(Dataset):
    def __init__(self, sequences: List[Dict], metadata: Dict):
        self.sequences = sequences
        self.activity_encoder = metadata['activity_encoder']
        self.tokenizer = metadata['tokenizer']
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = metadata['model_max_length']

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokenized = self.tokenizer(seq['llm_input'], padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")
        label = self.activity_encoder.transform([seq['next_activity_text']])[0]
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def llm_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def hybrid_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    graph_batch = Batch.from_data_list([item['graph_data'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'graph_data': graph_batch}

def base_collate_fn(batch):
    prefix_indices = torch.stack([item['prefix_indices'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'prefix_indices': prefix_indices, 'labels': labels}

def save_processed_data(data: Tuple[List[Dict], Dict], output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved processed data to {output_path}")

def load_processed_data(input_path: str) -> Tuple[List[Dict], Dict]:
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data['sequences'], data['metadata']

class LogDataProcessor:
    def __init__(self):
        self.dataset_name = config.DATASET_NAME
        self.raw_data_path = os.path.join(config.DATA_DIR, self.dataset_name)
        self.processed_data_path = os.path.join(config.PROCESSED_DATA_DIR, self.dataset_name.replace('.xes.gz', '_processed.pkl'))
        self.log = None
        self.processed_df = None
        self.activity_encoder = LabelEncoder()
        self.resource_encoder = LabelEncoder()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
            self.model_max_length = self.tokenizer.model_max_length
            if not self.model_max_length or self.model_max_length > 2048:
                self.model_max_length = 1024
        except Exception as e:
            print(f"Warning: Could not load tokenizer '{config.LLM_MODEL_NAME}'. Using 'gpt2' as default. Error: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.model_max_length = 1024

    def _load_log(self):
        print(f"Loading log from {self.raw_data_path}...")
        self.log = pm4py.read_xes(self.raw_data_path)

    def preprocess_dataframe(self):
        print(f"Original columns: {self.log.columns.tolist()}")
        column_mapping = {
            'case:concept:name': 'case_id', 'concept:name': 'activity',
            'time:timestamp': 'timestamp', 'org:resource': 'resource'
        }
        self.processed_df = self.log.rename(columns=column_mapping)
        if 'resource' not in self.processed_df.columns:
            self.processed_df['resource'] = 'Unknown'
        self.processed_df['timestamp'] = pd.to_datetime(self.processed_df['timestamp'], errors='coerce')
        self.processed_df = self.processed_df.sort_values(['case_id', 'timestamp'])
        print(f"Final columns: {self.processed_df.columns.tolist()}")
        return self.processed_df

    def create_sequences(self, all_activities_list: List[str]) -> List[Dict]:
        sequences = []
        for case_id, case_df in self.processed_df.groupby('case_id'):
            case_df = case_df.sort_values('timestamp')
            activities = case_df['activity'].tolist()
            resources = case_df['resource'].tolist()
            timestamps = case_df['timestamp'].tolist()
            
            if len(activities) < 3: continue
            
            for i in range(1, len(activities) - 1):
                if i + 1 > 50: continue # Limit prefix length

                input_activities = activities[:i+1]
                next_activity = activities[i+1]
                
                llm_input = self._create_enhanced_llm_prompt(input_activities, timestamps[:i+1])
                
                sequences.append({
                    'case_id': case_id,
                    'input_activities': input_activities,
                    'next_activity_text': next_activity,
                    'llm_input': llm_input
                })
        return sequences

    def filter_and_sample_sequences(self, sequences: List[Dict], num_samples: int) -> List[Dict]:
        if not sequences or num_samples <= 0 or num_samples >= len(sequences):
            return sequences
        random.seed(42)
        random.shuffle(sequences)
        return sequences[:num_samples]

    def get_sequences_and_metadata(self, num_samples: int) -> Tuple[List[Dict], Dict]:
        all_activities_list = self.processed_df['activity'].unique().tolist()
        all_sequences = self.create_sequences(all_activities_list)
        sampled_sequences = self.filter_and_sample_sequences(all_sequences, num_samples)
        
        if not sampled_sequences: return [], {}
        
        metadata = self.get_metadata(sampled_sequences)
        return sampled_sequences, metadata

    def get_metadata(self, sequences: List[Dict]) -> Dict:
        all_activities = sorted(list(self.processed_df['activity'].unique()))
        self.activity_encoder.fit(all_activities)
        
        return {
            'num_unique_activities': len(self.activity_encoder.classes_),
            'activity_encoder': self.activity_encoder,
            'tokenizer': self.tokenizer,
            'model_max_length': self.model_max_length
        }

    def _format_timedelta(self, td: pd.Timedelta) -> str:
        seconds = td.total_seconds()
        if seconds < 60: return f"{seconds:.0f}s"
        if seconds < 3600: return f"{seconds/60:.1f}m"
        if seconds < 86400: return f"{seconds/3600:.1f}h"
        return f"{seconds/86400:.1f}d" 

    def _create_enhanced_llm_prompt(self, activities: List[str], timestamps: List[pd.Timestamp]) -> str:
        prompt = "Case History:\n"
        for i, (activity, ts) in enumerate(zip(activities, timestamps)):
            duration_str = ""
            if i > 0:
                duration = ts - timestamps[i-1]
                duration_str = f" (Duration: {self._format_timedelta(duration)})"
            prompt += f"- {activity} at {ts.strftime('%Y-%m-%d %H:%M')}{duration_str}\n"
        prompt += "\nPredict the next activity:"
        return prompt

    def process_data(self):
        self._load_log()
        self.processed_df = self.preprocess_dataframe()
        sequences, metadata = self.get_sequences_and_metadata(config.NUM_SAMPLES)
        self.save_processed_data(sequences, metadata)

    def save_processed_data(self, sequences: List[Dict], metadata: Dict):
        with open(self.processed_data_path, 'wb') as f:
            pickle.dump({'sequences': sequences, 'metadata': metadata}, f)
        print(f"Saved processed data to {self.processed_data_path}")

    def load_processed_data(self):
        with open(self.processed_data_path, 'rb') as f:
            data = pickle.load(f)
        return data['sequences'], data['metadata'] 