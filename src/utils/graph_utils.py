import torch
import pm4py
import pandas as pd
from torch_geometric.data import Data

def create_dfg(trace_df: pd.DataFrame):
    """
    Creates a Directly-Follows Graph (DFG) from a given trace.
    Args:
        trace_df: A pandas DataFrame representing a single trace,
                  must contain a 'concept:name' column.
    Returns:
        A tuple containing the DFG, start activities, and end activities.
    """
    return pm4py.discover_dfg(trace_df, "concept:name", "time:timestamp")

def dfg_to_pyg_data(dfg: dict, activity_vocab: dict, device: str = 'cpu'):
    """
    Converts a DFG and activity vocabulary into a PyTorch Geometric Data object,
    including self-loops.
    Args:
        dfg: The DFG dictionary, where keys are (source, target) tuples
             and values are frequencies.
        activity_vocab: A dictionary mapping activity names to integer indices.
        device: The torch device to place tensors on.
    Returns:
        A PyTorch Geometric Data object.
    """
    num_nodes = len(activity_vocab)
    
    # Node features: Start with simple one-hot encoding for now
    node_features = torch.eye(num_nodes, dtype=torch.float)
    
    # Get all unique activities from the DFG to build edges
    activities_in_dfg = set()
    for source, target in dfg.keys():
        activities_in_dfg.add(source)
        activities_in_dfg.add(target)

    edge_sources = []
    edge_targets = []
    edge_weights = []

    # Add DFG edges
    for (source, target), weight in dfg.items():
        if source in activity_vocab and target in activity_vocab:
            edge_sources.append(activity_vocab[source])
            edge_targets.append(activity_vocab[target])
            edge_weights.append(float(weight))

    # Add self-loops for all nodes present in the DFG
    for activity in activities_in_dfg:
        if activity in activity_vocab:
            node_idx = activity_vocab[activity]
            edge_sources.append(node_idx)
            edge_targets.append(node_idx)
            edge_weights.append(1.0) # Self-loop weight, can be adjusted

    if not edge_sources:
        # Handle case with no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr).to(device) 