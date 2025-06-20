import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        # Use global mean pooling to get a single graph representation
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding

class HybridGNNLLMModel(nn.Module):
    def __init__(self, metadata, gnn_hidden_dim=128, llm_model_name="distilbert-base-uncased"):
        super().__init__()
        self.vocab_size = metadata['num_unique_activities']
        
        # 1. LLM component
        self.llm = AutoModel.from_pretrained(llm_model_name)
        # Get the actual embedding dimension from the LLM's input embeddings
        llm_embedding_dim = self.llm.get_input_embeddings().embedding_dim

        # 2. GNN component
        self.gnn_node_embedding = nn.Embedding(self.vocab_size, gnn_hidden_dim, padding_idx=0)
        # Ensure GNN output dimension matches LLM embedding dimension for fusion
        self.gnn_encoder = GNNEncoder(gnn_hidden_dim, gnn_hidden_dim, llm_embedding_dim)

        # 3. Classification component
        # The classification head input must match the LLM's final output dimension
        if hasattr(self.llm.config, 'word_embed_proj_dim'):
            llm_output_dim = self.llm.config.word_embed_proj_dim
        else:
            llm_output_dim = self.llm.config.hidden_size
        self.classification_head = nn.Linear(llm_output_dim, self.vocab_size)
        
        # 4. PEFT Strategy: Freeze most of the LLM
        self._apply_peft()

    def _apply_peft(self):
        # Freeze all LLM parameters first
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # Unfreeze the input embeddings layer of the LLM using a generic method
        for param in self.llm.get_input_embeddings().parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, graph_data):
        # Move graph data to the same device as the model
        graph_data = graph_data.to(input_ids.device)

        # --- GNN Branch ---
        # 1. Embed GNN node features (which are activity indices)
        gnn_node_features = self.gnn_node_embedding(graph_data.x)
        
        # 2. Get graph embedding
        graph_embedding = self.gnn_encoder(gnn_node_features, graph_data.edge_index, graph_data.batch)
        
        # --- LLM Branch ---
        # Get LLM input embeddings using a generic method
        input_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # --- Fusion ---
        # Inject graph embedding into the first token's embedding
        graph_embedding_expanded = graph_embedding.unsqueeze(1)
        input_embeddings[:, 0, :] += graph_embedding_expanded.squeeze(1)

        # Pass fused embeddings through the rest of the LLM
        llm_outputs = self.llm(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        
        # Get the final representation from the last token's hidden state
        sequence_output = llm_outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        final_representation = sequence_output[torch.arange(batch_size, device=input_ids.device), sequence_lengths]

        # --- Classification ---
        logits = self.classification_head(final_representation)
        
        return logits 