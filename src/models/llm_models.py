import torch
import torch.nn as nn
from transformers import AutoModel

class LLMClassifier(nn.Module):
    def __init__(self, vocab_size, llm_model_name, finetune_llm=False):
        super().__init__()
        self.llm = AutoModel.from_pretrained(llm_model_name)
        
        # For OPT models, the final projection layer dimension can be different from hidden_size
        if hasattr(self.llm.config, 'word_embed_proj_dim'):
            llm_output_dim = self.llm.config.word_embed_proj_dim
        else:
            llm_output_dim = self.llm.config.hidden_size
            
        self.classification_head = nn.Linear(llm_output_dim, vocab_size)
        
        self.set_finetune_mode(finetune_llm)

    def set_finetune_mode(self, finetune_llm):
        """Freeze or unfreeze LLM parameters based on the flag."""
        for param in self.llm.parameters():
            param.requires_grad = finetune_llm
        print(f"LLM parameters are {'trainable' if finetune_llm else 'frozen'}.")

    def forward(self, input_ids, attention_mask):
        # 1. Pass through LLM
        llm_outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Get the final representation from the last token's hidden state
        sequence_output = llm_outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        final_representation = sequence_output[torch.arange(batch_size, device=input_ids.device), sequence_lengths]

        # 3. Pass through the classification head
        logits = self.classification_head(final_representation)
        
        return logits 