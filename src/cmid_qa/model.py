import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross Attention
        # memory is encoder output
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed Forward
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn_weights

class CustomDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead):
        super().__init__()
        self.layers = nn.ModuleList([CustomDecoderLayer(d_model, nhead) for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        last_attn_weights = None
        for layer in self.layers:
            output, last_attn_weights = layer(output, memory, tgt_mask=tgt_mask, 
                                              tgt_key_padding_mask=tgt_key_padding_mask, 
                                              memory_key_padding_mask=memory_key_padding_mask)
        return output, last_attn_weights

class MultiTaskTransformer(nn.Module):
    def __init__(self, pretrained_model_path, num_classes_4, num_classes_36, vocab_size, decoder_layers=6, decoder_heads=8, hidden_dim=768):
        super().__init__()
        
        # --- Encoder (Pre-trained BERT) ---
        self.encoder = BertModel.from_pretrained(pretrained_model_path)
        self.hidden_dim = hidden_dim
        
        # --- Custom Decoder ---
        # Hand-written Decoder Stack
        self.decoder = CustomDecoder(decoder_layers, hidden_dim, decoder_heads)
        self.decoder_embeddings = self.encoder.embeddings # Share embeddings with encoder
        
        # --- Multi-Task Heads ---
        self.classifier_4 = nn.Linear(hidden_dim, num_classes_4)
        self.classifier_36 = nn.Linear(hidden_dim, num_classes_36)
        self.generator = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids, attention_mask, target_ids=None, target_mask=None):
        # --- Encoder Pass ---
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # [Batch, Seq, Dim]
        
        # Context Vector: Use CLS token (first token)
        context_vector = sequence_output[:, 0, :] # [Batch, Dim]
        
        # --- Classification Tasks ---
        logits_4 = self.classifier_4(context_vector)
        logits_36 = self.classifier_36(context_vector)
        
        # --- Generation Task (Decoder Pass) ---
        gen_logits = None
        cross_attention_weights = None
        
        if target_ids is not None:
            # Embeddings for decoder input
            tgt_embeddings = self.decoder_embeddings(input_ids=target_ids)
            
            # Create masks
            seq_len = target_ids.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(target_ids.device)
            
            if target_mask is None:
                tgt_key_padding_mask = (target_ids == 0)
            else:
                tgt_key_padding_mask = (target_mask == 0)

            memory_key_padding_mask = (attention_mask == 0)
            
            # Decoder Forward
            decoder_output, cross_attention_weights = self.decoder(
                tgt=tgt_embeddings,
                memory=sequence_output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            gen_logits = self.generator(decoder_output)
            
        return logits_4, logits_36, gen_logits, sequence_output, cross_attention_weights

    def generate(self, input_ids, attention_mask, max_length=50, start_token_id=101, end_token_id=102):
        # Greedy decoding for inference
        bs = input_ids.size(0)
        
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        memory = outputs.last_hidden_state
        
        # Initial decoder input: [CLS]
        curr_ids = torch.full((bs, 1), start_token_id, dtype=torch.long, device=input_ids.device)
        
        finished = torch.zeros(bs, dtype=torch.bool, device=input_ids.device)
        
        for _ in range(max_length):
            tgt_embeddings = self.decoder_embeddings(input_ids=curr_ids)
            
            seq_len = curr_ids.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(input_ids.device)
            
            decoder_output, _ = self.decoder(
                tgt=tgt_embeddings,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=(attention_mask == 0)
            )
            
            # Get last token output
            next_token_logits = self.generator(decoder_output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Update finished status
            is_end = (next_token.squeeze() == end_token_id)
            finished = finished | is_end
            
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            if finished.all():
                break
                
        return curr_ids
