import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_weights = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        ff = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(ff)
        tgt = self.norm3(tgt)
        return tgt, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers=4, dim_feedforward=2048, dropout=0.1, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, target_ids, memory, target_mask=None, target_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.embedding(target_ids)
        x = self.pos_encoding(x)
        attn = None
        for layer in self.layers:
            x, attn = layer(
                x,
                memory,
                tgt_mask=target_mask,
                tgt_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x, attn


class CustomMultiTaskTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes_4,
        num_classes_36,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        max_len=128,
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
        )
        self.classifier_4 = nn.Linear(d_model, num_classes_4)
        self.classifier_36 = nn.Linear(d_model, num_classes_36)
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, target_ids=None, target_mask=None):
        memory = self.encoder(input_ids, attention_mask=attention_mask)
        context = memory[:, 0, :]
        logits_4 = self.classifier_4(context)
        logits_36 = self.classifier_36(context)
        gen_logits = None
        attn = None
        if target_ids is not None:
            tgt_len = target_ids.size(1)
            tgt_causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=target_ids.device) * float("-inf"), diagonal=1
            )
            tgt_key_padding_mask = None
            if target_mask is not None:
                tgt_key_padding_mask = target_mask == 0
            mem_key_padding_mask = None
            if attention_mask is not None:
                mem_key_padding_mask = attention_mask == 0
            dec_out, attn = self.decoder(
                target_ids,
                memory,
                target_mask=tgt_causal_mask,
                target_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=mem_key_padding_mask,
            )
            gen_logits = self.generator(dec_out)
        return logits_4, logits_36, gen_logits, memory, attn

    def generate(self, input_ids, attention_mask=None, max_length=32, start_token_id=101, end_token_id=102):
        memory = self.encoder(input_ids, attention_mask=attention_mask)
        batch_size = input_ids.size(0)
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=input_ids.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        for _ in range(max_length):
            tgt_len = generated.size(1)
            tgt_causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=input_ids.device) * float("-inf"), diagonal=1
            )
            dec_out, _ = self.decoder(
                generated,
                memory,
                target_mask=tgt_causal_mask,
                target_key_padding_mask=None,
                memory_key_padding_mask=attention_mask == 0 if attention_mask is not None else None,
            )
            logits = self.generator(dec_out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == end_token_id)
            if finished.all():
                break
        return generated


def _demo():
    vocab_size = 30522
    model = CustomMultiTaskTransformer(vocab_size=vocab_size, num_classes_4=4, num_classes_36=36)
    model.eval()
    batch_size = 2
    src_len = 10
    tgt_len = 8
    input_ids = torch.randint(0, vocab_size, (batch_size, src_len))
    attention_mask = torch.ones(batch_size, src_len, dtype=torch.long)
    target_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))
    with torch.no_grad():
        logits_4, logits_36, gen_logits, memory, attn = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=target_ids,
        )
    print("logits_4", logits_4.shape)
    print("logits_36", logits_36.shape)
    print("gen_logits", gen_logits.shape)
    print("memory", memory.shape)
    if attn is not None:
        print("attn", attn.shape)


if __name__ == "__main__":
    _demo()

