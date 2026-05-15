import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

from cmid_qa.data import CMIDDataset
from cmid_qa.losses import DynamicWeightsLoss, get_loss_functions
from cmid_qa.model import MultiTaskTransformer
from cmid_qa.retrieval import RAGEvaluator
from cmid_qa.visualization import compute_classification_metrics, compute_bleu, plot_tsne, plot_attention_heatmap

# Configuration
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 5e-5
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_PATH = "bert-base-chinese" # Assumes internet access or cached
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "CMID.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "multitask_model.pth"
FIGURE_DIR = PROJECT_ROOT / "artifacts" / "figures"

def train(model, train_loader, optimizer, scheduler, dynamic_loss, device):
    model.train()
    loss_fct_4, loss_fct_36, loss_fct_gen = get_loss_functions(pad_id=0) # BERT pad_id is 0
    
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_4 = batch['labels_4'].to(device)
        labels_36 = batch['labels_36'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits_4, logits_36, gen_logits, _, _ = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            target_ids=target_ids
        )
        
        # Calculate Losses
        l4 = loss_fct_4(logits_4, labels_4)
        l36 = loss_fct_36(logits_36, labels_36)
        
        # Generation Loss: Shift targets
        # logits: [B, Seq, Vocab], target: [B, Seq]
        # We predict next token. Usually target_ids includes [CLS] ... [SEP] [PAD]
        # We want to predict from [CLS] -> first token, etc.
        # Standard Seq2Seq loss calculation:
        # logits[:, :-1, :] vs target_ids[:, 1:]
        shift_logits = gen_logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        l_gen = loss_fct_gen(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Dynamic Weighting
        loss = dynamic_loss(l4, l36, l_gen)
        
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, val_loader, tokenizer, device, dataset):
    model.eval()
    
    preds_4, true_4 = [], []
    preds_36, true_36 = [], []
    generated_texts = []
    original_texts = []
    target_texts = []
    context_vectors = [] # For t-SNE
    
    # For RAG
    rag_samples = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward for classification
            logits_4, logits_36, _, ctx_vec, _ = model(input_ids, attention_mask)
            
            preds_4.extend(torch.argmax(logits_4, dim=1).cpu().numpy())
            true_4.extend(batch['labels_4'].numpy())
            
            preds_36.extend(torch.argmax(logits_36, dim=1).cpu().numpy())
            true_36.extend(batch['labels_36'].numpy())
            
            context_vectors.append(ctx_vec[:, 0, :].cpu().numpy())
            
            # Generation (Greedy)
            gen_ids = model.generate(input_ids, attention_mask)
            gen_texts_batch = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            generated_texts.extend(gen_texts_batch)
            original_texts.extend(batch['original_text'])
            target_texts.extend(batch['target_text'])
            
            # Prepare RAG samples
            for orig, gen, tgt in zip(batch['original_text'], gen_texts_batch, batch['target_text']):
                rag_samples.append({
                    'original': orig,
                    'rewritten': gen,
                    'target': tgt
                })

    # Visualizations (Attention Map for last sample of last batch)
    if len(gen_ids) > 0:
        # Take first sample of last batch
        sample_input_ids = input_ids[0:1]
        sample_attn_mask = attention_mask[0:1]
        sample_gen_ids = gen_ids[0:1]
        
        # Forward to get attention weights
        _, _, _, _, attn_weights = model(sample_input_ids, sample_attn_mask, target_ids=sample_gen_ids)
        if attn_weights is not None:
            attn_matrix = attn_weights[0].detach().cpu().numpy()
            src_tokens = tokenizer.convert_ids_to_tokens(sample_input_ids[0])
            tgt_tokens = tokenizer.convert_ids_to_tokens(sample_gen_ids[0])
            plot_attention_heatmap(attn_matrix, src_tokens, tgt_tokens, save_path=FIGURE_DIR / "attention_heatmap.png")

    # Metrics
    acc_4, f1_4 = compute_classification_metrics(preds_4, true_4)
    acc_36, f1_36 = compute_classification_metrics(preds_36, true_36)
    bleu_score = compute_bleu(generated_texts, target_texts)
    
    print(f"Task 1 (4-Class): Acc={acc_4:.4f}, F1={f1_4:.4f}")
    print(f"Task 2 (36-Class): Acc={acc_36:.4f}, F1={f1_36:.4f}")
    print(f"Task 3 (Gen): BLEU-4={bleu_score:.4f}")
    
    # RAG Evaluation
    print("Running RAG Simulation...")
    # Knowledge Base = All targets in validation set (simplified)
    kb_targets = list(set(target_texts))
    rag_eval = RAGEvaluator(kb_targets)
    rag_results = rag_eval.evaluate(rag_samples)
    print("RAG Retrieval Results (Recall@10):", rag_results)
    
    # Visualizations
    # 1. t-SNE (Subset to avoid clutter if large)
    context_vectors = np.concatenate(context_vectors, axis=0)
    # Using true_4 labels for coloring
    plot_tsne(context_vectors[:500], np.array(true_4)[:500], save_path=FIGURE_DIR / "tsne_context.png")
    
    # 2. Attention Map (Pick one sample)
    # Need to run one forward pass with hook or access weights
    # Simplified: Get weights from last layer of decoder for the last batch
    # This requires modifying model to return weights or accessing them.
    # We will skip or do a quick hack if requested. 
    # For now, let's just use the last batch's info if we had it.
    # To implement properly, we need to access `model.decoder.layers[-1].multihead_attn` weights... 
    # but `nn.TransformerDecoder` doesn't easily expose attention weights during forward without `need_weights=True` 
    # which `TransformerDecoder` forward doesn't support directly in older versions or wraps it.
    # We will assume `vis_utils` handles dummy data or we skip actual heatmap generation here to keep it simple,
    # or we construct a dummy one for demonstration as per "Implement Drawing Function".
    
    return acc_4

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the CMID multi-task Transformer.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to CMID.json.")
    parser.add_argument("--bert-path", default=BERT_PATH, help="Hugging Face model name or local BERT path.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--checkpoint-path", default=str(CHECKPOINT_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    
    # Dataset
    full_dataset = CMIDDataset(args.data_path, tokenizer, max_len=args.max_len)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    model = MultiTaskTransformer(
        pretrained_model_path=args.bert_path,
        num_classes_4=4,
        num_classes_36=len(full_dataset.label_36_map),
        vocab_size=tokenizer.vocab_size
    ).to(DEVICE)
    
    # Dynamic Loss
    dynamic_loss = DynamicWeightsLoss().to(DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(dynamic_loss.parameters()), 
        lr=LEARNING_RATE
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * len(train_loader) * args.epochs),
        num_training_steps=len(train_loader) * args.epochs
    )
    
    # Training Loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        avg_loss = train(model, train_loader, optimizer, scheduler, dynamic_loss, DEVICE)
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Loss Weights: sigma1={torch.exp(dynamic_loss.log_vars[0]).item():.2f}, "
              f"sigma2={torch.exp(dynamic_loss.log_vars[1]).item():.2f}, "
              f"sigma3={torch.exp(dynamic_loss.log_vars[2]).item():.2f}")
        
        evaluate(model, val_loader, tokenizer, DEVICE, val_dataset)
        
    # Save Model
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_path)
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    main()
