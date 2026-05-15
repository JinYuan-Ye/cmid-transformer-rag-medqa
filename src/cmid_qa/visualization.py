import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_classification_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return acc, macro_f1

def compute_bleu(generated_texts, reference_texts):
    scores = []
    cc = SmoothingFunction()
    for gen, ref in zip(generated_texts, reference_texts):
        # Ref needs to be list of list of tokens, gen is list of tokens
        # We assume texts are already tokenized strings or list of chars
        # Here we do simple char level or space split
        gen_tokens = list(gen)
        ref_tokens = list(ref)
        score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=cc.method1)
        scores.append(score)
    return np.mean(scores)

def plot_tsne(features, labels, save_path="tsne.png"):
    """
    features: [N, Dim] numpy array
    labels: [N] numpy array (integers)
    """
    tsne = TSNE(n_components=2, random_state=42)
    proj = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE of Encoder Context Vectors")
    plt.savefig(save_path)
    plt.close()

def plot_attention_heatmap(attention_matrix, input_tokens, generated_tokens, save_path="attention.png"):
    """
    attention_matrix: [Gen_Len, Input_Len] numpy array
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_matrix, xticklabels=input_tokens, yticklabels=generated_tokens, cmap="viridis")
    plt.xlabel("Input Tokens")
    plt.ylabel("Generated Tokens")
    plt.title("Cross-Attention Heatmap")
    plt.savefig(save_path)
    plt.close()
