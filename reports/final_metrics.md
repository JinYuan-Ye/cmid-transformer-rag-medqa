# CMID Multi-Task Transformer Evaluation Report

## Metrics From Existing Run

| Metric | Value |
| --- | ---: |
| Accuracy (L4) | 0.5522 |
| Macro-F1 (L4) | 0.4173 |
| Accuracy (L36) | 0.3638 |
| Macro-F1 (L36) | 0.2009 |
| BLEU-4 (Generation) | 0.5530 |

## Metrics From Presentation

| Metric | Value |
| --- | ---: |
| Accuracy (L4) | 72.02% |
| Accuracy (L36) | 48.04% |
| BLEU-4 | 0.2407 |
| RAG Recall@1 (Ours) | 28.00% |
| RAG Recall@1 (Raw) | 10.00% |

## Artifacts

- Training loss: `artifacts/figures/training_loss.png`
- Encoder training metrics: `artifacts/figures/encoder_training_metrics.png`
- L4 confusion matrix: `artifacts/figures/confusion_matrix_l4.png`
- t-SNE clusters: `artifacts/figures/tsne_clusters.png`
- Encoder context t-SNE: `artifacts/figures/tsne_context.png`
- Cross-attention heatmap: `artifacts/figures/attention_heatmap.png`

## RAG Ablation Summary

The historical report compares three retrieval settings:

- Raw query retrieval.
- Keyword retrieval after tokenization.
- Rewritten query retrieval produced by the Decoder.

The presentation version reports a gain from 10.00% Recall@1 for raw queries to 28.00% Recall@1 for rewritten queries, supporting the design goal that query rewriting can align noisy patient expressions with standard knowledge-base keys.
