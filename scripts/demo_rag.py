import argparse
import json
import random
from pathlib import Path

import jieba
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

from cmid_qa.data import CMIDDataset
from cmid_qa.model import MultiTaskTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_kb_answers(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    return questions, answers


def vectorize_kb(questions):
    vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, norm="l2")
    kb_vectors = vectorizer.fit_transform(questions)
    return vectorizer, kb_vectors


def compute_cosine_scores(query_vec, kb_vectors):
    query = query_vec.toarray().flatten()
    kb = kb_vectors.toarray()
    dot = kb @ query
    query_norm = np.linalg.norm(query) + 1e-12
    kb_norm = np.linalg.norm(kb, axis=1) + 1e-12
    scores = dot / (query_norm * kb_norm)
    return scores


def load_model(args, dataset, tokenizer, device):
    model = MultiTaskTransformer(
        pretrained_model_path=args.bert_path,
        num_classes_4=4,
        num_classes_36=len(dataset.label_36_map),
        vocab_size=tokenizer.vocab_size,
    ).to(device)
    checkpoint = Path(args.model_path)
    if checkpoint.exists():
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"未找到权重文件 {checkpoint}，将使用未训练模型演示流程。")
    model.eval()
    return model


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    kb_questions, kb_answers = build_kb_answers(args.kb_path)
    vectorizer, kb_vectors = vectorize_kb(kb_questions)
    print(f"知识库条目: {len(kb_questions)}")
    print(f"TF-IDF 特征维度: {len(vectorizer.vocabulary_)}")

    dataset = CMIDDataset(args.data_path, tokenizer, max_len=args.max_len)
    sample = dataset[args.index] if args.index is not None else dataset[random.randint(0, len(dataset) - 1)]
    original_text = sample["original_text"]

    model = load_model(args, dataset, tokenizer, device)
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask, max_length=args.max_gen_len)
    rewritten = tokenizer.decode(gen_ids[0], skip_special_tokens=True).replace(" ", "")

    print("\n=== 检索流程 ===")
    print(f"原始查询: {original_text}")
    print(f"重写查询: {rewritten}")
    print(f"原始分词: {jieba.lcut(original_text)}")
    print(f"重写分词: {jieba.lcut(rewritten)}")
    print("相似度公式: cos(q,d) = (q*d) / (||q||*||d||)")

    for title, query in [("原始查询", original_text), ("重写查询", rewritten)]:
        query_vec = vectorizer.transform([query])
        scores = compute_cosine_scores(query_vec, kb_vectors)
        top_indices = scores.argsort()[-args.top_k:][::-1]
        print(f"\n{title} Top-{args.top_k}:")
        for rank, idx in enumerate(top_indices, start=1):
            print(f"{rank}. score={scores[idx]:.4f} question={kb_questions[idx]}")
        best_idx = top_indices[0]
        print(f"返回回答: {kb_answers[best_idx]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a retrieval-augmented medical QA demo.")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "raw" / "CMID.json"))
    parser.add_argument("--kb-path", default=str(PROJECT_ROOT / "data" / "kb" / "generated_answers.json"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoints" / "multitask_model.pth"))
    parser.add_argument("--bert-path", default="bert-base-chinese")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--max-gen-len", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--index", type=int, default=None, help="Optional CMID sample index for deterministic demo.")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
