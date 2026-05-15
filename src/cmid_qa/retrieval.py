import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEvaluator:
    """TF-IDF retrieval evaluator for raw, keyword, and rewritten queries."""

    def __init__(self, knowledge_base_questions):
        self.kb = list(set(knowledge_base_questions))
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut)
        self.kb_vectors = self.vectorizer.fit_transform(self.kb)

    def retrieve(self, query, top_k=10):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.kb_vectors).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        return [self.kb[i] for i in top_indices]

    def evaluate(self, test_samples):
        results = {
            "Raw Query": 0,
            "Keyword Search": 0,
            "Rewritten Query": 0,
        }
        total = len(test_samples)
        if total == 0:
            return {key: 0.0 for key in results}

        for sample in test_samples:
            target = sample["target"]
            if target in self.retrieve(sample["original"]):
                results["Raw Query"] += 1

            keywords = " ".join(jieba.cut(sample["original"]))
            if target in self.retrieve(keywords):
                results["Keyword Search"] += 1

            if target in self.retrieve(sample["rewritten"]):
                results["Rewritten Query"] += 1

        return {key: value / total for key, value in results.items()}
