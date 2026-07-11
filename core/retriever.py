"""
retriever.py
------------
A small TF-IDF retriever over the local knowledge base.
Stands in for a production vector database (Pinecone, Qdrant, pgvector, etc.)
The rest of the pipeline never needs to know the difference.
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.json")


class Retriever:
    def __init__(self, data_path: str = DATA_PATH):
        with open(data_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        self.corpus = [f"{d['title']}. {d['content']}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, k: int = 3):
        """Return the top-k documents for a query, each with a similarity score."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked_idx = scores.argsort()[::-1][:k]

        results = []
        for idx in ranked_idx:
            doc = self.docs[idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "score": round(float(scores[idx]), 4),
            })
        return results
