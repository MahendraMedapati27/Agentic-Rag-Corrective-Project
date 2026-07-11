"""
grader.py
---------
Grades retrieved documents as "relevant" or "irrelevant" to the question.
Two backends:
  - DemoGrader: rule-based, zero-cost, deterministic (keyword overlap threshold).
  - LiveGrader: uses Claude to make a real relevance judgment per document.
"""

import re


def _keywords(text: str):
    return set(re.findall(r"[a-z]{3,}", text.lower()))


class DemoGrader:
    """Rule-based grader: relevant if enough query keywords appear in the document."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def grade(self, query: str, document: dict) -> dict:
        q_words = _keywords(query)
        d_words = _keywords(document["title"] + " " + document["content"])
        if not q_words:
            overlap = 0.0
        else:
            overlap = len(q_words & d_words) / len(q_words)

        is_relevant = overlap >= self.threshold or document.get("score", 0) >= 0.22
        return {
            "doc_id": document["id"],
            "relevant": is_relevant,
            "overlap": round(overlap, 3),
            "reasoning": (
                f"{len(q_words & d_words)} of {len(q_words)} query keywords matched; "
                f"retrieval score {document.get('score', 0)}"
            ),
        }


class LiveGrader:
    """Claude-backed grader for real semantic relevance judgments."""

    def __init__(self, client, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def grade(self, query: str, document: dict) -> dict:
        prompt = (
            "You are a strict retrieval grader. Given a user question and one "
            "retrieved document, answer ONLY with 'yes' or 'no' on whether the "
            "document contains information that helps answer the question.\n\n"
            f"Question: {query}\n\n"
            f"Document title: {document['title']}\n"
            f"Document content: {document['content']}\n\n"
            "Is this document relevant? Answer only 'yes' or 'no'."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip().lower()
        is_relevant = answer.startswith("y")
        return {
            "doc_id": document["id"],
            "relevant": is_relevant,
            "overlap": None,
            "reasoning": f"LLM judged relevance as '{answer}'",
        }
