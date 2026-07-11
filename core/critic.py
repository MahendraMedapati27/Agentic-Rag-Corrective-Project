"""
critic.py
---------
Checks whether a generated answer is actually grounded in the retrieved
documents, or whether it drifted into unsupported claims.
"""

import re


def _keywords(text: str):
    return set(re.findall(r"[a-z]{4,}", text.lower()))


class DemoCritic:
    """Rule-based groundedness check: how much of the answer's vocabulary
    overlaps with the source documents' vocabulary."""

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def check(self, answer: str, documents: list) -> dict:
        if not documents:
            return {"grounded": False, "overlap": 0.0, "reasoning": "No source documents to ground against."}

        answer_words = _keywords(answer)
        source_words = set()
        for d in documents:
            source_words |= _keywords(d["title"] + " " + d["content"])

        if not answer_words:
            overlap = 0.0
        else:
            overlap = len(answer_words & source_words) / len(answer_words)

        grounded = overlap >= self.threshold
        return {
            "grounded": grounded,
            "overlap": round(overlap, 3),
            "reasoning": f"{round(overlap * 100)}% of the answer's vocabulary is traceable to the source documents.",
        }


class LiveCritic:
    """Claude-backed groundedness judge."""

    def __init__(self, client, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def check(self, answer: str, documents: list) -> dict:
        context = "\n\n".join(f"[{d['id']}] {d['content']}" for d in documents) or "No documents."
        prompt = (
            "You are a strict fact-checker. Given source documents and a generated "
            "answer, respond ONLY with 'grounded' if every claim in the answer is "
            "supported by the sources, or 'ungrounded' if the answer contains claims "
            "not supported by the sources.\n\n"
            f"Sources:\n{context}\n\nAnswer:\n{answer}\n\nVerdict:"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = response.content[0].text.strip().lower()
        grounded = "ungrounded" not in verdict and "grounded" in verdict
        return {"grounded": grounded, "overlap": None, "reasoning": f"LLM verdict: '{verdict}'"}
