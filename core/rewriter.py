"""
rewriter.py
-----------
Rewrites a query when retrieval comes back weak, so the next retrieval
pass has a better chance of surfacing relevant documents.
"""

SYNONYM_EXPANSIONS = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "hallucination": "unsupported ungrounded incorrect claim",
    "grading": "evaluating scoring relevance judgment",
    "agent": "autonomous reasoning system",
}


class DemoRewriter:
    """Deterministic rewrite: expands known jargon/acronyms and strips filler words."""

    FILLER = {"please", "can", "you", "tell", "me", "about", "what", "is", "the", "a", "an"}

    def rewrite(self, original_query: str, weak_results: list) -> str:
        words = [w for w in original_query.lower().split() if w not in self.FILLER]
        expanded = []
        for w in words:
            clean = w.strip("?.,!")
            expanded.append(SYNONYM_EXPANSIONS.get(clean, clean))
        rewritten = " ".join(expanded).strip()
        return rewritten if rewritten else original_query


class LiveRewriter:
    """Claude-backed query rewriter for genuinely better reformulations."""

    def __init__(self, client, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def rewrite(self, original_query: str, weak_results: list) -> str:
        titles = ", ".join(r["title"] for r in weak_results) or "none"
        prompt = (
            "The following search query returned weak or irrelevant results from a "
            f"knowledge base. Weakly-matching document titles were: {titles}.\n\n"
            f"Original query: {original_query}\n\n"
            "Rewrite this into a single, more specific search query that is more "
            "likely to retrieve relevant documents. Respond with ONLY the rewritten "
            "query, no explanation."
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
