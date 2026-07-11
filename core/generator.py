"""
generator.py
------------
Turns a question plus a set of graded-relevant documents into a final,
cited answer.
"""


class DemoGenerator:
    """Deterministic, template-based generator. No LLM required."""

    def generate(self, question: str, documents: list) -> str:
        if not documents:
            return (
                f"I could not find grounded information to answer: '{question}'. "
                "The knowledge base does not appear to cover this topic."
            )

        lines = [f"**Answer:** Based on {len(documents)} relevant source(s):\n"]
        for doc in documents:
            lines.append(f"- **{doc['title']}** (`{doc['id']}`): {doc['content']}")
        lines.append(
            "\nIn summary, "
            + " ".join(d["content"].split(".")[0] + "." for d in documents)
        )
        return "\n".join(lines)


class LiveGenerator:
    """Claude-backed generator for genuinely fluent, synthesized answers."""

    def __init__(self, client, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def generate(self, question: str, documents: list) -> str:
        if not documents:
            context = "No relevant documents were found in the knowledge base."
        else:
            context = "\n\n".join(
                f"[{d['id']}] {d['title']}: {d['content']}" for d in documents
            )

        prompt = (
            "Answer the user's question using ONLY the context provided below. "
            "Cite document ids like [d003] inline. If the context does not contain "
            "the answer, say so honestly instead of guessing.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
