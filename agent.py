"""
agent.py
--------
The Corrective Agentic RAG loop.

State machine:

    retrieve -> grade -> [enough relevant docs?] --yes--> generate -> critique -> [grounded?] --yes--> DONE
                              |no                                                     |no
                              v                                                       v
                          rewrite query                                   regenerate with remaining docs
                              |                                                       |
                              +--------------------(loop, up to MAX_ITERATIONS)-------+
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from core.retriever import Retriever
from core.grader import DemoGrader, LiveGrader
from core.rewriter import DemoRewriter, LiveRewriter
from core.generator import DemoGenerator, LiveGenerator
from core.critic import DemoCritic, LiveCritic

MAX_ITERATIONS = 3
MIN_RELEVANT_DOCS = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


class CorrectiveRAGAgent:
    def __init__(self, mode: str = "demo", verbose: bool = True):
        self.mode = mode
        self.verbose = verbose
        self.retriever = Retriever()
        self.trace = []

        if mode == "live":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Add it to your .env file to use --live mode."
                )
            try:
                import anthropic
            except ImportError:
                raise RuntimeError(
                    "The 'anthropic' package is not installed. Run: pip install anthropic"
                )
            client = anthropic.Anthropic(api_key=api_key)
            self.grader = LiveGrader(client)
            self.rewriter = LiveRewriter(client)
            self.generator = LiveGenerator(client)
            self.critic = LiveCritic(client)
        else:
            self.grader = DemoGrader()
            self.rewriter = DemoRewriter()
            self.generator = DemoGenerator()
            self.critic = DemoCritic()

    def _log(self, step: str, detail: str):
        entry = {"step": step, "detail": detail}
        self.trace.append(entry)
        if self.verbose:
            print(f"[{step}] {detail}")

    def run(self, question: str) -> dict:
        query = question
        relevant_docs = []
        iteration = 0

        # --- retrieve -> grade -> rewrite loop ---
        while iteration < MAX_ITERATIONS:
            iteration += 1
            self._log("RETRIEVE", f"iteration {iteration} | query = '{query}'")
            candidates = self.retriever.retrieve(query, k=3)

            graded = [self.grader.grade(query, doc) for doc in candidates]
            newly_relevant = [
                doc for doc, g in zip(candidates, graded) if g["relevant"]
            ]
            for doc, g in zip(candidates, graded):
                self._log(
                    "GRADE",
                    f"'{doc['title']}' -> {'RELEVANT' if g['relevant'] else 'irrelevant'} ({g['reasoning']})",
                )

            for doc in newly_relevant:
                if doc["id"] not in {d["id"] for d in relevant_docs}:
                    relevant_docs.append(doc)

            if len(relevant_docs) >= MIN_RELEVANT_DOCS:
                break

            if iteration < MAX_ITERATIONS:
                new_query = self.rewriter.rewrite(query, candidates)
                self._log("REWRITE", f"'{query}' -> '{new_query}'")
                query = new_query

        if not relevant_docs:
            self._log("RETRIEVE", "No relevant documents found after max iterations. Answering with no context.")

        # --- generate -> critique loop ---
        gen_iteration = 0
        answer = ""
        remaining_docs = list(relevant_docs)

        while gen_iteration < MAX_ITERATIONS:
            gen_iteration += 1
            answer = self.generator.generate(question, remaining_docs)
            self._log("GENERATE", f"draft #{gen_iteration} produced ({len(answer)} chars)")

            verdict = self.critic.check(answer, remaining_docs)
            self._log(
                "CRITIQUE",
                f"{'GROUNDED' if verdict['grounded'] else 'NOT GROUNDED'} - {verdict['reasoning']}",
            )

            if verdict["grounded"] or not remaining_docs:
                break

            # Drop the weakest remaining doc and try again with a leaner context,
            # forcing the generator to lean only on what it can actually support.
            if len(remaining_docs) > 1:
                remaining_docs = remaining_docs[:-1]

        result = {
            "question": question,
            "final_query_used": query,
            "relevant_documents": relevant_docs,
            "iterations_retrieval": iteration,
            "iterations_generation": gen_iteration,
            "answer": answer,
            "grounded": verdict["grounded"],
            "trace": self.trace,
        }
        return result

    def save_report(self, result: dict) -> str:
        safe_title = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "" for c in result["question"][:60]
        ).strip().replace(" ", "_")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_title or 'report'}.md"
        filepath = os.path.join(REPORTS_DIR, filename)

        lines = [
            f"# Research Answer: {result['question']}",
            "",
            f"**Grounded:** {result['grounded']}  ",
            f"**Retrieval iterations:** {result['iterations_retrieval']}  ",
            f"**Generation iterations:** {result['iterations_generation']}  ",
            "",
            "## Answer",
            "",
            result["answer"],
            "",
            "## Sources",
            "",
        ]
        for doc in result["relevant_documents"]:
            lines.append(f"- `{doc['id']}` — {doc['title']}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Corrective Agentic RAG demo")
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument("--live", action="store_true", help="Use real Claude calls instead of the offline demo backend")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step trace output")
    args = parser.parse_args()

    mode = "live" if args.live else "demo"
    try:
        agent = CorrectiveRAGAgent(mode=mode, verbose=not args.quiet)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    result = agent.run(args.question)
    path = agent.save_report(result)

    print("\n" + "=" * 70)
    print(result["answer"])
    print("=" * 70)
    print(f"\nSaved report to: {path}")


if __name__ == "__main__":
    main()
