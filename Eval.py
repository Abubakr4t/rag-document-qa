
import json
import re
import sys
import statistics

from app import process_pdf, rerank_chunks, llm, PROMPT, RERANK_TOP_K
JUDGE_FAITHFULNESS = """You are evaluating a RAG answer for FAITHFULNESS.
Decide whether EVERY claim in the ANSWER is supported by the CONTEXT.
Reply with ONLY a number from 0.0 to 1.0 (1.0 = fully supported, 0.0 = unsupported / hallucinated). No words.

CONTEXT:
{context}

ANSWER:
{answer}

Score:"""

JUDGE_RELEVANCE = """You are evaluating a RAG answer for RELEVANCE to the question.
Reply with ONLY a number from 0.0 to 1.0 (1.0 = directly and fully answers, 0.0 = off-topic). No words.

QUESTION:
{question}

ANSWER:
{answer}

Score:"""


DEFAULT_QUESTIONS = [
    {"question": "What is this document about?", "must_include": []},
    {"question": "Summarize the key findings.", "must_include": []},
]

def _score(prompt):
    """Run an LLM judge and parse a 0.0-1.0 float from the reply."""
    try:
        txt = llm.invoke(prompt).content.strip()
        m = re.search(r'(?:0(?:\.\d+)?|1(?:\.0+)?)', txt)
        return float(m.group()) if m else None
    except Exception as e:
        print(f"    judge error: {str(e)[:120]}")
        return None
def _keyword_recall(must_include, context):
    """Fraction of expected terms present in the retrieved context."""
    if not must_include:
        return None
    ctx = context.lower()
    hits = sum(1 for kw in must_include if kw.lower() in ctx)
    return hits / len(must_include)
def _build_context(docs):
    return "\n\n---\n\n".join(
        f"[S{i}] (Page {d.metadata.get('page', 'N/A')})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )
def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py <document.pdf> [questions.json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    questions = DEFAULT_QUESTIONS
    if len(sys.argv) > 2:
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            questions = json.load(f)

    print(f"Indexing {pdf_path} ...")
    _, retriever, num_pages, num_chunks = process_pdf(pdf_path)
    print(f"  {num_pages} pages, {num_chunks} chunks\n")

    recalls, faiths, rels = [], [], []

    for i, item in enumerate(questions, 1):
        q = item["question"]

        docs = retriever.invoke(q)
        docs = rerank_chunks(q, docs, top_k=RERANK_TOP_K)
        context = _build_context(docs)

        answer = llm.invoke(
            PROMPT.format(context=context, question=q, history="No previous conversation.")
        ).content.strip()

        kr = _keyword_recall(item.get("must_include"), context)
        fa = _score(JUDGE_FAITHFULNESS.format(context=context, answer=answer))
        rl = _score(JUDGE_RELEVANCE.format(question=q, answer=answer))

        if kr is not None:
            recalls.append(kr)
        if fa is not None:
            faiths.append(fa)
        if rl is not None:
            rels.append(rl)

        print(f"[{i}/{len(questions)}] "
              f"kw_recall={kr if kr is not None else 'n/a'}  "
              f"faith={fa if fa is not None else 'n/a'}  "
              f"rel={rl if rl is not None else 'n/a'}  | {q[:60]}")

    def avg(xs):
        return round(statistics.mean(xs), 3) if xs else "n/a"

    print("\n==================== SUMMARY ====================")
    print(f"questions evaluated : {len(questions)}")
    print(f"avg keyword_recall  : {avg(recalls)}")
    print(f"avg faithfulness    : {avg(faiths)}")
    print(f"avg relevance       : {avg(rels)}")
    print("=================================================")


if __name__ == "__main__":
    main()
