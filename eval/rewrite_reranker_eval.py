# eval/rewrite_reranker_eval.py

"""
Rewrite + Reranker 评估。

链路：
Query Rewrite → Chroma Retriever → BGE Reranker → LLM
"""

from eval.questions import TEST_QUESTIONS
from eval.eval_runner import run_eval
from pipelines.rag_pipeline import build_rewrite_reranker_rag_chain


def main():
    rag_chain = build_rewrite_reranker_rag_chain()

    run_eval(
        pipeline_name="rewrite_reranker",
        rag_chain=rag_chain,
        questions=TEST_QUESTIONS,
        output_file="data/eval/rewrite_reranker_results.jsonl",
        retrieve_top_k=10,
        rerank_top_k=5,
    )


if __name__ == "__main__":
    main()