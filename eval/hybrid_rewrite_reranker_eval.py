# eval/hybrid_rewrite_reranker_eval.py

"""
Hybrid + Rewrite + Reranker 评估。

链路：
Query Rewrite
→ Hybrid Retriever(Dense + BM25)
→ BGE Reranker
→ LLM
"""

from eval.questions import TEST_QUESTIONS
from eval.eval_runner import run_eval
from pipelines.rag_pipeline import build_hybrid_rag_chain


def main():
    rag_chain = build_hybrid_rag_chain()

    run_eval(
        pipeline_name="hybrid_rewrite_reranker",
        rag_chain=rag_chain,
        questions=TEST_QUESTIONS,
        output_file="data/eval/hybrid_rewrite_reranker_results.jsonl",
        retrieve_top_k=10,
        rerank_top_k=5,
    )


if __name__ == "__main__":
    main()