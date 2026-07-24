# eval/reranker_eval.py

"""
Reranker 评估。

链路：
Chroma Retriever → BGE Reranker → LLM
"""

from eval.questions import TEST_QUESTIONS
from eval.eval_runner import run_eval
from pipelines.rag_pipeline import build_reranker_rag_chain


def main():
    rag_chain = build_reranker_rag_chain()

    run_eval(
        pipeline_name="reranker",
        rag_chain=rag_chain,
        questions=TEST_QUESTIONS,
        output_file="data/eval/reranker_results.jsonl",
        retrieve_top_k=10,
        rerank_top_k=5,
    )


if __name__ == "__main__":
    main()