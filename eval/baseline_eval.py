# eval/baseline_eval.py

"""
Baseline 评估。

链路：
Chroma Retriever → LLM
"""

from eval.questions import TEST_QUESTIONS
from eval.eval_runner import run_eval
from pipelines.rag_pipeline import build_baseline_rag_chain


def main():
    rag_chain = build_baseline_rag_chain()

    run_eval(
        pipeline_name="baseline",
        rag_chain=rag_chain,
        questions=TEST_QUESTIONS,
        output_file="data/eval/baseline_results.jsonl",
        retrieve_top_k=5,
        rerank_top_k=5,
    )


if __name__ == "__main__":
    main()