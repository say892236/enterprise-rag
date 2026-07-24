# test_hybrid_rag_chain.py

"""
Hybrid RAG Chain 测试脚本。

目标：
1. 使用 Query Rewrite 改写用户问题。
2. 使用 Hybrid Retriever 同时执行 Dense + BM25 检索。
3. 使用 BGE Reranker 对 Hybrid 结果重排。
4. 使用 LLM 基于检索上下文生成答案。
5. 打印完整溯源信息，方便观察每个 chunk 是怎么被召回和排序的。
"""

from retriever.chroma_retriever import ChromaRetriever
from retriever.bm25_retriever import BM25Retriever
from retriever.hybrid_retriever import HybridRetriever

from reranker.bge_reranker import BGEReranker
from rewriter.query_rewriter import QueryRewriter
from chains.rag_chain import EnterpriseRAGChain


def print_sources(sources):
    """
    打印来源信息。

    这里会同时打印：
    - Dense 检索排名
    - BM25 检索排名
    - Hybrid 融合分数
    - Reranker 重排分数
    """

    for i, source in enumerate(sources, start=1):
        print(f"【来源 {i}】")

        print("source:", source.get("source"))
        print("header_1:", source.get("header_1"))
        print("header_2:", source.get("header_2"))
        print("chunk_id:", source.get("chunk_id"))

        print("retrieval_sources:", source.get("retrieval_sources"))

        print("dense_rank:", source.get("dense_rank"))
        print("chroma_score:", source.get("chroma_score"))
        print("dense_rrf_score:", source.get("dense_rrf_score"))

        print("bm25_rank:", source.get("bm25_rank"))
        print("bm25_score:", source.get("bm25_score"))
        print("bm25_rrf_score:", source.get("bm25_rrf_score"))

        print("hybrid_score:", source.get("hybrid_score"))

        print("raw_rerank_score:", source.get("raw_rerank_score"))
        print("rerank_boost:", source.get("rerank_boost"))
        print("rerank_boost_reasons:", source.get("rerank_boost_reasons"))
        print("rerank_score:", source.get("rerank_score"))

        print("-" * 100)


def main():
    # 1. 初始化 Dense Retriever
    # 负责语义召回。
    dense_retriever = ChromaRetriever(
        persist_directory="data/vectorstore/chroma",
        collection_name="enterprise_rag",
        model_name="BAAI/bge-m3",
        device="cpu",
        top_k=10,
    )

    # 2. 初始化 BM25 Retriever
    # 负责编号、关键词、金额、条款号等精确召回。
    bm25_retriever = BM25Retriever(
        chunks_file="data/processed/chunks.jsonl",
        top_k=10,
    )

    # 3. 初始化 Hybrid Retriever
    # 同时调用 Dense 和 BM25，然后使用 RRF 做融合排序。
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=10,
        dense_weight=1.0,
        bm25_weight=1.0,
        rrf_k=60,
    )

    # 4. 初始化 Reranker
    # 对 Hybrid 召回结果做精排。
    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu",
        max_length=512,
        use_business_boost=True,
    )

    # 5. 初始化 Query Rewriter
    # 把用户口语化问题改写成更适合检索的问题。
    rewriter = QueryRewriter(
        llm_model="deepseek-v4-flash",
        llm_provider="deepseek",
        temperature=0,
    )

    # 6. 初始化完整 RAG Chain
    # 注意：这里传入的是 hybrid_retriever，不再是单独的 ChromaRetriever。
    rag_chain = EnterpriseRAGChain(
        retriever=hybrid_retriever,
        reranker=reranker,
        rewriter=rewriter,
        llm_model="deepseek-v4-flash",
        llm_provider="deepseek",
        temperature=0,
    )

    # 7. 测试问题
    # 这里选一个编号类问题，验证 BM25 是否能帮上忙。
    question = "FIN-TR-2026-01 是什么制度？"

    result = rag_chain.invoke(
        question=question,
        retrieve_top_k=10,
        rerank_top_k=5,
    )

    print("=" * 100)
    print("原始问题：")
    print(result["question"])

    print("=" * 100)
    print("改写问题：")
    print(result.get("rewritten_question"))

    print("=" * 100)
    print("答案：")
    print(result["answer"])

    print("=" * 100)
    print("引用来源：")
    print_sources(result["sources"])


if __name__ == "__main__":
    main()