# test_rag_chain_with_reranker.py

from retriever.chroma_retriever import ChromaRetriever
from reranker.bge_reranker import BGEReranker
from chains.rag_chain import EnterpriseRAGChain


def main():
    retriever = ChromaRetriever(
        persist_directory="data/vectorstore/chroma",
        collection_name="enterprise_rag",
        model_name="BAAI/bge-m3",
        device="cpu",
        top_k=10,
    )

    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu",
        max_length=512,
        use_business_boost=True,
    )

    rag_chain = EnterpriseRAGChain(
        retriever=retriever,
        reranker=reranker,
        llm_model="deepseek-v4-flash",
        llm_provider="deepseek",
        temperature=0,
    )

    question = "员工出差住宿标准是什么？"

    result = rag_chain.invoke(
        question=question,
        retrieve_top_k=10,
        rerank_top_k=5,
    )

    print("=" * 80)
    print("问题：")
    print(result["question"])

    print("=" * 80)
    print("答案：")
    print(result["answer"])

    print("=" * 80)
    print("引用来源：")

    for i, source in enumerate(result["sources"], start=1):
        print(f"【来源 {i}】")
        print("source:", source.get("source"))
        print("header_1:", source.get("header_1"))
        print("header_2:", source.get("header_2"))
        print("chunk_id:", source.get("chunk_id"))
        print("chroma_score:", source.get("chroma_score"))
        print("raw_rerank_score:", source.get("raw_rerank_score"))
        print("rerank_boost:", source.get("rerank_boost"))
        print("rerank_score:", source.get("rerank_score"))
        print("-" * 80)


if __name__ == "__main__":
    main()