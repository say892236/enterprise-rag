# test_reranker.py
# test_reranker.py

from retriever.chroma_retriever import ChromaRetriever
from reranker.bge_reranker import BGEReranker



"""
模型相关性
+
业务排序规则
=
最终排序


"""




def print_docs(title, docs, score_key):
    print(f"\n====== {title} ======")

    for doc in docs:
        print(
            "source:",
            doc.metadata.get("source"),
            "| header_2:",
            doc.metadata.get("header_2"),
            "| chunk_id:",
            doc.metadata.get("chunk_id"),
            "| chroma_score:",
            doc.metadata.get("score"),
            "| raw_rerank_score:",
            doc.metadata.get("raw_rerank_score"),
            "| boost:",
            doc.metadata.get("rerank_boost"),
            "| boost_reasons:",
            doc.metadata.get("rerank_boost_reasons"),
            "| final_score:",
            doc.metadata.get(score_key),
        )

def main():
    retriever = ChromaRetriever(
        persist_directory="data/vectorstore/chroma",
        collection_name="enterprise_rag",
        model_name="BAAI/bge-m3",
        device="cpu",
        top_k=10,
    )

    # query = "员工出差住宿标准是什么？"
    query = "出差可以住五星级酒店吗？"

    docs = retriever.retrieve(
        query=query,
        top_k=10,
    )

    print_docs(
        title="原始排序",
        docs=docs,
        score_key="score",
    )

    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu",
        max_length=512,
        use_business_boost=True,
    )

    reranked_docs = reranker.rerank(
        query=query,
        documents=docs,
        top_k=5,
    )

    print_docs(
        title="Rerank 后",
        docs=reranked_docs,
        score_key="rerank_score",
    )


if __name__ == "__main__":
    main()