# test_retriever.py

from retriever.chroma_retriever import ChromaRetriever


def main():
    retriever = ChromaRetriever(
        persist_directory="data/vectorstore/chroma",
        collection_name="enterprise_rag",
        model_name="BAAI/bge-m3",
        device="cpu",
        top_k=5,
    )
    print("Chroma collection count:", retriever.vectorstore._collection.count())

    query = "员工出差住宿标准是什么？"

    docs = retriever.retrieve(query)

    print(f"问题：{query}")
    print(f"检索结果数量：{len(docs)}")
    print("-" * 80)

    for i, doc in enumerate(docs, start=1):
        print(f"【结果 {i}】")
        print("source:", doc.metadata.get("source"))
        print("header_1:", doc.metadata.get("header_1"))
        print("header_2:", doc.metadata.get("header_2"))
        print("chunk_id:", doc.metadata.get("chunk_id"))
        print("score:", doc.metadata.get("score"))
        print("内容预览:")
        print(doc.page_content[:300])
        print("-" * 80)


if __name__ == "__main__":
    main()