# retriever/chroma_retriever.py

"""
Chroma Retriever 模块。

作用：
1. 从已经构建好的 Chroma 向量库中检索相关 chunks。
2. 支持 metadata filter。
3. 返回 LangChain Document 列表。
4. 将 Chroma score 写入 metadata，方便调试。
"""

from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma

from embeddings.bge_embedding import create_bge_embeddings


class ChromaRetriever:
    """
    Chroma 向量检索器。

    当前职责：
    - 加载已有 Chroma 向量库
    - 执行 similarity search
    - 返回带 score 的 Document
    """

    def __init__(
        self,
        persist_directory: str = "data/vectorstore/chroma",
        collection_name: str = "enterprise_rag",
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        top_k: int = 5,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device
        self.top_k = top_k

        # 1. 统一从 embeddings 模块创建 Embedding
        self.embeddings = create_bge_embeddings(
            model_name=self.model_name,
            device=self.device,
        )

        # 2. 加载已经持久化的 Chroma 向量库
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        执行向量检索。

        参数：
        query:
            用户问题或改写后的检索问题。

        top_k:
            返回前几个结果。

        metadata_filter:
            metadata 过滤条件，例如：
            {"source": "travel_policy.md"}

        返回：
            Document 列表。
        """

        k = top_k or self.top_k

        # Chroma 返回的是 (Document, score)
        # 注意：Chroma score 一般越低越相关
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=metadata_filter,
        )

        documents = []

        for doc, score in results:
            # 把 Chroma 原始分数写入 metadata，方便后面调试和评估
            doc.metadata["score"] = score
            documents.append(doc)

        return documents

    def count(self) -> int:
        """
        返回当前 Chroma collection 中的 chunk 数量。
        """

        return self.vectorstore._collection.count()

    def as_langchain_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        返回 LangChain 标准 retriever。

        后续如果要接 LangChain 官方 Retrieval Chain，可以用这个。
        """

        if search_kwargs is None:
            search_kwargs = {"k": self.top_k}

        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )