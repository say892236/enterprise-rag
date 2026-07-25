# retriever/milvus_retriever.py

"""
Milvus Dense Retriever 封装模块。

作用：
1. 从 Milvus 向量库执行语义检索。
2. 对齐 ChromaRetriever 的 retrieve(query, top_k, metadata_filter) 接口。
3. 支持 metadata_filter 中的 source 过滤。
4. 将 Milvus score 写入 Document.metadata，方便后续 Hybrid / Reranker / API 返回。
"""
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from embeddings.bge_embedding import create_bge_embeddings
from vectorstore.milvus_store import (
    get_milvus_collection_count,
    load_milvus_vectorstore,
)

logger = logging.getLogger(__name__)

# 是否开启 RAG 详细调试日志
RAG_DEBUG = os.getenv("RAG_DEBUG", "false").lower() == "true"

class MilvusRetriever:
    """
    Milvus 向量检索器。

    注意：
    - Milvus 检索分数和 Chroma 分数含义可能不同。
    - 当前项目后续会用 RRF 做融合，所以不直接横向比较 dense 分数。
    """

    def __init__(self):
        """
        初始化 Milvus Retriever。

        流程：
        1. 创建 bge-m3 Embedding。
        2. 加载已有 Milvus VectorStore。
        """

        self.embeddings = create_bge_embeddings()
        self.vectorstore = load_milvus_vectorstore(
            embeddings=self.embeddings,
        )

    def _build_source_expr(
        self,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        将项目中的 metadata_filter 转换为 Milvus expr 表达式。

        当前主要支持：
        1. {"source": "travel_policy.md"}
           -> source == "travel_policy.md"

        2. {"source": {"$in": ["a.md", "b.md"]}}
           -> source in ["a.md", "b.md"]

        其他复杂过滤条件暂不支持，直接忽略。
        权限控制层已经在 API 层提前处理，这里只负责向量库过滤。
        """

        if not metadata_filter:
            return None

        source_filter = metadata_filter.get("source")

        if not source_filter:
            return None

        # source 精确匹配
        if isinstance(source_filter, str):
            return f'source == "{source_filter}"'

        # source $in 匹配
        if isinstance(source_filter, dict):
            source_values = source_filter.get("$in")

            if isinstance(source_values, list) and source_values:
                quoted_values = ", ".join(
                    f'"{source}"'
                    for source in source_values
                )

                return f"source in [{quoted_values}]"

        return None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        执行 Milvus Dense 检索。

        参数：
            query: 用户问题或改写后的问题。
            top_k: 返回文档数量。
            metadata_filter: metadata 过滤条件。

        返回：
            Document 列表。
        """

        expr = self._build_source_expr(metadata_filter)

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            expr=expr,
        )

        documents = []

        for rank, item in enumerate(results, start=1):
            doc, score = item

            # 复制 metadata，避免直接污染底层对象
            metadata = dict(doc.metadata or {})

            metadata["score"] = score
            metadata["milvus_score"] = score
            metadata["dense_rank"] = rank
            metadata["retrieval_sources"] = ["dense"]

            documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        if RAG_DEBUG:
            logger.debug(
                "Milvus Retriever | returned_docs=%s",
                len(documents),
            )

            for index, doc in enumerate(documents[:3], start=1):
                logger.debug(
                    "Milvus Top %s | source=%s | header_2=%s | milvus_score=%s | content_preview=%s",
                    index,
                    doc.metadata.get("source"),
                    doc.metadata.get("header_2"),
                    doc.metadata.get("milvus_score"),
                    doc.page_content[:50].replace("\n", " "),
                )

        return documents

    def count(self) -> int | None:
        """
        返回 Milvus Collection 中的实体数量。

        如果当前版本无法读取，返回 None。
        """

        return get_milvus_collection_count(self.vectorstore)

    def as_langchain_retriever(
        self,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        返回 LangChain 原生 retriever。

        当前主项目主要使用 retrieve()，
        该方法留给后续扩展。
        """

        if search_kwargs is None:
            search_kwargs = {
                "k": 5,
            }

        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs,
        )