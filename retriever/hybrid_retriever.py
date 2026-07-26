# retriever/hybrid_retriever.py

"""
Hybrid Retriever 模块。

作用：
1. 同时调用 Chroma Dense Retriever 和 BM25 Retriever。
2. 将两路检索结果合并去重。
3. 使用 RRF（Reciprocal Rank Fusion）按排名融合结果。
4. 返回统一的 LangChain Document 列表，方便后续接 Reranker 和 RAG Chain。

为什么不用直接相加分数？
- Chroma score：越低越相关。
- BM25 score：越高越相关。
- 两者分数尺度完全不同，不能直接相加。

所以这里使用“排名融合”：
- 排名第 1 的文档贡献最高。
- 排名第 2 的文档贡献略低。
- 同时被 Dense 和 BM25 找到的文档，会获得两边加分。
"""
import logging
import os
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from retriever.chroma_retriever import ChromaRetriever
from retriever.bm25_retriever import BM25Retriever
from retriever.milvus_retriever import MilvusRetriever


logger = logging.getLogger(__name__)

# 是否开启 RAG 详细调试日志
RAG_DEBUG = os.getenv("RAG_DEBUG", "false").lower() == "true"


class HybridRetriever:
    """
    企业级 RAG 混合检索器。

    当前版本：
    - Dense Retriever：Chroma + bge-m3
    - Sparse Retriever：BM25
    - 融合算法：RRF 排名融合
    """

    def __init__(
        self,
        dense_retriever: MilvusRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        final_top_k: int = 10,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        """
        初始化 HybridRetriever。

        参数：
        dense_retriever:
            Chroma 向量检索器，负责语义召回。

        bm25_retriever:
            BM25 关键词检索器，负责精确关键词召回。

        dense_top_k:
            从 Chroma 召回多少个候选。

        bm25_top_k:
            从 BM25 召回多少个候选。

        final_top_k:
            融合后最终返回多少个候选。

        dense_weight:
            Dense 检索结果的权重。

        bm25_weight:
            BM25 检索结果的权重。

        rrf_k:
            RRF 平滑系数。
            常见取值是 60。
            值越大，不同排名之间的分数差距越小。

     注意：
     这里不能在函数参数默认值里写 MilvusRetriever()。
     否则 Python 在导入模块时就会提前连接 Milvus。
        """

        # 如果外部传入了检索器，就使用外部传入的；
        # 如果没有传入，就在真正创建 HybridRetriever 实例时再初始化。
        # 注意：这样不会在 import hybrid_retriever.py 时提前连接 Milvus。
        self.dense_retriever = dense_retriever or MilvusRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()

        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def _get_doc_key(self, doc: Document) -> str:
        """
        生成文档去重 key。

        优先使用 chunk_id。
        因为 chunk_id 是每个切片的唯一标识。

        如果某些文档没有 chunk_id，则退化使用：
        source + header_2 + 正文前 80 个字符
        """

        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id:
            return chunk_id

        source = doc.metadata.get("source", "")
        header_2 = doc.metadata.get("header_2", "")
        content_preview = doc.page_content[:80]

        return f"{source}|{header_2}|{content_preview}"

    def _rrf_score(
        self,
        rank: int,
        weight: float,
    ) -> float:
        """
        计算 RRF 分数。

        rank:
            排名，从 1 开始。

        weight:
            当前检索通道的权重。

        公式：
            weight / (rrf_k + rank)

        例子：
            rank=1 时，分数大约是 1 / 61
            rank=2 时，分数大约是 1 / 62

        排名越靠前，贡献越大。
        """

        return weight / (self.rrf_k + rank)

    def _add_results_to_pool(
        self,
        pool: Dict[str, Document],
        docs: List[Document],
        source_type: str,
        weight: float,
    ) -> None:
        """
        将某一路检索结果加入融合池。

        pool:
            最终融合池，key 是 chunk_id，value 是 Document。

        docs:
            某一路检索器返回的 Document 列表。

        source_type:
            当前来源类型。
            可选：
            - "dense"
            - "bm25"

        weight:
            当前检索通道权重。
        """

        for rank, doc in enumerate(docs, start=1):
            doc_key = self._get_doc_key(doc)

            # 当前检索结果的 RRF 分数
            score = self._rrf_score(
                rank=rank,
                weight=weight,
            )

            # 如果这个 chunk 第一次出现，先放进 pool
            if doc_key not in pool:
                # 复制 metadata，避免直接污染原始 Document
                metadata = dict(doc.metadata)

                # 初始化 Hybrid 相关字段
                metadata["hybrid_score"] = 0.0
                metadata["dense_rank"] = None
                metadata["bm25_rank"] = None
                metadata["dense_rrf_score"] = 0.0
                metadata["bm25_rrf_score"] = 0.0
                metadata["retrieval_sources"] = []

                pool[doc_key] = Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )

            # 从 pool 中取出统一的 Document
            pooled_doc = pool[doc_key]

            # 累加总的 hybrid_score
            pooled_doc.metadata["hybrid_score"] += score

            # 记录这个 chunk 被哪些检索器召回
            if source_type not in pooled_doc.metadata["retrieval_sources"]:
                pooled_doc.metadata["retrieval_sources"].append(source_type)

            # 记录 Dense 检索信息
            if source_type == "dense":
                pooled_doc.metadata["dense_rank"] = rank
                pooled_doc.metadata["dense_rrf_score"] = score

                # Chroma 原始分数越低越相关
                pooled_doc.metadata["chroma_score"] = doc.metadata.get("score")

            # 记录 BM25 检索信息
            if source_type == "bm25":
                pooled_doc.metadata["bm25_rank"] = rank
                pooled_doc.metadata["bm25_rrf_score"] = score

                # BM25 原始分数越高越相关
                pooled_doc.metadata["bm25_score"] = doc.metadata.get("bm25_score")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        执行 Hybrid 检索。

        流程：
        1. 用 Chroma 做 dense semantic search。
        2. 用 BM25 做 sparse keyword search。
        3. 用 RRF 融合两路结果。
        4. 按 hybrid_score 从高到低排序。
        5. 返回 top_k 个 Document。

        参数：
        query:
            用户问题或 Query Rewrite 后的问题。

        top_k:
            最终返回数量。
            如果不传，则使用 self.final_top_k。

        metadata_filter:
            metadata 过滤条件。
            会同时传给 dense 和 bm25。
        """

        final_k = top_k or self.final_top_k

        # 1. Dense 语义召回
        dense_docs = self.dense_retriever.retrieve(
            query=query,
            top_k=self.dense_top_k,
            metadata_filter=metadata_filter,
        )

        # 2. BM25 关键词召回
        bm25_docs = self.bm25_retriever.retrieve(
            query=query,
            top_k=self.bm25_top_k,
            metadata_filter=metadata_filter,
        )

        # 3. 创建融合池
        # key: chunk_id
        # value: Document
        pool: Dict[str, Document] = {}

        # 4. 加入 Dense 结果
        self._add_results_to_pool(
            pool=pool,
            docs=dense_docs,
            source_type="dense",
            weight=self.dense_weight,
        )

        # 5. 加入 BM25 结果
        self._add_results_to_pool(
            pool=pool,
            docs=bm25_docs,
            source_type="bm25",
            weight=self.bm25_weight,
        )

        # 6. 从融合池取出所有 Document
        fused_docs = list(pool.values())

        # 7. 按 hybrid_score 从高到低排序
        fused_docs.sort(
            key=lambda doc: doc.metadata.get("hybrid_score", 0.0),
            reverse=True,
        )

        if RAG_DEBUG:
            logger.debug(
                "Hybrid Retriever | dense_docs=%s | bm25_docs=%s | fused_docs=%s | final_k=%s",
                len(dense_docs),
                len(bm25_docs),
                len(fused_docs),
                final_k,
            )

            for index, doc in enumerate(fused_docs[:5], start=1):
                logger.debug(
                    "Hybrid Top %s | source=%s | header_2=%s | hybrid_score=%s | retrieval_sources=%s",
                    index,
                    doc.metadata.get("source"),
                    doc.metadata.get("header_2"),
                    doc.metadata.get("hybrid_score"),
                    doc.metadata.get("retrieval_sources"),
                )
        # 8. 返回最终 top_k
        return fused_docs[:final_k]