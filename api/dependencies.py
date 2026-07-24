# api/dependencies.py

"""
FastAPI 依赖管理模块。

作用：
1. 初始化并缓存 RAG Chain。
2. 避免每次请求都重新加载 Embedding、BM25、Reranker。
3. 供路由层直接调用。
"""

from functools import lru_cache

from chains.rag_chain import EnterpriseRAGChain
from pipelines.rag_pipeline import build_hybrid_rag_chain


@lru_cache(maxsize=1)
def get_rag_chain() -> EnterpriseRAGChain:
    """
    获取全局唯一 RAG Chain 实例。

    第一次调用时：
    - 加载 Chroma Retriever
    - 加载 BM25 Retriever
    - 加载 Query Rewriter
    - 加载 BGE Reranker
    - 初始化 LLM

    后续请求：
    - 直接复用缓存对象
    """

    return build_hybrid_rag_chain()