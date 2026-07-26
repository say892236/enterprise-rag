# api/dependencies.py

"""
FastAPI 依赖管理模块。

作用：
1. 初始化并缓存 RAG Chain。
2. 避免每次请求都重新加载 Embedding、BM25、Reranker。
3. 供路由层直接调用。
"""

from functools import lru_cache

from fastapi import Depends

from chains.rag_chain import EnterpriseRAGChain
from pipelines.rag_pipeline import build_hybrid_rag_chain
from services.chat_service import ChatService

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

def get_chat_service(
    rag_chain: EnterpriseRAGChain = Depends(get_rag_chain),
) -> ChatService:
    """
    获取 ChatService 实例。

    注意：
    - ChatService 本身很轻，只是业务编排层。
    - 真正重的 RAG Chain 已经由 get_rag_chain() 做了 lru_cache 缓存。
    """

    return ChatService(
        rag_chain=rag_chain,
    )
