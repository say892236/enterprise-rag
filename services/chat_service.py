# services/chat_service.py

"""
ChatService 业务编排层。

作用：
1. 根据当前用户角色构造最终 metadata_filter。
2. 调用 EnterpriseRAGChain 执行完整 RAG 问答。
3. 格式化 sources。
4. 组装 ChatResponse。

注意：
- Route 层只负责 HTTP。
- Service 层负责业务流程。
- RAG Chain 层负责具体 RAG 能力。
"""

import logging
import time
from typing import Any, Dict, List

from chains.rag_chain import EnterpriseRAGChain
from api.access_control import build_effective_metadata_filter
from api.schemas.chat import ChatRequest, ChatResponse, SourceItem


logger = logging.getLogger(__name__)


class ChatService:
    """
    Chat 业务服务。
    """

    def __init__(
        self,
        rag_chain: EnterpriseRAGChain,
    ):
        """
        初始化 ChatService。

        rag_chain 由 FastAPI dependency 注入，
        实际对象来自 get_rag_chain() 的 lru_cache 缓存。
        """

        self.rag_chain = rag_chain

    def chat(
        self,
        chat_request: ChatRequest,
        current_user: Dict[str, Any],
        request_id: str,
    ) -> ChatResponse:
        """
        执行一次企业知识库问答。

        参数：
        - chat_request：/chat 请求体
        - current_user：JWT 解析出的当前用户
        - request_id：请求追踪 ID

        返回：
        - ChatResponse：API 响应体 data 部分
        """

        start_time = time.perf_counter()

        # 1. 根据当前登录用户角色，生成最终权限 filter
        effective_metadata_filter = build_effective_metadata_filter(
            user_role=current_user["role"],
            request_metadata_filter=chat_request.metadata_filter,
        )

        logger.info(
            "RAG 请求开始 | request_id=%s | username=%s | user_role=%s | question=%s | metadata_filter=%s",
            request_id,
            current_user["username"],
            current_user["role"],
            chat_request.question,
            effective_metadata_filter,
        )

        # 2. 执行完整 RAG 链路
        result = self.rag_chain.invoke(
            question=chat_request.question,
            retrieve_top_k=chat_request.retrieve_top_k,
            rerank_top_k=chat_request.rerank_top_k,
            metadata_filter=effective_metadata_filter,
            enable_rewrite=chat_request.enable_rewrite,
            enable_rerank=chat_request.enable_rerank,
            enable_compression=chat_request.enable_compression,
        )

        # 3. 计算接口业务耗时
        cost_time = round(
            time.perf_counter() - start_time,
            3,
        )

        # 4. 规范化 sources
        sources = self.format_sources(
            result.get("sources", []),
        )

        # 5. 组装标准响应
        # 只有 admin 可以查看 RAG 调试信息。
        # 普通员工、HR、IT、Finance 等业务角色只返回业务答案和引用来源。
        can_view_debug = current_user["role"] == "admin"

        response = ChatResponse(
            request_id=request_id,
            username=current_user["username"],
            user_id=current_user["user_id"],
            user_role=current_user["role"],
            question=result["question"],
            rewritten_question=result.get("rewritten_question") if can_view_debug else None,
            answer=result["answer"],
            sources=sources,
            metadata_filter=effective_metadata_filter,
            query_route=result.get("query_route") if can_view_debug else None,
            timings=result.get("timings") if can_view_debug else None,
            compression=result.get("compression") if can_view_debug else None,
            cost_time_seconds=cost_time,
        )

        logger.info(
            "RAG 请求完成 | request_id=%s | cost_time=%ss",
            request_id,
            cost_time,
        )

        return response

    @staticmethod
    def build_content_preview(
        source: Dict[str, Any],
        max_length: int = 120,
    ) -> str:
        """
        构造引用片段预览。
        """

        content = (
            source.get("content")
            or source.get("page_content")
            or source.get("text")
            or ""
        )

        content = str(content).strip().replace("\n", " ")

        if len(content) <= max_length:
            return content

        return content[:max_length] + "..."

    @classmethod
    def format_sources(
        cls,
        sources: List[Dict[str, Any]],
    ) -> List[SourceItem]:
        """
        将 RAG Chain 返回的 sources 转成 API 规范结构。
        """

        formatted_sources = []

        for source in sources:
            item = SourceItem(
                source=source.get("source"),
                file_path=source.get("file_path"),
                file_type=source.get("file_type"),
                header_1=source.get("header_1"),
                header_2=source.get("header_2"),
                chunk_id=source.get("chunk_id"),
                chunk_index=source.get("chunk_index"),
                page_content=source.get("page_content"),
                content=source.get("content"),
                content_preview=cls.build_content_preview(source),
                retrieval_sources=source.get("retrieval_sources"),
                score=source.get("score"),
                chroma_score=source.get("chroma_score") or source.get("score"),
                milvus_score=source.get("milvus_score") or source.get("score"),
                dense_rank=source.get("dense_rank"),
                dense_rrf_score=source.get("dense_rrf_score"),
                bm25_score=source.get("bm25_score"),
                bm25_rank=source.get("bm25_rank"),
                bm25_rrf_score=source.get("bm25_rrf_score"),
                hybrid_score=source.get("hybrid_score"),
                raw_rerank_score=source.get("raw_rerank_score"),
                rerank_boost=source.get("rerank_boost"),
                rerank_boost_reasons=source.get("rerank_boost_reasons"),
                rerank_score=source.get("rerank_score"),
            )

            formatted_sources.append(item)

        return formatted_sources