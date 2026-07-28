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
from queue import Queue
from threading import Thread
from typing import Any, Dict, Iterator, List

from langchain_core.callbacks import BaseCallbackHandler

from api.services.redis_history import RedisChatHistory
from chains.rag_chain import EnterpriseRAGChain
from api.access_control import (
    PermissionDeniedError,
    build_effective_metadata_filter,
)
from api.schemas.chat import ChatRequest, ChatResponse, SourceItem


logger = logging.getLogger(__name__)



class _RAGStreamCallback(BaseCallbackHandler):
    """
    LangChain 大模型流式回调。

    作用：
    1. 大模型开始生成时，发送 generating 状态。
    2. 大模型每生成一段文本时，发送 token 事件。
    3. 通过 Queue 将后台线程中的事件交给 SSE 响应生成器。
    """

    def __init__(self, event_queue: Queue):
        self.event_queue = event_queue

        # 防止 on_chat_model_start 和 on_llm_start 重复发送状态
        self.generation_started = False

        # 用于判断模型是否真的返回了流式 token
        self.token_received = False

    def _emit_generating_once(self) -> None:
        """
        只发送一次“正在生成结果”状态。
        """

        if self.generation_started:
            return

        self.generation_started = True

        self.event_queue.put(
            {
                "event": "status",
                "data": {
                    "stage": "generating",
                    "message": "正在生成结果...",
                },
            }
        )

    def on_chat_model_start(
        self,
        serialized,
        messages,
        **kwargs,
    ) -> None:
        """
        Chat Model 开始生成。
        """

        self._emit_generating_once()

    def on_llm_start(
        self,
        serialized,
        prompts,
        **kwargs,
    ) -> None:
        """
        普通 LLM 开始生成。

        不同模型可能触发 on_chat_model_start 或 on_llm_start，
        因此两个入口都保留，并通过标记避免重复事件。
        """

        self._emit_generating_once()

    def on_llm_new_token(
        self,
        token: str,
        **kwargs,
    ) -> None:
        """
        大模型产生新的文本片段。
        """

        if not token:
            return

        self.token_received = True

        self.event_queue.put(
            {
                "event": "token",
                "data": {
                    "content": token,
                },
            }
        )


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
        """

        self.rag_chain = rag_chain

        # Redis 聊天历史服务
        self.redis_history = RedisChatHistory()

        # 保留原来的调用兼容
        self.history = self.redis_history



    @staticmethod
    def _model_to_dict(model: Any) -> Dict[str, Any]:
        """
        将 Pydantic 模型转换成可 JSON 序列化的字典。

        同时兼容 Pydantic v1 和 v2。
        """

        if hasattr(model, "model_dump"):
            return model.model_dump(mode="json")

        return model.dict()

    def _save_chat_history(
        self,
        chat_request: ChatRequest,
        current_user: Dict[str, Any],
        answer: str,
    ) -> None:
        """
        在完整回答生成结束后，一次性保存本轮对话。

        注意：
        不能在每个 token 到达时保存 Redis，
        否则一条回答会被拆成大量历史消息。
        """

        history_before = self.redis_history.get_session_messages(
            user_id=current_user["user_id"],
            session_id=chat_request.session_id,
        )

        # 添加本轮消息前没有历史，说明这是当前会话的第一次提问
        is_first_message = len(history_before) == 0

        self.history.add_message(
            user_id=current_user["user_id"],
            session_id=chat_request.session_id,
            role="user",
            content=chat_request.question,
        )

        self.history.add_message(
            user_id=current_user["user_id"],
            session_id=chat_request.session_id,
            role="assistant",
            content=answer,
        )

        # 第一次提问后，使用问题前 20 个字符更新会话标题
        if is_first_message:
            self.redis_history.update_session_title(
                user_id=current_user["user_id"],
                session_id=chat_request.session_id,
                title=chat_request.question[:20],
            )

    def _build_chat_response(
        self,
        result: Dict[str, Any],
        current_user: Dict[str, Any],
        request_id: str,
        effective_metadata_filter: Dict[str, Any] | None,
        cost_time: float,
    ) -> ChatResponse:
        """
        将 RAG Chain 结果组装成统一 ChatResponse。

        普通接口和 SSE 接口共用此方法，
        避免两套接口返回结构不一致。
        """

        sources = self.format_sources(
            result.get("sources", []),
        )

        # 只有管理员可以查看 RAG 调试数据
        can_view_debug = current_user["role"] == "admin"

        return ChatResponse(
            request_id=request_id,
            username=current_user["username"],
            user_id=current_user["user_id"],
            user_role=current_user["role"],
            question=result["question"],
            rewritten_question=(
                result.get("rewritten_question")
                if can_view_debug
                else None
            ),
            answer=result["answer"],
            sources=sources,
            metadata_filter=effective_metadata_filter,
            query_route=(
                result.get("query_route")
                if can_view_debug
                else None
            ),
            timings=(
                result.get("timings")
                if can_view_debug
                else None
            ),
            compression=(
                result.get("compression")
                if can_view_debug
                else None
            ),
            cost_time_seconds=cost_time,
        )

    def chat(
        self,
        chat_request: ChatRequest,
        current_user: Dict[str, Any],
        request_id: str,
    ) -> ChatResponse:
        """
        执行普通非流式企业知识库问答。
        """

        start_time = time.perf_counter()

        # 1. 根据当前用户角色构造权限过滤条件
        effective_metadata_filter = build_effective_metadata_filter(
            user_role=current_user["role"],
            request_metadata_filter=chat_request.metadata_filter,
        )

        logger.info(
            "RAG 请求开始 | request_id=%s | username=%s | "
            "user_role=%s | question=%s | metadata_filter=%s",
            request_id,
            current_user["username"],
            current_user["role"],
            chat_request.question,
            effective_metadata_filter,
        )

        # 2. 获取当前会话历史
        history_messages = self.history.get_messages(
            user_id=current_user["user_id"],
            session_id=chat_request.session_id,
        )

        # 3. 执行原有完整 RAG 链路
        result = self.rag_chain.invoke(
            question=chat_request.question,
            retrieve_top_k=chat_request.retrieve_top_k,
            rerank_top_k=chat_request.rerank_top_k,
            chat_history=history_messages,
            metadata_filter=effective_metadata_filter,
            enable_rewrite=chat_request.enable_rewrite,
            enable_rerank=chat_request.enable_rerank,
            enable_compression=chat_request.enable_compression,
        )

        # 4. 完整回答生成后保存 Redis
        self._save_chat_history(
            chat_request=chat_request,
            current_user=current_user,
            answer=result["answer"],
        )

        # 5. 计算业务耗时
        cost_time = round(
            time.perf_counter() - start_time,
            3,
        )

        # 6. 组装与原来一致的响应结构
        response = self._build_chat_response(
            result=result,
            current_user=current_user,
            request_id=request_id,
            effective_metadata_filter=effective_metadata_filter,
            cost_time=cost_time,
        )

        logger.info(
            "RAG 请求完成 | request_id=%s | cost_time=%ss",
            request_id,
            cost_time,
        )

        return response


    def stream_chat(
        self,
        chat_request: ChatRequest,
        current_user: Dict[str, Any],
        request_id: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        执行 SSE 流式企业知识库问答。

        事件顺序：

        1. status / retrieving
        2. status / generating
        3. token
        4. sources
        5. done

        发生异常时返回 error 事件。
        """

        start_time = time.perf_counter()

        # 后台 RAG 线程通过 Queue 向 SSE 生成器传递事件
        event_queue = Queue()

        # 用于通知主生成器：后台任务已经彻底结束
        end_marker = object()

        def run_rag_task() -> None:
            """
            在线程中执行同步 RAG 链路。

            原因：
            rag_chain.invoke() 是同步阻塞方法。
            如果直接在 SSE 生成器中执行，无法一边生成一边向前端发送 token。
            """

            try:
                # 1. 根据当前用户角色构造权限过滤条件
                effective_metadata_filter = build_effective_metadata_filter(
                    user_role=current_user["role"],
                    request_metadata_filter=chat_request.metadata_filter,
                )

                logger.info(
                    "RAG 流式请求开始 | request_id=%s | username=%s | "
                    "user_role=%s | question=%s | metadata_filter=%s",
                    request_id,
                    current_user["username"],
                    current_user["role"],
                    chat_request.question,
                    effective_metadata_filter,
                )

                # 2. 获取当前会话历史
                history_messages = self.history.get_messages(
                    user_id=current_user["user_id"],
                    session_id=chat_request.session_id,
                )

                # 3. 创建 LangChain 流式回调
                stream_callback = _RAGStreamCallback(
                    event_queue=event_queue,
                )

                # 4. 执行完整 RAG 链路
                #
                # 检索、重排、置信度判断等流程仍然走原来的 invoke。
                # 只有大模型生成阶段通过 callback 实时发送 token。
                result = self.rag_chain.invoke(
                    question=chat_request.question,
                    retrieve_top_k=chat_request.retrieve_top_k,
                    rerank_top_k=chat_request.rerank_top_k,
                    chat_history=history_messages,
                    metadata_filter=effective_metadata_filter,
                    enable_rewrite=chat_request.enable_rewrite,
                    enable_rerank=chat_request.enable_rerank,
                    enable_compression=chat_request.enable_compression,
                    callbacks=[stream_callback],
                )

                # 5. 兼容以下两种情况：
                #
                # - 低置信度直接拒答，没有调用 LLM
                # - 当前模型供应商没有触发 token callback
                #
                # 这种情况下至少保证前端仍然能收到完整答案。
                if not stream_callback.generation_started:
                    event_queue.put(
                        {
                            "event": "status",
                            "data": {
                                "stage": "generating",
                                "message": "正在生成结果...",
                            },
                        }
                    )

                if (
                    not stream_callback.token_received
                    and result.get("answer")
                ):
                    event_queue.put(
                        {
                            "event": "token",
                            "data": {
                                "content": result["answer"],
                            },
                        }
                    )

                # 6. 完整回答结束后，一次性保存 Redis
                self._save_chat_history(
                    chat_request=chat_request,
                    current_user=current_user,
                    answer=result["answer"],
                )

                cost_time = round(
                    time.perf_counter() - start_time,
                    3,
                )

                # 7. 构造与普通 /chat 完全一致的最终响应
                response = self._build_chat_response(
                    result=result,
                    current_user=current_user,
                    request_id=request_id,
                    effective_metadata_filter=effective_metadata_filter,
                    cost_time=cost_time,
                )

                # 8. 单独发送引用来源，前端可以提前渲染
                event_queue.put(
                    {
                        "event": "sources",
                        "data": {
                            "sources": [
                                self._model_to_dict(source)
                                for source in response.sources
                            ],
                        },
                    }
                )

                # 9. 最后发送完整响应
                #
                # 前端可以使用该数据校正最终答案、
                # request_id、耗时以及管理员调试信息。
                event_queue.put(
                    {
                        "event": "done",
                        "data": {
                            "success": True,
                            "data": self._model_to_dict(response),
                        },
                    }
                )

                logger.info(
                    "RAG 流式请求完成 | request_id=%s | cost_time=%ss",
                    request_id,
                    cost_time,
                )

            except PermissionDeniedError as exc:
                cost_time = round(
                    time.perf_counter() - start_time,
                    3,
                )

                logger.warning(
                    "RAG 流式权限不足 | request_id=%s | "
                    "cost_time=%ss | error=%s",
                    request_id,
                    cost_time,
                    str(exc),
                )

                event_queue.put(
                    {
                        "event": "error",
                        "data": {
                            "success": False,
                            "error_code": "PERMISSION_DENIED",
                            "message": str(exc),
                            "request_id": request_id,
                        },
                    }
                )

            except Exception as exc:
                cost_time = round(
                    time.perf_counter() - start_time,
                    3,
                )

                logger.exception(
                    "RAG 流式请求失败 | request_id=%s | "
                    "cost_time=%ss | error=%s",
                    request_id,
                    cost_time,
                    str(exc),
                )

                event_queue.put(
                    {
                        "event": "error",
                        "data": {
                            "success": False,
                            "error_code": "RAG_STREAM_FAILED",
                            "message": "RAG 流式问答执行失败",
                            "request_id": request_id,
                            "error": str(exc),
                        },
                    }
                )

            finally:
                # 无论成功还是失败，都通知 SSE 生成器结束连接
                event_queue.put(end_marker)

        # 先立即向前端发送检索状态
        yield {
            "event": "status",
            "data": {
                "stage": "retrieving",
                "message": "正在检索知识库...",
            },
        }

        # 启动后台线程执行同步 RAG
        worker_thread = Thread(
            target=run_rag_task,
            daemon=True,
            name=f"rag-stream-{request_id}",
        )
        worker_thread.start()

        # 持续读取后台线程产生的事件并交给 FastAPI
        while True:
            event = event_queue.get()

            if event is end_marker:
                break

            yield event





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


    def get_sessions(
        self,
        user_id: str,
    ):
        """
        获取用户聊天会话列表。
        """

        return self.redis_history.get_sessions(
            user_id=user_id,
        )


    def create_session(
        self,
        user_id: str,
        title: str = "新聊天",
    ):
        """
        创建新的聊天会话。
        """

        return self.redis_history.create_session(
            user_id=user_id,
            title=title,
        )


    def delete_session(
        self,
        user_id: str,
        session_id: str,
    ):
        """
        删除聊天会话。
        """

        return self.redis_history.delete_session(
            user_id=user_id,
            session_id=session_id,
        )


    def get_session_messages(
        self,
        user_id: str,
        session_id: str,
    ):
        """
        获取指定聊天记录。
        """

        return self.redis_history.get_session_messages(
            user_id=user_id,
            session_id=session_id,
        )



    def update_session_title(
            self,
            user_id: str,
            session_id: str,
            title: str,
    ):
        return self.redis_history.update_session_title(
            user_id=user_id,
            session_id=session_id,
            title=title,
        )