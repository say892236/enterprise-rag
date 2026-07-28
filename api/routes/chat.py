# api/routes/chat.py

"""
Chat 路由模块。

提供企业知识库问答接口：
POST /chat
"""
import json
import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from api.services.redis_history import RedisChatHistory


from api.dependencies import get_chat_service
from api.schemas.chat import (
    ChatRequest,
    ChatAPIResponse,
)

from api.auth.dependencies import get_current_user
from api.access_control import PermissionDeniedError
from services.chat_service import ChatService

logger = logging.getLogger(__name__)
redis_history = RedisChatHistory()


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


def encode_sse_event(
    event: str,
    data: Dict[str, Any],
) -> str:
    """
    将事件转换成标准 SSE 文本格式。

    每条事件必须以两个换行符结束，
    否则浏览器可能不会立即处理数据。
    """

    json_data = json.dumps(
        data,
        ensure_ascii=False,
        default=str,
    )

    return (
        f"event: {event}\n"
        f"data: {json_data}\n\n"
    )




@router.post(
    "",
    response_model=ChatAPIResponse,
    summary="企业知识库问答",
    description=(
        "基于企业知识库回答用户问题。\n\n"
        "接口能力：\n\n"
        "1. 自动校验 JWT Token。\n"
        "2. 根据当前用户角色进行文档权限过滤。\n"
        "3. 支持 Query Rewrite 查询改写。\n"
        "4. 支持 Hybrid Search(Dense + BM25)。\n"
        "5. 支持 BGE Reranker 重排。\n"
        "6. 支持低置信度拒答。\n"
        "7. 返回答案、引用来源、request_id 和耗时信息。\n"
    ),
)
def chat(
    request: Request,
    chat_request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),

    # 当前登录用户由 JWT Token 自动解析
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    企业知识库问答接口。

    注意：
    用户角色不再从请求体读取。
    当前用户身份来自 Authorization Bearer Token。
    """

    request_id = getattr(
        request.state,
        "request_id",
        None,
    ) or "unknown-request-id"

    start_time = time.perf_counter()

    try:
        # Service 层负责权限过滤、RAG 调用和响应组装
        response = chat_service.chat(
            chat_request=chat_request,
            current_user=current_user,
            request_id=request_id,
        )

        return ChatAPIResponse(
            success=True,
            data=response,
        )
    except PermissionDeniedError as e:
        cost_time = round(time.perf_counter() - start_time, 3)

        logger.warning(
            "RAG 权限不足 | request_id=%s | cost_time=%ss | error=%s",
            request_id,
            cost_time,
            str(e),
        )

        raise HTTPException(
            status_code=403,
            detail={
                "success": False,
                "error_code": "PERMISSION_DENIED",
                "message": str(e),
                "request_id": request_id,
            },
        )


    except Exception as e:
        cost_time = round(time.perf_counter() - start_time, 3)

        logger.exception(
            "RAG 请求失败 | request_id=%s | cost_time=%ss | error=%s",
            request_id,
            cost_time,
            str(e),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "RAG_EXECUTION_FAILED",
                "message": "RAG 问答执行失败",
                "request_id": request_id,
                "error": str(e),
            },
        )


@router.post(
    "/stream",
    summary="企业知识库流式问答",
    description=(
        "使用 SSE 流式返回企业知识库问答结果。\n\n"
        "事件顺序：\n\n"
        "1. status：正在检索知识库。\n"
        "2. status：正在生成结果。\n"
        "3. token：大模型增量文本。\n"
        "4. sources：引用来源。\n"
        "5. done：完整最终响应。\n"
        "6. error：执行失败。\n"
    ),
)
def stream_chat(
    request: Request,
    chat_request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """
    企业知识库 SSE 流式问答接口。

    JWT、用户角色、会话历史和权限过滤逻辑
    与普通 /chat 接口保持一致。
    """

    request_id = getattr(
        request.state,
        "request_id",
        None,
    ) or "unknown-request-id"

    # ChatService 返回结构化事件，
    # Route 层只负责将事件编码成 SSE 格式。
    service_events = chat_service.stream_chat(
        chat_request=chat_request,
        current_user=current_user,
        request_id=request_id,
    )

    def event_stream():
        """
        将 Service 事件转换为 SSE 字符串。
        """

        for item in service_events:
            yield encode_sse_event(
                event=item["event"],
                data=item["data"],
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # 禁止浏览器或代理缓存流式内容
            "Cache-Control": "no-cache",

            # 禁止 Nginx 缓冲，否则事件可能最后一次性返回
            "X-Accel-Buffering": "no",

            # 保持 SSE 长连接
            "Connection": "keep-alive",
        },
    )





@router.post(
    "/session/new",
    summary="创建新的聊天会话",
)
def create_new_session(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    创建新的聊天窗口。

    登录后进入新聊天页面时调用。

    返回:
    session_id
    """

    session_id = redis_history.create_session(
        user_id=current_user["user_id"],
        title="新聊天",
    )


    return {
        "success": True,
        "session_id": session_id,
    }


@router.get(
    "/sessions",
    summary="获取聊天历史列表",
)
def get_chat_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    获取当前用户历史聊天。

    用于左侧聊天列表。
    """
    sessions = chat_service.get_sessions(
        user_id=current_user["user_id"]
    )

    return {
        "success": True,
        "sessions": sessions,
    }


@router.get(
    "/session/{session_id}",
    summary="获取聊天记录",
)
def get_chat_session_messages(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取当前用户指定 Redis 会话中的历史消息。"""
    messages = redis_history.get_messages(
        user_id=current_user["user_id"],
        session_id=session_id,
    )

    return {
        "success": True,
        "messages": messages,
    }


@router.delete(
    "/session/{session_id}",
    summary="删除聊天记录",
)
def delete_chat_session(
    session_id:str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    删除指定聊天。
    """


    redis_history.delete_session(
        user_id=current_user["user_id"],
        session_id=session_id,
    )


    return {
        "success":True,
        "message":"聊天删除成功",
    }

