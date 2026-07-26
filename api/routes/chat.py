# api/routes/chat.py

"""
Chat 路由模块。

提供企业知识库问答接口：
POST /chat
"""
import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request


from api.dependencies import get_chat_service
from api.schemas.chat import (
    ChatRequest,
    ChatAPIResponse,
)

from api.auth.dependencies import get_current_user
from api.access_control import PermissionDeniedError
from services.chat_service import ChatService

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
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