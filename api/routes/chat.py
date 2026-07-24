# api/routes/chat.py

"""
Chat 路由模块。

提供企业知识库问答接口：
POST /chat
"""
import logging
import time
import uuid
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from chains.rag_chain import EnterpriseRAGChain
from api.dependencies import get_rag_chain
from api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatAPIResponse,
    SourceItem,
)

from api.auth.dependencies import get_current_user

from api.access_control import (
    build_effective_metadata_filter,
    PermissionDeniedError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


def build_content_preview(source: Dict[str, Any], max_length: int = 120) -> str:
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


def format_sources(sources: List[Dict[str, Any]]) -> List[SourceItem]:
    """
    将 RAG Chain 返回的 sources 转成 API 规范结构。
    """

    formatted_sources = []

    for source in sources:
        item = SourceItem(
            source=source.get("source"),
            header_1=source.get("header_1"),
            header_2=source.get("header_2"),
            chunk_id=source.get("chunk_id"),
            content_preview=build_content_preview(source),
            retrieval_sources=source.get("retrieval_sources"),
            chroma_score=source.get("chroma_score") or source.get("score"),
            milvus_score=source.get("milvus_score") or source.get("score"),
            bm25_score=source.get("bm25_score"),
            hybrid_score=source.get("hybrid_score"),
            rerank_score=source.get("rerank_score"),
        )

        formatted_sources.append(item)

    return formatted_sources


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
    responses={
        401: {
            "description": "未登录或 Token 无效",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error_code": "AUTH_REQUIRED",
                        "message": "未登录，请先传入 Authorization Bearer Token",
                        "request_id": None,
                    }
                }
            },
        },
        403: {
            "description": "当前角色无权限访问指定文档",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error_code": "PERMISSION_DENIED",
                        "message": "当前角色无权访问文档: hr_salary_policy.md",
                        "request_id": "example-request-id",
                    }
                }
            },
        },
        422: {
            "description": "请求参数校验失败",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error_code": "VALIDATION_ERROR",
                        "message": "请求参数校验失败，请检查请求体格式和字段类型",
                        "request_id": None,
                        "errors": [],
                    }
                }
            },
        },
        500: {
            "description": "RAG 执行失败",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error_code": "RAG_EXECUTION_FAILED",
                        "message": "RAG 执行失败，请稍后重试",
                        "request_id": "example-request-id",
                    }
                }
            },
        },
    },
)
def chat(
    request: ChatRequest,
    rag_chain: EnterpriseRAGChain = Depends(get_rag_chain),

    # 当前登录用户由 JWT Token 自动解析
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    企业知识库问答接口。

    注意：
    用户角色不再从请求体读取。
    当前用户身份来自 Authorization Bearer Token。
    """

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        # 1. 根据当前登录用户角色，生成最终权限 filter
        effective_metadata_filter = build_effective_metadata_filter(
            user_role=current_user["role"],
            request_metadata_filter=request.metadata_filter,
        )

        logger.info(
            "RAG 请求开始 | request_id=%s | username=%s | user_role=%s | question=%s | metadata_filter=%s",
            request_id,
            current_user["username"],
            current_user["role"],
            request.question,
            effective_metadata_filter,
        )


        # 2. 执行完整 RAG 链路
        result = rag_chain.invoke(
            question=request.question,
            retrieve_top_k=request.retrieve_top_k,
            rerank_top_k=request.rerank_top_k,
            metadata_filter=effective_metadata_filter,
            enable_rewrite=request.enable_rewrite,
            enable_rerank=request.enable_rerank,
        )

        # 3. 计算耗时
        cost_time = round(time.perf_counter() - start_time, 3)

        # 4. 规范化 sources
        sources = format_sources(result["sources"])

        # 5. 构造响应
        response = ChatResponse(
            request_id=request_id,
            username=current_user["username"],
            user_id=current_user["user_id"],
            user_role=current_user["role"],
            question=result["question"],
            rewritten_question=result.get("rewritten_question"),
            answer=result["answer"],
            sources=sources,
            metadata_filter=effective_metadata_filter,
            cost_time_seconds=cost_time,
        )

        logger.info(
            "RAG 请求完成 | request_id=%s | cost_time=%ss",
            request_id,
            cost_time,
        )

        return ChatAPIResponse(
            success=True,
            data=response
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