# api/main.py

from contextlib import asynccontextmanager
import asyncio
import logging

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routes.chat import router as chat_router

from api.core.logging_config import setup_logging
from api.core.request_logging import RequestLoggingMiddleware
from api.core.exceptions import (
    http_exception_handler,
    validation_exception_handler,
)
from config.settings import (
    CORS_ALLOW_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS,
)


# 先初始化日志
setup_logging()

logger = logging.getLogger(__name__)




# 日志初始化之后，再导入业务模块
from api.dependencies import get_rag_chain
from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router


load_dotenv()

def get_env_list(name: str, default: str = "") -> list[str]:
    """
    从环境变量中读取逗号分隔的列表配置。

    例如：
    CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173

    会被解析成：
    [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    """

    raw_value = os.getenv(name, default)

    return [
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    ]


def get_env_bool(name: str, default: bool = False) -> bool:
    """
    从环境变量中读取布尔值配置。
    """

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理。

    启动时预加载 RAG Chain，避免第一次请求过慢。

    注意：
    Docker Compose 启动时，即使 Milvus 容器已经启动，
    Milvus Proxy 也可能还没完全 ready。
    所以这里增加重试，避免 API 因 Milvus 暂时不可用而启动失败。
    """

    max_retry_count = 20
    retry_interval_seconds = 5

    for attempt in range(1, max_retry_count + 1):
        try:
            logger.info(
                "正在预加载 RAG Chain... attempt=%s/%s",
                attempt,
                max_retry_count,
            )

            get_rag_chain()

            logger.info("RAG Chain 预加载完成")
            break

        except Exception as e:
            logger.warning(
                "RAG Chain 预加载失败，等待后重试 | attempt=%s/%s | error=%s",
                attempt,
                max_retry_count,
                str(e),
            )

            if attempt >= max_retry_count:
                logger.exception("RAG Chain 预加载最终失败，API 启动中止")
                raise

            await asyncio.sleep(retry_interval_seconds)

    yield

    logger.info("FastAPI 服务关闭")

tags_metadata = [
    {
        "name": "Health",
        "description": "服务健康检查接口，用于确认 API 服务是否正常运行。",
    },
    {
        "name": "Auth",
        "description": "用户登录认证接口。登录成功后返回 JWT Token，调用 /chat 时需要携带 Bearer Token。",
    },
    {
        "name": "Chat",
        "description": "企业知识库问答接口，支持 JWT 认证、角色权限控制、Hybrid Search、Reranker 和低置信度拒答。",
    },
]


app = FastAPI(
    title="Enterprise RAG API",
    description=(
        "企业级 RAG 知识库问答系统 API。\n\n"
        "核心能力包括：\n\n"
        "- Markdown 企业制度文档问答\n"
        "- JWT 登录认证\n"
        "- 角色权限控制\n"
        "- Query Rewrite\n"
        "- Hybrid Search(Dense + BM25)\n"
        "- BGE Reranker\n"
        "- 低置信度拒答\n"
        "- 统一成功 / 错误响应\n"
    ),
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)





# 注册全局异常处理器
app.add_exception_handler(
    StarletteHTTPException,
    http_exception_handler,
)

app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler,
)


# 配置 CORS
# 作用：允许后续前端页面从浏览器访问后端 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173","http://127.0.0.1:5173",
    ],
    allow_credentials=get_env_bool("CORS_ALLOW_CREDENTIALS", True),
    allow_methods=get_env_list("CORS_ALLOW_METHODS", "*"),
    allow_headers=get_env_list("CORS_ALLOW_HEADERS", "*"),
)

# 注册请求日志中间件
app.add_middleware(RequestLoggingMiddleware)

@app.get(
    "/health",
    tags=["Health"],
    summary="健康检查",
    description="用于检查 FastAPI 服务是否正常启动。",
)
def health_check():
    """
    健康检查接口。
    """

    return {
        "status": "ok",
        "service": "enterprise-rag-api",
    }
@app.get(
    "/readiness",
    tags=["Health"],
    summary="就绪检查",
    description="用于检查 RAG Chain、Retriever、向量库等核心依赖是否已准备好。",
)
def readiness_check():
    """
    就绪检查接口。

    和 /health 的区别：
    - /health 只说明 FastAPI 服务还活着。
    - /readiness 会进一步检查 RAG Chain 是否已加载、Retriever 是否存在、向量库是否可访问。

    适合用于：
    1. Docker 部署后的服务可用性判断。
    2. 后续 Nginx / K8s / 自动化测试的 readiness probe。
    """

    try:
        # get_rag_chain() 已经由 lru_cache 缓存。
        # 如果 lifespan 阶段已经预加载成功，这里不会重复加载大模型。
        rag_chain = get_rag_chain()

        retriever = getattr(
            rag_chain,
            "retriever",
            None,
        )

        # HybridRetriever 里面一般有 dense_retriever。
        # 如果不是 HybridRetriever，就退回使用 retriever 本身。
        dense_retriever = getattr(
            retriever,
            "dense_retriever",
            retriever,
        )

        vector_count = None
        retrieval_smoke_test = "skipped"
        retrieval_smoke_docs = 0

        # 1. 尝试读取向量数量。
        # 注意：不同 Milvus / LangChain 封装下 count() 不一定稳定，
        # 所以 count 为 None 不代表服务不可用。
        if dense_retriever is not None and hasattr(dense_retriever, "count"):
            vector_count = dense_retriever.count()

        # 2. 做一次轻量检索，用来确认向量库是否真的可访问。
        # 这里只取 top_k=1，避免 readiness 检查成本过高。
        if dense_retriever is not None and hasattr(dense_retriever, "retrieve"):
            smoke_docs = dense_retriever.retrieve(
                query="员工出差住宿标准",
                top_k=1,
                metadata_filter=None,
            )

            retrieval_smoke_docs = len(smoke_docs)
            retrieval_smoke_test = "ready" if retrieval_smoke_docs > 0 else "empty"

        return {
            "status": "ready",
            "service": "enterprise-rag-api",
            "checks": {
                "api": "ready",
                "rag_chain": "ready",
                "retriever": "ready" if retriever is not None else "missing",
                "dense_retriever": "ready" if dense_retriever is not None else "missing",
                "vector_count": vector_count,
                "vector_count_status": "unknown" if vector_count is None else "ready",
                "retrieval_smoke_test": retrieval_smoke_test,
                "retrieval_smoke_docs": retrieval_smoke_docs,
            },
        }

    except Exception as e:
        logger.exception(
            "Readiness 检查失败 | error=%s",
            str(e),
        )

        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error_code": "SERVICE_NOT_READY",
                "message": "RAG 服务尚未准备好",
                "error": str(e),
            },
        )

app.include_router(auth_router)
app.include_router(chat_router)