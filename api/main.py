# api/main.py

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理。
    启动时预加载 RAG Chain，避免第一次请求过慢。
    """

    logger.info("正在预加载 RAG Chain...")

    get_rag_chain()

    logger.info("RAG Chain 预加载完成")

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

# 注册请求日志中间件
app.add_middleware(RequestLoggingMiddleware)

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
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)


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

app.include_router(auth_router)
app.include_router(chat_router)