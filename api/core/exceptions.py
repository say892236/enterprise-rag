# api/core/exceptions.py

"""
全局异常处理模块。

作用：
1. 统一 FastAPI HTTPException 的返回格式。
2. 统一请求参数校验错误的返回格式。
3. 避免 FastAPI 默认返回 detail 包裹。
"""

import logging

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    """
    处理 HTTPException。

    支持两种 detail：
    1. dict：从里面读取 error_code / message / request_id
    2. str：作为普通错误信息返回
    """

    detail = exc.detail

    if isinstance(detail, dict):
        error_code = detail.get("error_code", "HTTP_ERROR")
        message = detail.get("message", "请求失败")
        request_id = detail.get("request_id")
    else:
        error_code = "HTTP_ERROR"
        message = str(detail)
        request_id = None

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": error_code,
            "message": message,
            "request_id": request_id,
        },
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """
    处理请求参数校验错误。

    注意：
    Pydantic v2 的 exc.errors() 里可能包含 ValueError 对象，
    不能直接丢给 JSONResponse。
    所以这里必须先用 jsonable_encoder 转成可 JSON 序列化的数据。
    """

    errors = jsonable_encoder(exc.errors())

    logger.warning(
        "请求参数校验失败 | path=%s | errors=%s",
        request.url.path,
        errors,
    )

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "请求参数校验失败，请检查请求体格式和字段类型",
            "request_id": None,
            "errors": errors,
        },
    )