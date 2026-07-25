# api/core/request_logging.py

"""
请求日志中间件。

作用：
1. 为每个 HTTP 请求生成全局 request_id。
2. 记录请求方法、路径、状态码和耗时。
3. 将 request_id 写入响应头，方便前端和日志排查问题。
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    全局请求日志中间件。
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        # 每个请求生成一个唯一 request_id
        request_id = str(uuid.uuid4())

        # 挂到 request.state，后续 route / exception handler 都能读取
        request.state.request_id = request_id

        start_time = time.perf_counter()

        try:
            response = await call_next(request)

        except Exception:
            cost_time = round(
                time.perf_counter() - start_time,
                3,
            )

            logger.exception(
                "请求异常 | request_id=%s | method=%s | path=%s | cost_time=%ss",
                request_id,
                request.method,
                request.url.path,
                cost_time,
            )

            raise

        cost_time = round(
            time.perf_counter() - start_time,
            3,
        )

        logger.info(
            "请求完成 | request_id=%s | method=%s | path=%s | status_code=%s | cost_time=%ss",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            cost_time,
        )

        # 响应头返回 request_id
        response.headers["X-Request-ID"] = request_id

        return response