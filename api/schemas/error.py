# api/schemas/error.py

"""
错误响应模型。

用于统一 API 的错误返回结构。
"""

from typing import Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    统一错误响应结构。
    """

    success: bool = Field(
        default=False,
        description="请求是否成功",
    )

    error_code: str = Field(
        description="错误码，方便前端或调用方判断错误类型",
    )

    message: str = Field(
        description="错误提示信息",
    )

    request_id: Optional[str] = Field(
        default=None,
        description="请求 ID，用于日志追踪；部分认证错误可能没有",
    )