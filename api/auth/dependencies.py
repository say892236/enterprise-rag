# api/auth/dependencies.py

"""
认证依赖模块。

作用：
1. 从请求头 Authorization 中读取 Bearer Token。
2. 解析 JWT。
3. 获取当前用户信息。
4. 供 /chat 等需要登录的接口使用。
"""

from typing import Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth.jwt import decode_access_token
from api.auth.users import get_user


security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    获取当前登录用户。
    """

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "error_code": "AUTH_REQUIRED",
                "message": "未登录，请先传入 Authorization Bearer Token",
            },
        )

    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "error_code": "TOKEN_INVALID",
                "message": "Token 无效或已过期",
            },
        )

    username = payload.get("sub")

    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "error_code": "TOKEN_MISSING_SUB",
                "message": "Token 中缺少用户标识",
            },
        )

    user = get_user(username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "error_code": "USER_NOT_FOUND",
                "message": "用户不存在",
            },
        )

    return {
        "username": user["username"],
        "user_id": user["user_id"],
        "role": user["role"],
    }