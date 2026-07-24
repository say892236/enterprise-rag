"""
认证接口。

提供：

POST /auth/login

用于：
用户名密码登录
生成 JWT Token
"""

import logging
from datetime import timedelta

from fastapi import APIRouter, HTTPException


from api.schemas.auth import (
    LoginRequest,
    TokenResponse,
)

from api.auth.users import (
    get_user,
    verify_password,
)

from api.auth.jwt import (
    create_access_token,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
)



@router.post(
    "/login",
    response_model=TokenResponse,
    summary="用户登录",
    description=(
        "使用用户名和密码登录系统。\n\n"
        "登录成功后返回 JWT access_token。\n\n"
        "调用 `/chat` 接口时，需要在请求头中携带：\n\n"
        "`Authorization: Bearer <access_token>`"
    ),
    responses={
        401: {
            "description": "用户名或密码错误",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error_code": "LOGIN_FAILED",
                        "message": "用户名或密码错误",
                        "request_id": None,
                    }
                }
            },
        }
    },
)
def login(
    request: LoginRequest,
):
    """
    用户登录接口。

    流程：

    1. 查询用户
    2. 校验密码
    3. 创建 JWT
    4. 返回 Token
    """


    # ==========================
    # 1. 查询用户
    # ==========================

    user = get_user(
        request.username
    )

    if not user:
        logger.warning(
            "登录失败 | username=%s | reason=用户不存在",
            request.username,
        )

        raise HTTPException(
            status_code=401,
            detail={
                "success": False,
                "error_code": "LOGIN_FAILED",
                "message": "用户名或密码错误",
            },
        )
    # ==========================
    # 2. 校验密码
    # ==========================

    password_correct = verify_password(
        request.password,
        user["hashed_password"],
    )


    if not password_correct:
        logger.warning(
            "登录失败 | username=%s | reason=密码错误",
            request.username,
        )


        raise HTTPException(
            status_code=401,
            detail={
                "success": False,
                "error_code": "LOGIN_FAILED",
                "message": "用户名或密码错误",
            },
        )


    # ==========================
    # 3. 创建JWT Payload
    # ==========================

    token_data = {

        # JWT标准字段
        "sub": user["username"],


        # 业务字段
        "user_id": user["user_id"],

        "role": user["role"],
    }



    access_token = create_access_token(
        data=token_data,

        # 这里设置token有效时间
        expires_delta=timedelta(
            minutes=60
        ),
    )

    logger.info(
        "登录成功 | username=%s | user_id=%s | role=%s",
        user["username"],
        user["user_id"],
        user["role"],
    )

    # ==========================
    # 4. 返回Token
    # ==========================

    return TokenResponse(

        access_token=access_token,

        token_type="bearer",
    )