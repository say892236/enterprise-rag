# api/auth/jwt.py

"""
JWT 工具模块。

负责：
1. 创建 JWT Token
2. 解析 JWT Token
3. 验证 Token 是否有效

配置来源：
config/settings.py
.env
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt

from config.settings import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    创建 JWT Token。

    参数：
        data:
            要写入 token 的业务数据，例如：
            {
                "sub": "ay",
                "user_id": 10001,
                "role": "employee"
            }

        expires_delta:
            自定义过期时间。
            如果不传，则使用 settings 中的默认过期时间。

    返回：
        JWT 字符串。
    """

    # 复制一份，避免修改原始 data
    to_encode = data.copy()

    # 计算 token 过期时间
    if expires_delta is None:
        expires_delta = timedelta(
            minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

    expire = datetime.now(timezone.utc) + expires_delta

    # exp 是 JWT 标准过期时间字段
    to_encode.update(
        {
            "exp": expire,
        }
    )

    # 生成 JWT
    encoded_jwt = jwt.encode(
        to_encode,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )

    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    解析 JWT Token。

    参数：
        token:
            前端传来的 JWT 字符串。

    返回：
        成功：payload 字典
        失败：None
    """

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )

        return payload

    except JWTError:
        return None