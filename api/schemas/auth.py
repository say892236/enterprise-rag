"""
认证相关的数据模型。

负责：
1. 登录请求参数
2. 登录成功返回格式
"""


from pydantic import BaseModel, Field,ConfigDict



class LoginRequest(BaseModel):
    """
    登录请求体。
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "ay",
                "password": "123456",
            }
        }
    )

    username: str = Field(
        description="用户名",
        examples=["ay"],
    )

    password: str = Field(
        description="密码",
        examples=["123456"],
    )


class TokenResponse(BaseModel):
    """
    登录成功返回的数据。

    前端拿到 access_token 后，
    后续请求带上：

    Authorization:
    Bearer token
    """

    access_token: str = Field(
        description="JWT访问令牌"
    )

    token_type: str = Field(
        default="bearer",
        description="Token类型",
    )