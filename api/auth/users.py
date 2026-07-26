# api/auth/users.py

"""
模拟用户数据库。

真实项目中，用户信息应该存储在数据库中。
当前为了学习 JWT 登录流程，先使用内存 Mock 用户表。

当前 Mock 用户：
- ay: 普通员工
- hr_user: HR
- finance_user: 财务
- it_user: IT
- admin: 管理员

所有测试账号密码均为：123456
"""

from passlib.context import CryptContext


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def hash_password(plain_password: str) -> str:
    """
    对明文密码进行 bcrypt 加密。
    """

    return pwd_context.hash(plain_password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    校验用户输入的明文密码是否正确。
    """

    return pwd_context.verify(
        plain_password,
        hashed_password,
    )


# 为了避免重复调用 hash_password，多生成一次公共 hash 即可
DEFAULT_PASSWORD_HASH = hash_password("123456")


fake_users_db = {
    "say": {
        "username": "say",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "user_id": 10001,
        "role": "employee",
    },

    "hr_user": {
        "username": "hr_user",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "user_id": 20001,
        "role": "hr",
    },

    "finance_user": {
        "username": "finance_user",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "user_id": 30001,
        "role": "finance",
    },

    "it_user": {
        "username": "it_user",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "user_id": 40001,
        "role": "it",
    },

    "admin": {
        "username": "admin",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "user_id": 1,
        "role": "admin",
    },
}


def get_user(username: str):
    """
    根据用户名查询用户。
    """

    return fake_users_db.get(username)