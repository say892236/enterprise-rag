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


# 本地演示账号统一使用相同密码，方便测试不同角色权限
# 生产环境必须改为数据库存储，并为每个用户设置独立密码
DEFAULT_TEST_PASSWORD = "123456"
DEFAULT_PASSWORD_HASH = hash_password(DEFAULT_TEST_PASSWORD)


"""
测试用户数据。

注意：
1. 当前用户仅用于项目演示和本地权限测试。
2. role 必须与 access_control.py 中配置的角色名称完全一致。
3. 实际生产环境应改用数据库保存用户和密码哈希。
"""



# 测试用户数据库
#
# key 使用用户名，方便登录时快速查询。
# 密码不保存明文，而是在程序启动时生成哈希。
# 当前 USERS_DB 仅用于本地开发、接口演示和多角色权限测试。
# 生产环境应替换为 MySQL/PostgreSQL 等持久化用户系统。
USERS_DB = {
    # 普通员工：只能访问公共制度文档
    "say": {
        "user_id": 10001,
        "username": "say",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "role": "employee",
        "disabled": False,
    },

    # 人力资源部门
    "hr01": {
        "user_id": 10002,
        "username": "hr01",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "role": "hr",
        "disabled": False,
    },

    # 财务部门
    "finance01": {
        "user_id": 10003,
        "username": "finance01",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "role": "finance",
        "disabled": False,
    },

    # 研发部门
    "engineer01": {
        "user_id": 10004,
        "username": "engineer01",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "role": "engineering",
        "disabled": False,
    },

    # 行政部门
    #
    # administration 是行政部门角色，
    # 不要写成 admin。
    "administration01": {
        "user_id": 10005,
        "username": "administration01",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "role": "administration",
        "disabled": False,
    },

    # 法务部门
    "legal01": {
        "user_id": 10006,
        "username": "legal01",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "role": "legal",
        "disabled": False,
    },

    # 管理层
    "manager01": {
        "user_id": 10007,
        "username": "manager01",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "role": "management",
        "disabled": False,
    },

    # 系统管理员：可以访问全部文档和调试信息
    "admin": {
        "user_id": 10008,
        "username": "admin",
        "hashed_password": DEFAULT_PASSWORD_HASH,
        "role": "admin",
        "disabled": False,
    },
}

def get_user(username: str):
    """
    根据用户名查询用户。
    """

    return USERS_DB.get(username)