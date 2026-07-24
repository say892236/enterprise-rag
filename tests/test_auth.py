# tests/test_auth.py

"""
认证模块基础测试。

测试目标：
1. Mock 用户表能正常查询用户。
2. 密码校验逻辑正常。
3. JWT 能正常生成和解析。
4. Token 中的用户身份能映射回用户表。
5. 过期 Token 会被识别为无效。

运行方式：
python -m tests.test_auth
"""

from datetime import timedelta

from api.auth.users import (
    get_user,
    verify_password,
)

from api.auth.jwt import (
    create_access_token,
    decode_access_token,
)


def assert_raises(expected_exception, func, *args, **kwargs):
    """
    简单异常断言工具。

    当前测试文件主要使用 assert，不引入 pytest。
    """

    try:
        func(*args, **kwargs)
    except expected_exception:
        return

    raise AssertionError(f"预期抛出异常: {expected_exception.__name__}")


def test_get_existing_user():
    """
    测试可以从 Mock 用户表中读取已存在用户。
    """

    user = get_user("ay")

    assert user is not None
    assert user["username"] == "ay"
    assert user["user_id"] == 10001
    assert user["role"] == "employee"

    print("[PASS] 可以读取已存在用户 ay")


def test_get_unknown_user():
    """
    测试不存在的用户返回 None。
    """

    user = get_user("not_exists")

    assert user is None

    print("[PASS] 不存在用户返回 None")


def test_verify_password_success():
    """
    测试正确密码可以校验通过。
    """

    user = get_user("ay")

    assert user is not None
    assert verify_password("123456", user["hashed_password"]) is True

    print("[PASS] 正确密码校验通过")


def test_verify_password_failed():
    """
    测试错误密码校验失败。
    """

    user = get_user("ay")

    assert user is not None
    assert verify_password("wrong_password", user["hashed_password"]) is False

    print("[PASS] 错误密码校验失败")


def test_create_and_decode_token():
    """
    测试 JWT 可以正常生成和解析。
    """

    token_data = {
        "sub": "ay",
        "user_id": 10001,
        "role": "employee",
    }

    token = create_access_token(data=token_data)
    payload = decode_access_token(token)

    assert payload is not None
    assert payload["sub"] == "ay"
    assert payload["user_id"] == 10001
    assert payload["role"] == "employee"
    assert "exp" in payload

    print("[PASS] JWT 生成和解析正常")


def test_token_user_mapping():
    """
    测试 Token 中的 sub 可以映射回用户表。
    """

    token = create_access_token(
        data={
            "sub": "admin",
            "user_id": 1,
            "role": "admin",
        }
    )

    payload = decode_access_token(token)

    assert payload is not None

    username = payload["sub"]
    user = get_user(username)

    assert user is not None
    assert user["username"] == "admin"
    assert user["role"] == "admin"

    print("[PASS] Token 用户身份可以映射回用户表")


def test_expired_token():
    """
    测试过期 Token 会解析失败。

    expires_delta 设置成 -1 秒，表示生成时已经过期。
    """

    token = create_access_token(
        data={
            "sub": "ay",
            "user_id": 10001,
            "role": "employee",
        },
        expires_delta=timedelta(seconds=-1),
    )

    payload = decode_access_token(token)

    assert payload is None

    print("[PASS] 过期 Token 会被识别为无效")


def main():
    """
    统一执行所有认证测试。
    """

    print("=" * 100)
    print("开始执行认证模块测试")
    print("=" * 100)

    test_get_existing_user()
    test_get_unknown_user()
    test_verify_password_success()
    test_verify_password_failed()
    test_create_and_decode_token()
    test_token_user_mapping()
    test_expired_token()

    print("=" * 100)
    print("全部认证模块测试通过")
    print("=" * 100)


if __name__ == "__main__":
    main()