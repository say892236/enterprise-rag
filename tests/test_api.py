# tests/test_api.py

"""
FastAPI 集成测试脚本。

运行前必须先启动 API 服务：

python run_api.py

然后另开一个终端运行：

python -m tests.test_api

测试内容：
1. /health 健康检查
2. /auth/login 正确登录
3. /auth/login 错误密码返回 LOGIN_FAILED
4. /chat 未登录返回 AUTH_REQUIRED
5. /chat 携带 Token 正常问答
6. /chat 访问无权限文档返回 PERMISSION_DENIED
"""

import os
import sys
from typing import Any

import requests


# 支持通过环境变量覆盖 API 地址
BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# 当前 Mock 用户表里 ay 的默认密码是 123456
TEST_USERNAME = "ay"
TEST_PASSWORD = "123456"


def print_pass(message: str) -> None:
    """
    打印测试通过信息。
    """

    print(f"[PASS] {message}")


def print_fail(message: str) -> None:
    """
    打印测试失败信息。
    """

    print(f"[FAIL] {message}")


def request_json(response: requests.Response) -> dict[str, Any]:
    """
    安全解析响应 JSON。

    如果接口返回的不是 JSON，直接抛出更清晰的错误。
    """

    try:
        return response.json()
    except Exception as exc:
        raise AssertionError(
            f"响应不是合法 JSON，status_code={response.status_code}, text={response.text}"
        ) from exc


def check_api_server() -> None:
    """
    检查 API 服务是否已经启动。

    如果没有启动，给出清晰提示，避免后面的测试全部连接失败。
    """

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=5,
        )
    except requests.exceptions.RequestException as exc:
        print_fail("API 服务未启动或无法访问")
        print()
        print("请先启动服务：")
        print()
        print("python run_api.py")
        print()
        print(f"当前 BASE_URL: {BASE_URL}")
        raise SystemExit(1) from exc

    if response.status_code != 200:
        print_fail(f"/health 状态异常，status_code={response.status_code}")
        raise SystemExit(1)


def test_health() -> None:
    """
    测试 /health 健康检查接口。
    """

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=5,
    )

    data = request_json(response)

    assert response.status_code == 200
    assert data["status"] == "ok"
    assert data["service"] == "enterprise-rag-api"

    print_pass("/health 健康检查正常")

def login_user(username: str, password: str = "123456") -> str:
    """
    通用登录函数。

    用于测试不同角色用户：
    - ay
    - hr_user
    - finance_user
    - it_user
    - admin

    返回：
        access_token
    """

    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={
            "username": username,
            "password": password,
        },
        timeout=10,
    )

    data = request_json(response)

    assert response.status_code == 200
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    return data["access_token"]

def test_login_success() -> str:
    """
    测试 /auth/login 正确登录。

    返回：
        access_token，供后续 /chat 测试使用。
    """

    access_token = login_user(
        username=TEST_USERNAME,
        password=TEST_PASSWORD,
    )

    print_pass("/auth/login 正确登录成功")

    return access_token

def test_login_failed() -> None:
    """
    测试 /auth/login 错误密码。

    预期：
    - HTTP 401
    - success = false
    - error_code = LOGIN_FAILED
    """

    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={
            "username": TEST_USERNAME,
            "password": "wrong_password",
        },
        timeout=10,
    )

    data = request_json(response)

    assert response.status_code == 401
    assert data["success"] is False
    assert data["error_code"] == "LOGIN_FAILED"
    assert "message" in data

    print_pass("/auth/login 错误密码返回 LOGIN_FAILED")


def test_chat_without_token() -> None:
    """
    测试未登录访问 /chat。

    预期：
    - HTTP 401
    - success = false
    - error_code = AUTH_REQUIRED
    """

    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "question": "员工出差住宿标准是什么？",
        },
        timeout=30,
    )

    data = request_json(response)

    assert response.status_code == 401
    assert data["success"] is False
    assert data["error_code"] == "AUTH_REQUIRED"
    assert "message" in data

    print_pass("/chat 未登录返回 AUTH_REQUIRED")


def test_chat_with_token(access_token: str) -> None:
    """
    测试携带 Token 正常调用 /chat。

    注意：
    /chat 成功响应现在是统一外层结构：

    {
        "success": true,
        "data": {...}
    }
    """

    response = requests.post(
        f"{BASE_URL}/chat",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        json={
            "question": "员工出差住宿标准是什么？",
            "retrieve_top_k": 6,
            "rerank_top_k": 3,
            "metadata_filter": {
                "source": "travel_policy.md",
            },
            "enable_rewrite": True,
            "enable_rerank": True,
        },
        timeout=120,
    )

    response_data = request_json(response)

    assert response.status_code == 200
    assert response_data["success"] is True
    assert "data" in response_data

    data = response_data["data"]

    assert data["username"] == TEST_USERNAME
    assert data["user_id"] == 10001
    assert data["user_role"] == "employee"

    assert data["question"] == "员工出差住宿标准是什么？"
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0

    assert "sources" in data
    assert isinstance(data["sources"], list)

    assert "request_id" in data
    assert isinstance(data["request_id"], str)
    assert len(data["request_id"]) > 0

    assert "cost_time_seconds" in data
    assert isinstance(data["cost_time_seconds"], float | int)

    print_pass("/chat 携带 Token 正常问答成功")

def assert_private_doc_access_success(
    access_token: str,
    question: str,
    source: str,
    expected_role: str,
) -> None:
    """
    断言某个用户可以访问指定私有文档。

    参数：
        access_token: 当前用户 token
        question: 测试问题
        source: 指定访问的文档 source
        expected_role: 预期用户角色
    """

    response = requests.post(
        f"{BASE_URL}/chat",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        json={
            "question": question,
            "retrieve_top_k": 6,
            "rerank_top_k": 3,
            "metadata_filter": {
                "source": source,
            },
            "enable_rewrite": True,
            "enable_rerank": True,
        },
        timeout=120,
    )

    response_data = request_json(response)

    assert response.status_code == 200
    assert response_data["success"] is True
    assert "data" in response_data

    data = response_data["data"]

    assert data["user_role"] == expected_role
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0

    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0

    # 因为请求里指定了 metadata_filter.source，
    # 所以返回来源中应该包含该 source。
    returned_sources = {
        item.get("source")
        for item in data["sources"]
    }

    assert source in returned_sources

def test_chat_permission_denied(access_token: str) -> None:
    """
    测试访问无权限文档。

    当前 ay 是 employee 角色。

    employee 只能访问 PUBLIC_SOURCES。

    这里故意请求一个不在 PUBLIC_SOURCES 里的 source，
    应该被权限控制层拦截，返回 PERMISSION_DENIED。
    """

    response = requests.post(
        f"{BASE_URL}/chat",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        json={
            "question": "薪资发放规则是什么？",
            "retrieve_top_k": 6,
            "rerank_top_k": 3,
            "metadata_filter": {
                "source": "hr_salary_policy.md",
            },
            "enable_rewrite": True,
            "enable_rerank": True,
        },
        timeout=60,
    )

    data = request_json(response)

    assert response.status_code == 403
    assert data["success"] is False
    assert data["error_code"] == "PERMISSION_DENIED"
    assert "message" in data

    print_pass("/chat 无权限文档访问返回 PERMISSION_DENIED")


def test_chat_validation_error(access_token: str) -> None:
    """
    测试 /chat 请求参数校验失败。

    这里故意传入非法参数：
    1. question 为空字符串
    2. retrieve_top_k 超过允许范围
    3. rerank_top_k 大于 retrieve_top_k

    预期：
    - HTTP 422
    - success = false
    - error_code = VALIDATION_ERROR
    """

    response = requests.post(
        f"{BASE_URL}/chat",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        json={
            "question": "   ",
            "retrieve_top_k": 30,
            "rerank_top_k": 10,
        },
        timeout=30,
    )

    data = request_json(response)

    assert response.status_code == 422
    assert data["success"] is False
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "message" in data
    assert "errors" in data

    print_pass("/chat 非法参数返回 VALIDATION_ERROR")

def test_hr_user_can_access_hr_private_doc() -> None:
    """
    测试 HR 用户可以访问 HR 私有文档。
    """

    access_token = login_user("hr_user")

    assert_private_doc_access_success(
        access_token=access_token,
        question="薪资调整流程是什么？",
        source="hr_salary_policy.md",
        expected_role="hr",
    )

    print_pass("hr_user 可以访问 HR 私有文档")


def test_finance_user_can_access_finance_private_doc() -> None:
    """
    测试财务用户可以访问财务私有文档。
    """

    access_token = login_user("finance_user")

    assert_private_doc_access_success(
        access_token=access_token,
        question="年度预算管理要求是什么？",
        source="finance_budget_policy.md",
        expected_role="finance",
    )

    print_pass("finance_user 可以访问财务私有文档")


def test_it_user_can_access_it_private_doc() -> None:
    """
    测试 IT 用户可以访问 IT 私有文档。
    """

    access_token = login_user("it_user")

    assert_private_doc_access_success(
        access_token=access_token,
        question="管理员账号管理要求是什么？",
        source="it_admin_policy.md",
        expected_role="it",
    )

    print_pass("it_user 可以访问 IT 私有文档")


def test_admin_can_access_all_private_docs() -> None:
    """
    测试 admin 可以访问所有部门私有文档。
    """

    access_token = login_user("admin")

    test_cases = [
        {
            "question": "薪资调整流程是什么？",
            "source": "hr_salary_policy.md",
        },
        {
            "question": "年度预算管理要求是什么？",
            "source": "finance_budget_policy.md",
        },
        {
            "question": "管理员账号管理要求是什么？",
            "source": "it_admin_policy.md",
        },
    ]

    for item in test_cases:
        assert_private_doc_access_success(
            access_token=access_token,
            question=item["question"],
            source=item["source"],
            expected_role="admin",
        )

    print_pass("admin 可以访问所有部门私有文档")

def test_chat_low_confidence_answer(access_token: str) -> None:
    """
    测试知识库无关问题触发低置信度拒答。

    问题“公司食堂几点开门？”不在当前知识库中，
    预期直接返回"根据现有知识库内容，未能查询到相关信息。请调整提问方式，或咨询相关行政/HR 人员。"，且 sources 为空。
    """

    response = requests.post(
        f"{BASE_URL}/chat",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
        json={
            "question": "公司食堂几点开门？",
            "retrieve_top_k": 6,
            "rerank_top_k": 3,
            "enable_rewrite": True,
            "enable_rerank": True,
        },
        timeout=120,
    )

    response_data = request_json(response)

    assert response.status_code == 200, (
        f"预期 status_code=200，实际 {response.status_code}，响应：{response_data}"
    )

    assert response_data["success"] is True, (
        f"预期 success=true，实际响应：{response_data}"
    )

    assert "data" in response_data, (
        f"响应中缺少 data 字段：{response_data}"
    )

    data = response_data["data"]

    assert data["answer"] == "根据现有知识库内容，未能查询到相关信息。请调整提问方式，或咨询相关行政/HR 人员。", (
        f"低置信度拒答答案不符合预期，实际 answer={data.get('answer')}，完整响应={data}"
    )

    assert data["sources"] == [], (
        f"低置信度拒答时 sources 应该为空，实际 sources={data.get('sources')}，完整响应={data}"
    )

    print_pass("/chat 知识库无关问题触发低置信度拒答")

def run_all_tests() -> None:
    """
    按顺序运行 API 集成测试。
    """

    print("=" * 100)
    print("开始运行 FastAPI 集成测试")
    print("BASE_URL:", BASE_URL)
    print("=" * 100)

    check_api_server()

    test_health()

    access_token = test_login_success()

    test_login_failed()
    test_chat_without_token()
    test_chat_validation_error(access_token)
    test_chat_with_token(access_token)

    # employee 访问 HR 私有文档，应该被拒绝
    test_chat_permission_denied(access_token)

    # 部门用户访问各自私有文档，应该成功
    test_hr_user_can_access_hr_private_doc()
    test_finance_user_can_access_finance_private_doc()
    test_it_user_can_access_it_private_doc()

    # admin 可以访问全部私有文档
    test_admin_can_access_all_private_docs()

    #低质信度
    test_chat_low_confidence_answer(access_token)

    print("=" * 100)
    print("FastAPI 集成测试全部通过")
    print("=" * 100)

if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as exc:
        print_fail(str(exc))
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print_fail(f"请求 API 失败: {exc}")
        sys.exit(1)