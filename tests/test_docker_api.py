# tests/test_docker_api.py

"""
Docker Compose 版 API 集成测试。

使用前提：
1. 已启动 Docker Compose 服务：
   docker compose up -d etcd minio milvus api

2. API 服务已正常启动：
   http://127.0.0.1:8000

3. Milvus Server 中已经写入知识库数据：
   python build_database.py

运行方式：
python -m tests.test_docker_api
"""

import os
import sys
from typing import Any, Dict

import requests


# Docker Compose API 默认访问地址
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def print_step(message: str) -> None:
    """
    打印当前测试步骤。
    """

    print(f"\n[STEP] {message}")


def print_pass(message: str) -> None:
    """
    打印测试通过信息。
    """

    print(f"[PASS] {message}")


def assert_true(condition: bool, message: str) -> None:
    """
    简单断言工具。

    如果条件不成立，直接抛出 AssertionError。
    """

    if not condition:
        raise AssertionError(message)


def get_json(response: requests.Response) -> Dict[str, Any]:
    """
    将响应转换为 JSON。

    如果响应不是合法 JSON，抛出更清晰的错误。
    """

    try:
        return response.json()
    except Exception as exc:
        raise AssertionError(
            f"响应不是合法 JSON，status_code={response.status_code}, text={response.text}"
        ) from exc


def test_health() -> None:
    """
    测试健康检查接口。
    """

    print_step("测试 /health")

    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=30,
    )

    assert_true(
        response.status_code == 200,
        f"/health 状态码异常：{response.status_code}, body={response.text}",
    )

    print_pass("/health 正常")


def login(username: str, password: str) -> str:
    """
    登录并返回 access_token。

    兼容两种响应结构：
    1. {"access_token": "..."}
    2. {"success": true, "data": {"access_token": "..."}}
    """

    response = requests.post(
        f"{API_BASE_URL}/auth/login",
        json={
            "username": username,
            "password": password,
        },
        timeout=30,
    )

    assert_true(
        response.status_code == 200,
        f"登录失败：status_code={response.status_code}, body={response.text}",
    )

    data = get_json(response)

    # 兼容统一响应格式
    if "data" in data and isinstance(data["data"], dict):
        token = data["data"].get("access_token")
    else:
        token = data.get("access_token")

    assert_true(
        bool(token),
        f"登录响应中没有 access_token：{data}",
    )

    return token


def test_login() -> str:
    """
    测试普通员工登录。
    """

    print_step("测试 /auth/login")

    token = login(
        username="ay",
        password="123456",
    )

    print_pass("/auth/login 正常")

    return token


def chat(
    token: str,
    question: str,
    metadata_filter: Dict[str, Any] | None = None,
) -> requests.Response:
    """
    调用 /chat 接口。

    参数：
        token: 登录后拿到的 JWT token。
        question: 用户问题。
        metadata_filter: 可选 metadata 过滤条件。
    """

    body: Dict[str, Any] = {
        "question": question,
        "retrieve_top_k": 10,
        "rerank_top_k": 5,
    }

    if metadata_filter is not None:
        body["metadata_filter"] = metadata_filter

    response = requests.post(
        f"{API_BASE_URL}/chat",
        json=body,
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=120,
    )

    return response


def get_chat_data(response: requests.Response) -> Dict[str, Any]:
    """
    获取 /chat 的 data 部分。

    兼容：
    1. {"success": true, "data": {...}}
    2. {...}
    """

    data = get_json(response)

    if "data" in data and isinstance(data["data"], dict):
        return data["data"]

    return data


def test_chat_answer(token: str) -> None:
    """
    测试正常知识库问答。
    """

    print_step("测试 /chat 正常问答")

    response = chat(
        token=token,
        question="员工出差住宿标准是什么？",
    )

    assert_true(
        response.status_code == 200,
        f"/chat 状态码异常：{response.status_code}, body={response.text}",
    )

    data = get_chat_data(response)

    answer = data.get("answer", "")
    sources = data.get("sources", [])

    assert_true(
        "住宿" in answer or "600" in answer or "450" in answer,
        f"回答内容不像住宿标准答案：{answer}",
    )

    assert_true(
        isinstance(sources, list),
        f"sources 应该是列表，实际：{type(sources)}",
    )

    print_pass("/chat 正常问答通过")
    print(f"回答摘要：{answer[:120]}")


def test_no_answer(token: str) -> None:
    """
    测试知识库无关问题的低置信度拒答。
    """

    print_step("测试低置信度拒答")

    response = chat(
        token=token,
        question="公司食堂几点开门？",
    )

    assert_true(
        response.status_code == 200,
        f"低置信度拒答接口状态码异常：{response.status_code}, body={response.text}",
    )

    data = get_chat_data(response)

    answer = data.get("answer", "")
    sources = data.get("sources", [])

    assert_true(
        answer == "知识库中未找到相关信息。",
        f"低置信度拒答答案异常：{answer}",
    )

    assert_true(
        sources == [],
        f"低置信度拒答时 sources 应为空，实际：{sources}",
    )

    print_pass("低置信度拒答通过")


def test_permission_denied(token: str) -> None:
    """
    测试普通员工访问 HR 私有文档时应被拒绝。
    """

    print_step("测试权限控制")

    response = chat(
        token=token,
        question="HR 私有薪酬制度有哪些内容？",
        metadata_filter={
            "source": "hr_salary_policy.md",
        },
    )

    assert_true(
        response.status_code == 403,
        f"普通员工访问 HR 私有文档应返回 403，实际：{response.status_code}, body={response.text}",
    )

    data = get_json(response)

    assert_true(
        data.get("error_code") == "PERMISSION_DENIED",
        f"权限错误码异常：{data}",
    )

    print_pass("权限控制通过")


def run_all_tests() -> None:
    """
    运行 Docker Compose API 集成测试。
    """

    print("=" * 100)
    print("开始运行 Docker Compose API 集成测试")
    print(f"API_BASE_URL: {API_BASE_URL}")
    print("=" * 100)

    test_health()

    token = test_login()

    test_chat_answer(token)
    test_no_answer(token)
    test_permission_denied(token)

    print("\n" + "=" * 100)
    print("Docker Compose API 集成测试全部通过")
    print("=" * 100)


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"\n[FAIL] 请求 API 失败：{exc}")
        sys.exit(1)