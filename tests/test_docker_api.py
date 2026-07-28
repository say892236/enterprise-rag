# tests/test_docker_api.py

"""
Docker Compose 版 API 集成测试。

使用前提：
1. 已启动 Docker Compose 基础服务和 API：

   docker compose up -d etcd minio milvus redis api

2. API 服务可以访问：

   http://127.0.0.1:8000

3. Milvus 中已经写入当前知识库数据：

   python build_database.py

运行方式：

   python -m tests.test_docker_api

说明：
- 该测试会真实调用 API、Redis、Milvus、Embedding、Reranker 和 LLM。
- 测试速度比普通单元测试慢。
- 每个问答测试都会创建独立会话，完成后自动删除，避免历史记录互相干扰。
"""

import json
import os
import sys
from collections.abc import Iterator
from typing import Any, Dict

import requests

from api.access_control import DEPARTMENT_SOURCE_MAP


# Docker Compose API 默认访问地址
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

# 当前测试账号统一使用的演示密码
TEST_PASSWORD = "123456"

# 当前知识库中的公共制度文档
TRAVEL_SOURCE = "差旅住宿与交通标准.md"

COMPANY_PROFILE_SOURCE = "公司概况与组织信息.md"

# 从实际权限配置读取部门私有文档，
# 避免测试再次写死已经过期的文件名。
HR_PRIVATE_SOURCE = DEPARTMENT_SOURCE_MAP["hr"][0]
FINANCE_PRIVATE_SOURCE = DEPARTMENT_SOURCE_MAP["finance"][0]

# 统一低置信度拒答文本
NO_INFO_ANSWER = (
    "根据现有知识库内容，未能查询到相关信息。"
    "请调整提问方式，或咨询相关行政/HR 人员。"
)


def print_step(message: str) -> None:
    """打印当前测试步骤。"""

    print(f"\n[STEP] {message}")


def print_pass(message: str) -> None:
    """打印测试通过信息。"""

    print(f"[PASS] {message}")


def assert_true(
    condition: bool,
    message: str,
) -> None:
    """简单断言工具。"""

    if not condition:
        raise AssertionError(message)


def get_json(
    response: requests.Response,
) -> Dict[str, Any]:
    """
    将响应转换为 JSON。

    响应不是合法 JSON 时，给出清晰错误。
    """

    try:
        return response.json()
    except Exception as exc:
        raise AssertionError(
            "响应不是合法 JSON，"
            f"status_code={response.status_code}, "
            f"text={response.text}"
        ) from exc


def get_chat_data(
    response: requests.Response,
) -> Dict[str, Any]:
    """
    获取普通 /chat 响应中的业务数据。

    兼容：

    1. {"success": true, "data": {...}}
    2. 直接返回 {...}
    """

    payload = get_json(response)

    if isinstance(payload.get("data"), dict):
        return payload["data"]

    return payload


def get_error_data(
    response: requests.Response,
) -> Dict[str, Any]:
    """
    获取错误响应数据。

    同时兼容：

    1. {"error_code": "..."}
    2. {"detail": {"error_code": "..."}}
    """

    payload = get_json(response)
    detail = payload.get("detail")

    if isinstance(detail, dict):
        return detail

    return payload


def build_auth_headers(
    token: str,
) -> Dict[str, str]:
    """构造 JWT Authorization 请求头。"""

    return {
        "Authorization": f"Bearer {token}",
    }


def test_health() -> None:
    """测试健康检查接口。"""

    print_step("测试 /health")

    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=30,
    )

    assert_true(
        response.status_code == 200,
        (
            f"/health 状态码异常：{response.status_code}，"
            f"body={response.text}"
        ),
    )

    print_pass("/health 正常")


def test_readiness() -> None:
    """
    测试 /readiness 就绪检查。

    验证 Redis、RAG Chain、Retriever 和轻量检索。
    """

    print_step("测试 /readiness")

    response = requests.get(
        f"{API_BASE_URL}/readiness",
        timeout=60,
    )

    assert_true(
        response.status_code == 200,
        (
            f"/readiness 状态码异常：{response.status_code}，"
            f"body={response.text}"
        ),
    )

    data = get_json(response)
    checks = data.get("checks", {})

    assert_true(
        data.get("status") == "ready",
        f"/readiness status 异常：{data}",
    )

    assert_true(
        checks.get("redis") == "ready",
        f"Redis readiness 异常：{checks}",
    )

    assert_true(
        checks.get("rag_chain") == "ready",
        f"RAG Chain 未 ready：{checks}",
    )

    assert_true(
        checks.get("retriever") == "ready",
        f"Retriever 未 ready：{checks}",
    )

    assert_true(
        checks.get("dense_retriever") == "ready",
        f"Dense Retriever 未 ready：{checks}",
    )

    assert_true(
        checks.get("retrieval_smoke_test") == "ready",
        f"轻量检索未通过：{checks}",
    )

    assert_true(
        checks.get("retrieval_smoke_docs", 0) > 0,
        f"轻量检索没有返回文档：{checks}",
    )

    print_pass("/readiness 正常")


def login(
    username: str,
    password: str = TEST_PASSWORD,
) -> str:
    """
    登录并返回 access_token。

    兼容直接响应和统一响应两种格式。
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
        (
            f"用户 {username} 登录失败："
            f"status_code={response.status_code}，"
            f"body={response.text}"
        ),
    )

    data = get_json(response)

    if isinstance(data.get("data"), dict):
        token = data["data"].get("access_token")
    else:
        token = data.get("access_token")

    assert_true(
        bool(token),
        f"用户 {username} 的登录响应中没有 access_token：{data}",
    )

    return str(token)


def test_login() -> str:
    """测试普通员工登录。"""

    print_step("测试 /auth/login")

    token = login("say")

    print_pass("/auth/login 正常")

    return token


def create_session(
    token: str,
) -> str:
    """
    创建独立聊天会话并返回 session_id。
    """

    response = requests.post(
        f"{API_BASE_URL}/chat/session/new",
        headers=build_auth_headers(token),
        timeout=30,
    )

    assert_true(
        response.status_code == 200,
        (
            "创建会话失败："
            f"status_code={response.status_code}，"
            f"body={response.text}"
        ),
    )

    data = get_json(response)
    session_id = data.get("session_id")

    assert_true(
        bool(session_id),
        f"创建会话响应中没有 session_id：{data}",
    )

    return str(session_id)


def delete_session(
    token: str,
    session_id: str,
) -> None:
    """
    删除测试过程中创建的聊天会话。

    删除失败只打印提示，不掩盖原始测试结果。
    """

    try:
        response = requests.delete(
            f"{API_BASE_URL}/chat/session/{session_id}",
            headers=build_auth_headers(token),
            timeout=30,
        )

        if response.status_code != 200:
            print(
                "[WARN] 测试会话删除失败："
                f"session_id={session_id}，"
                f"status_code={response.status_code}"
            )
    except requests.RequestException as exc:
        print(
            "[WARN] 测试会话删除异常："
            f"session_id={session_id}，error={exc}"
        )


def chat(
    token: str,
    session_id: str,
    question: str,
    metadata_filter: Dict[str, Any] | None = None,
) -> requests.Response:
    """
    调用普通非流式 /chat 接口。
    """

    body: Dict[str, Any] = {
        "question": question,
        "session_id": session_id,
        "retrieve_top_k": 6,
        "rerank_top_k": 3,
        "enable_rewrite": True,
        "enable_rerank": True,
        "enable_compression": True,
        "metadata_filter": metadata_filter,
    }

    return requests.post(
        f"{API_BASE_URL}/chat",
        json=body,
        headers=build_auth_headers(token),
        timeout=180,
    )


def stream_chat(
    token: str,
    session_id: str,
    question: str,
    metadata_filter: Dict[str, Any] | None = None,
) -> requests.Response:
    """
    调用 SSE 流式 /chat/stream 接口。
    """

    body: Dict[str, Any] = {
        "question": question,
        "session_id": session_id,
        "retrieve_top_k": 6,
        "rerank_top_k": 3,
        "enable_rewrite": True,
        "enable_rerank": True,
        "enable_compression": True,
        "metadata_filter": metadata_filter,
    }

    headers = {
        **build_auth_headers(token),
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    return requests.post(
        f"{API_BASE_URL}/chat/stream",
        json=body,
        headers=headers,
        stream=True,
        timeout=(30, 180),
    )


def iter_sse_events(
    response: requests.Response,
) -> Iterator[Dict[str, Any]]:
    """
    解析 requests 返回的 SSE 事件流。

    输出格式：

    {
        "event": "token",
        "data": {"content": "员工"}
    }
    """

    response.encoding = "utf-8"

    event_name = "message"
    data_lines: list[str] = []

    for raw_line in response.iter_lines(
        decode_unicode=True,
    ):
        line = (raw_line or "").rstrip("\r")

        # 空行表示一个 SSE 事件结束
        if line == "":
            if data_lines:
                raw_data = "\n".join(data_lines)

                try:
                    event_data = json.loads(raw_data)
                except json.JSONDecodeError as exc:
                    raise AssertionError(
                        f"SSE data 不是合法 JSON：{raw_data}"
                    ) from exc

                yield {
                    "event": event_name,
                    "data": event_data,
                }

            event_name = "message"
            data_lines = []
            continue

        # SSE 注释行
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")

        if not separator:
            continue

        if value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    # 处理连接结束前没有空行的最后一个事件
    if data_lines:
        raw_data = "\n".join(data_lines)

        yield {
            "event": event_name,
            "data": json.loads(raw_data),
        }


def test_chat_answer(
    token: str,
) -> None:
    """
    测试正常知识库问答。
    """

    print_step("测试 /chat 正常问答")

    session_id = create_session(token)

    try:
        response = chat(
            token=token,
            session_id=session_id,
            question="员工出差住宿标准是什么？",
        )

        assert_true(
            response.status_code == 200,
            (
                f"/chat 状态码异常：{response.status_code}，"
                f"body={response.text}"
            ),
        )

        data = get_chat_data(response)

        answer = data.get("answer", "")
        sources = data.get("sources", [])

        assert_true(
            answer != NO_INFO_ANSWER and "住宿" in answer,
            f"回答内容不像住宿标准答案：{answer}",
        )

        assert_true(
            isinstance(sources, list) and len(sources) > 0,
            f"正常回答应返回引用来源，实际：{sources}",
        )

        assert_true(
            any(
                source.get("source") == TRAVEL_SOURCE
                for source in sources
            ),
            (
                f"住宿问题没有引用目标差旅文档："
                f"sources={sources}"
            ),
        )

        print_pass("/chat 正常问答通过")
        print(f"回答摘要：{answer[:120]}")

    finally:
        delete_session(token, session_id)


def test_stream_chat(
    token: str,
) -> None:
    """
    测试 SSE 真流式输出。

    验证：
    1. retrieving 状态。
    2. generating 状态。
    3. 多条 token。
    4. sources。
    5. done。
    6. token 拼接结果与最终答案一致。
    """

    print_step("测试 /chat/stream SSE 流式问答")

    session_id = create_session(token)
    response: requests.Response | None = None

    try:
        response = stream_chat(
            token=token,
            session_id=session_id,
            question="员工出差住宿标准是什么？",
        )

        assert_true(
            response.status_code == 200,
            (
                f"/chat/stream 状态码异常："
                f"{response.status_code}，body={response.text}"
            ),
        )

        content_type = response.headers.get(
            "content-type",
            "",
        )

        assert_true(
            "text/event-stream" in content_type,
            f"SSE Content-Type 异常：{content_type}",
        )

        events = list(iter_sse_events(response))
        event_names = [
            event["event"]
            for event in events
        ]

        assert_true(
            "status" in event_names,
            f"SSE 缺少 status 事件：{event_names}",
        )

        status_stages = [
            event["data"].get("stage")
            for event in events
            if event["event"] == "status"
        ]

        assert_true(
            "retrieving" in status_stages,
            f"SSE 缺少 retrieving 状态：{status_stages}",
        )

        assert_true(
            "generating" in status_stages,
            f"SSE 缺少 generating 状态：{status_stages}",
        )

        token_events = [
            event
            for event in events
            if event["event"] == "token"
        ]

        # 当前问题答案较长，应产生多条流式 token，
        # 防止接口退化为完整答案一次性返回。
        assert_true(
            len(token_events) >= 2,
            (
                "SSE token 事件数量过少，"
                f"实际数量：{len(token_events)}"
            ),
        )

        sources_events = [
            event
            for event in events
            if event["event"] == "sources"
        ]

        assert_true(
            len(sources_events) == 1,
            f"SSE sources 事件数量异常：{len(sources_events)}",
        )

        stream_sources = sources_events[0]["data"].get(
            "sources",
            [],
        )

        assert_true(
            len(stream_sources) > 0,
            "SSE 正常回答没有返回引用来源",
        )

        done_events = [
            event
            for event in events
            if event["event"] == "done"
        ]

        assert_true(
            len(done_events) == 1,
            f"SSE done 事件数量异常：{len(done_events)}",
        )

        done_data = done_events[0]["data"]

        assert_true(
            done_data.get("success") is True,
            f"SSE done 事件不是成功状态：{done_data}",
        )

        final_data = done_data.get("data", {})
        final_answer = final_data.get("answer", "")

        streamed_answer = "".join(
            event["data"].get("content", "")
            for event in token_events
        )

        assert_true(
            streamed_answer.strip() == final_answer.strip(),
            (
                "SSE token 拼接结果与 done 最终答案不一致："
                f"\nstreamed={streamed_answer}"
                f"\nfinal={final_answer}"
            ),
        )

        print_pass("/chat/stream SSE 流式问答通过")

    finally:
        if response is not None:
            response.close()

        delete_session(token, session_id)


def test_company_profile_answer(
    token: str,
) -> None:
    """
    测试公司基础信息问答。

    验证：
    1. 能回答公司名称。
    2. 不会触发低置信度拒答。
    3. 引用来源包含公司概况文档。
    """

    print_step("测试公司基础信息问答")

    session_id = create_session(token)

    try:
        response = chat(
            token=token,
            session_id=session_id,
            question="公司的全称、简称和英文名称分别是什么？",
        )

        assert_true(
            response.status_code == 200,
            (
                "公司基础信息问答状态码异常："
                f"{response.status_code}，body={response.text}"
            ),
        )

        data = get_chat_data(response)

        answer = data.get("answer", "")
        sources = data.get("sources", [])

        # 调试时可以直接看到实际结果
        print(f"公司信息回答：{answer}")
        print(
            "公司信息来源：",
            [
                source.get("source")
                for source in sources
            ],
        )

        # 先确认不是拒答
        assert_true(
            answer != NO_INFO_ANSWER,
            f"公司基础信息问题触发了拒答：answer={answer}",
        )

        # 再确认公司名称正确
        assert_true(
            "星澜智能科技有限公司" in answer,
            f"公司名称未正确命中：answer={answer}",
        )

        # 正常回答必须有来源
        assert_true(
            isinstance(sources, list) and len(sources) > 0,
            f"公司基础信息回答没有引用来源：sources={sources}",
        )

        # 最后检查是否引用目标文档
        actual_sources = [
            source.get("source")
            for source in sources
        ]

        assert_true(
            COMPANY_PROFILE_SOURCE in actual_sources,
            (
                "公司基础信息回答未引用目标文档："
                f"expected={COMPANY_PROFILE_SOURCE}，"
                f"actual={actual_sources}"
            ),
        )

        print_pass("公司基础信息问答通过")

    finally:
        delete_session(token, session_id)


def test_no_answer(
    token: str,
) -> None:
    """
    测试知识库无关问题的低置信度拒答。
    """

    print_step("测试低置信度拒答")

    session_id = create_session(token)

    try:
        response = chat(
            token=token,
            session_id=session_id,
            question="公司食堂几点开门？",
        )

        assert_true(
            response.status_code == 200,
            (
                "低置信度接口状态码异常："
                f"{response.status_code}，body={response.text}"
            ),
        )

        data = get_chat_data(response)

        answer = data.get("answer", "")
        sources = data.get("sources", [])

        assert_true(
            answer == NO_INFO_ANSWER,
            f"低置信度拒答答案异常：{answer}",
        )

        assert_true(
            sources == [],
            (
                "低置信度拒答时 sources 应为空，"
                f"实际：{sources}"
            ),
        )

        print_pass("低置信度拒答通过")

    finally:
        delete_session(token, session_id)


def test_employee_permission_denied(
    employee_token: str,
) -> None:
    """
    测试普通员工不能访问 HR 私有文档。
    """

    print_step("测试 employee 访问 HR 私有文档被拒绝")

    session_id = create_session(employee_token)

    try:
        response = chat(
            token=employee_token,
            session_id=session_id,
            question="请概括这份 HR 内部制度的主要内容。",
            metadata_filter={
                "source": HR_PRIVATE_SOURCE,
            },
        )

        assert_true(
            response.status_code == 403,
            (
                "普通员工访问 HR 私有文档应返回 403，"
                f"实际：{response.status_code}，"
                f"body={response.text}"
            ),
        )

        error_data = get_error_data(response)

        assert_true(
            error_data.get("error_code") == "PERMISSION_DENIED",
            f"权限错误码异常：{error_data}",
        )

        print_pass("employee 越权访问已被拒绝")

    finally:
        delete_session(employee_token, session_id)


def test_hr_private_access() -> str:
    """
    测试 HR 可以访问 HR 私有文档。

    返回 HR Token，供后续跨部门测试复用。
    """

    print_step("测试 hr 访问 HR 私有文档")

    hr_token = login("hr01")
    session_id = create_session(hr_token)

    try:
        policy_name = HR_PRIVATE_SOURCE.removesuffix(".md")

        response = chat(
            token=hr_token,
            session_id=session_id,
            question=f"请概括《{policy_name}》的主要规定。",
            metadata_filter={
                "source": HR_PRIVATE_SOURCE,
            },
        )

        assert_true(
            response.status_code == 200,
            (
                "HR 访问本部门文档失败："
                f"{response.status_code}，body={response.text}"
            ),
        )

        data = get_chat_data(response)
        answer = data.get("answer", "")
        sources = data.get("sources", [])

        assert_true(
            bool(answer) and answer != NO_INFO_ANSWER,
            f"HR 私有文档回答异常：{answer}",
        )

        assert_true(
            any(
                source.get("source") == HR_PRIVATE_SOURCE
                for source in sources
            ),
            (
                "HR 私有文档回答未引用目标文档："
                f"sources={sources}"
            ),
        )

        print_pass("hr 可以访问 HR 私有文档")

    finally:
        delete_session(hr_token, session_id)

    return hr_token


def test_hr_cross_department_denied(
    hr_token: str,
) -> None:
    """
    测试 HR 不能访问财务私有文档。
    """

    print_step("测试 hr 跨部门访问财务文档被拒绝")

    session_id = create_session(hr_token)

    try:
        response = chat(
            token=hr_token,
            session_id=session_id,
            question="请概括这份财务内部制度。",
            metadata_filter={
                "source": FINANCE_PRIVATE_SOURCE,
            },
        )

        assert_true(
            response.status_code == 403,
            (
                "HR 访问财务私有文档应返回 403，"
                f"实际：{response.status_code}，"
                f"body={response.text}"
            ),
        )

        error_data = get_error_data(response)

        assert_true(
            error_data.get("error_code") == "PERMISSION_DENIED",
            f"跨部门权限错误码异常：{error_data}",
        )

        print_pass("hr 跨部门访问已被拒绝")

    finally:
        delete_session(hr_token, session_id)


def test_admin_private_access() -> None:
    """
    测试 admin 可以访问财务私有文档。
    """

    print_step("测试 admin 访问部门私有文档")

    admin_token = login("admin")
    session_id = create_session(admin_token)

    try:
        policy_name = FINANCE_PRIVATE_SOURCE.removesuffix(".md")

        response = chat(
            token=admin_token,
            session_id=session_id,
            question=f"请概括《{policy_name}》的主要规定。",
            metadata_filter={
                "source": FINANCE_PRIVATE_SOURCE,
            },
        )

        assert_true(
            response.status_code == 200,
            (
                "admin 访问财务私有文档失败："
                f"{response.status_code}，body={response.text}"
            ),
        )

        data = get_chat_data(response)
        answer = data.get("answer", "")
        sources = data.get("sources", [])

        assert_true(
            bool(answer) and answer != NO_INFO_ANSWER,
            f"admin 私有文档回答异常：{answer}",
        )

        assert_true(
            any(
                source.get("source") == FINANCE_PRIVATE_SOURCE
                for source in sources
            ),
            (
                "admin 回答未引用目标私有文档："
                f"sources={sources}"
            ),
        )

        print_pass("admin 可以访问部门私有文档")

    finally:
        delete_session(admin_token, session_id)


def run_all_tests() -> None:
    """
    运行 Docker Compose API 集成测试。
    """

    print("=" * 100)
    print("开始运行 Docker Compose API 集成测试")
    print(f"API_BASE_URL: {API_BASE_URL}")
    print("=" * 100)

    # 基础服务
    test_health()
    test_readiness()

    # 普通员工
    employee_token = test_login()
    test_chat_answer(employee_token)
    test_stream_chat(employee_token)
    test_company_profile_answer(employee_token)
    test_no_answer(employee_token)
    test_employee_permission_denied(employee_token)

    # 部门角色
    hr_token = test_hr_private_access()
    test_hr_cross_department_denied(hr_token)

    # 管理员
    test_admin_private_access()

    print("\n" + "=" * 100)
    print("Docker Compose API 集成测试全部通过")
    print("=" * 100)


if __name__ == "__main__":
    try:
        run_all_tests()

    except AssertionError as exc:
        print(f"\n[FAIL] AssertionError: {exc}")
        sys.exit(1)

    except requests.RequestException as exc:
        print(f"\n[FAIL] 请求 API 失败：{exc}")
        sys.exit(1)

    except Exception as exc:
        print(
            f"\n[FAIL] {type(exc).__name__}: {exc}"
        )
        raise