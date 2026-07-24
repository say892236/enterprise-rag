# examples/api_client_demo.py

"""
FastAPI 客户端调用示例。

作用：
1. 自动调用 /auth/login 登录。
2. 自动携带 Bearer Token 调用 /chat。
3. 打印 RAG 回答和引用来源。

运行前需要先启动 API 服务：

python run_api.py

然后另开终端运行：

python examples/api_client_demo.py
"""

import os
import requests


# 支持通过环境变量覆盖 API 地址
BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)


def login(username: str, password: str) -> str:
    """
    调用 /auth/login 登录接口。

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

    response.raise_for_status()

    data = response.json()

    return data["access_token"]


def ask_question(
    access_token: str,
    question: str,
):
    """
    调用 /chat 企业知识库问答接口。
    """

    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    response = requests.post(
        f"{BASE_URL}/chat",
        headers=headers,
        json={
            "question": question,
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

    response.raise_for_status()

    response_data = response.json()

    if not response_data.get("success"):
        raise RuntimeError(f"API 请求失败: {response_data}")

    return response_data["data"]

def print_sources(sources: list):
    """
    打印引用来源。
    """

    for index, source in enumerate(sources, start=1):
        print(f"\n【来源 {index}】")
        print("source:", source.get("source"))
        print("header_1:", source.get("header_1"))
        print("header_2:", source.get("header_2"))
        print("chunk_id:", source.get("chunk_id"))
        print("retrieval_sources:", source.get("retrieval_sources"))
        print("rerank_score:", source.get("rerank_score"))
        print("content_preview:", source.get("content_preview"))


def main():
    """
    API 调用示例主流程。
    """

    print("=" * 100)
    print("开始调用 Enterprise RAG API")
    print("BASE_URL:", BASE_URL)
    print("=" * 100)

    # 1. 登录，当前 Mock 用户密码均为 123456
    access_token = login(
        username="ay",
        password="123456",
    )

    print("登录成功，已获取 access_token")

    # 2. 调用 /chat
    result = ask_question(
        access_token=access_token,
        question="员工出差住宿标准是什么？",
    )

    print("=" * 100)
    print("请求信息")
    print("=" * 100)
    print("request_id:", result.get("request_id"))
    print("username:", result.get("username"))
    print("user_role:", result.get("user_role"))
    print("question:", result.get("question"))
    print("rewritten_question:", result.get("rewritten_question"))
    print("cost_time_seconds:", result.get("cost_time_seconds"))

    print("=" * 100)
    print("答案")
    print("=" * 100)
    print(result.get("answer"))

    print("=" * 100)
    print("引用来源")
    print("=" * 100)
    print_sources(result.get("sources", []))


if __name__ == "__main__":
    main()