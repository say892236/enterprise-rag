import json
import requests

API_BASE_URL = "http://127.0.0.1:8000"


def main():
    # 1. 登录
    login_response = requests.post(
        f"{API_BASE_URL}/auth/login",
        json={
            "username": "ay",
            "password": "123456",
        },
        timeout=30,
    )

    login_data = login_response.json()

    print("登录响应：")
    print(json.dumps(login_data, ensure_ascii=False, indent=2))

    # 兼容两种登录响应格式：
    # 1. {"access_token": "..."}
    # 2. {"success": true, "data": {"access_token": "..."}}
    if isinstance(login_data.get("data"), dict):
        token = login_data["data"].get("access_token")
    else:
        token = login_data.get("access_token")

    if not token:
        raise RuntimeError(f"登录响应中没有 access_token：{login_data}")

    # 2. 提问
    chat_response = requests.post(
        f"{API_BASE_URL}/chat",
        json={
            "question": "出差可以住五星级酒店吗？",
            "retrieve_top_k": 10,
            "rerank_top_k": 5,
        },
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=180,
    )

    data = chat_response.json()

    print("=" * 100)
    print("完整响应：")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print("=" * 100)

    chat_data = data.get("data", data)
    sources = chat_data.get("sources", [])

    print(f"sources 数量：{len(sources)}")

    if sources:
        first_source = sources[0]

        print("第一个 source 的字段：")
        print(first_source.keys())

        print("\n正文字段检查：")

        for key in ["page_content", "content", "content_preview"]:
            value = first_source.get(key)
            print(f"{key}: type={type(value)}, length={len(value) if isinstance(value, str) else 'N/A'}")
            print(f"{key} preview: {repr(value[:200]) if isinstance(value, str) else value}")


        print("\n第一个 source：")
        print(json.dumps(first_source, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()