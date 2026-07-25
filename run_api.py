# run_api.py

"""
FastAPI 服务启动入口。

运行方式：
python run_api.py

说明：
1. API_HOST / API_PORT / API_RELOAD 从 config/settings.py 读取。
2. 开发阶段默认开启 reload。
3. 启动后访问 http://127.0.0.1:8000/docs 查看 Swagger 文档。
"""

import uvicorn

from config.settings import (
    API_HOST,
    API_PORT,
    API_RELOAD,
)

def main():
    """
    启动 FastAPI 服务。
    """

    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )

if __name__ == "__main__":
    main()