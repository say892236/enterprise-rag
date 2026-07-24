# 使用 Python 3.11 官方精简镜像
FROM python:3.11-slim

# 设置容器内工作目录
WORKDIR /app

# 避免 Python 生成 __pycache__ 和 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1

# 让 Python 日志实时输出，方便 docker logs 查看
ENV PYTHONUNBUFFERED=1

# 先复制依赖文件，利用 Docker 缓存加速后续构建
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码到容器
COPY . .

# FastAPI 默认端口
EXPOSE 8000

# 启动 FastAPI 服务
# 注意：这里不用 --reload，容器部署环境不建议开启热重载
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]