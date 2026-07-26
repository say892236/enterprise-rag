# Docker Compose 启动与自检说明

本文档用于说明 Enterprise-RAG 项目的 Docker Compose 启动、健康检查、就绪检查和集成测试流程。

## 1. 启动服务

在项目根目录执行：

`docker compose up -d`

查看容器状态：

`docker compose ps`

正常情况下，核心服务包括
- api
- milvus
- etcd
- minio
其中 api 服务应显示为 healthy。
---
## 2. 查看 API 日志
`docker compose logs -f api`

当看到类似输出时，说明 FastAPI 已启动完成：

`Application startup complete`

如果 RAG Chain 预加载阶段 Milvus 尚未准备好，API 会自动重试。

## 3. 存活检查：/health
 /health 用于检查 FastAPI 进程是否存活。

`Invoke-RestMethod http://127.0.0.1:8000/health`

预期返回：

```
{
  "status": "ok",
  "service": "enterprise-rag-api"
}
```
---
## 4. 就绪检查：/readiness
/readiness 用于检查 RAG 服务是否真正可用，包括：

- RAG Chain 是否已加载。
- Retriever 是否已初始化。
- Dense Retriever 是否已初始化。
- 向量库是否可以通过轻量检索访问。
`Invoke-RestMethod http://127.0.0.1:8000/readiness | ConvertTo-Json -Depth 10`
预期返回：
```
{
  "status": "ready",
  "service": "enterprise-rag-api",
  "checks": {
    "api": "ready",
    "rag_chain": "ready",
    "retriever": "ready",
    "dense_retriever": "ready",
    "vector_count": null,
    "vector_count_status": "unknown",
    "retrieval_smoke_test": "ready",
    "retrieval_smoke_docs": 1
  }
}
```

说明：
- vector_count 为 null 不一定代表异常。
- 只要 retrieval_smoke_test 为 ready，并且 retrieval_smoke_docs > 0，说明向量库可以正常检索。
---
## 5. 运行 Docker API 集成测试
`python -m tests.test_docker_api`

测试覆盖：
- /health
- /readiness
- /auth/login
- /chat
- 低置信度拒答
- 权限控制
全部通过时会输出：

`Docker Compose API 集成测试全部通过`
---
## 6. 常用维护命令
重启 API：
`docker compose restart api`

查看 API 日志：
`docker compose logs -f api`

查看所有容器状态：
`docker compose ps`

关闭服务：
`docker compose down`
---
## 7. 常见问题
### 7.1 Milvus 尚未 ready
现象：

`Milvus Proxy is not ready yet. please wait`
处理：
API 启动阶段已经内置重试逻辑。一般等待一段时间即可。
如果长时间失败，检查：
```
docker compose ps
docker compose logs -f milvus
```
### 7.2 API 容器未 healthy
检查：
```aiignore
docker compose logs -f api
Invoke-RestMethod http://127.0.0.1:8000/health
```

如果 /health 正常但容器仍未 healthy，检查 docker-compose.yml 中 api.healthcheck 配置。
### 7.3 Docker Desktop / WSL 异常
可以尝试：
```aiignore
Get-Process *docker* -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process *com.docker* -ErrorAction SilentlyContinue | Stop-Process -Force
wsl --unmount
wsl --shutdown
```
然后重新打开 Docker Desktop。

---