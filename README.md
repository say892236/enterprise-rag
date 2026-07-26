# Enterprise-RAG 企业级知识库问答系统

本项目是一个面向企业制度文档的 RAG 知识库问答系统，支持文档切分、向量检索、Hybrid Search、Query Rewrite、Reranker、Context Compression、低置信度拒答、权限过滤、FastAPI 接口化和 Docker Compose 部署。

## 1. 项目目标

本项目模拟企业内部知识库场景，用于解决员工对企业制度文档的问答需求，例如：

- 差旅住宿标准
- 报销提交时限
- 请假制度
- 员工手册
- 安全合规
- 不同角色的文档访问权限控制

系统目标不是简单调用大模型，而是构建一套可追踪、可评估、可部署的企业级 RAG 应用。

## 2. 核心能力

### 2.1 RAG 问答链路

- Markdown 企业制度文档加载
- 文本清洗
- 分层切分
- Metadata 保留
- Embedding 向量化
- Milvus 向量库检索
- BM25 关键词检索
- Hybrid Search 融合检索
- Query Rewrite 查询改写
- BGE Reranker 精排
- Context Compression 上下文压缩
- 低置信度拒答
- 引用来源返回

### 2.2 API 能力

- FastAPI 接口服务
- JWT 登录认证
- 基于用户角色的权限过滤
- 统一异常响应
- Request ID 请求追踪
- 请求日志中间件
- `/health` 存活检查
- `/readiness` 就绪检查

### 2.3 工程化能力

- Docker Compose 一键启动
- Milvus / Etcd / MinIO / API 服务编排
- API 容器 healthcheck
- 集成测试覆盖核心 API
- 最终 RAG 评估脚本
- 评估报告输出 JSON / Markdown

## 3. 技术栈

| 模块 | 技术 |
|---|---|
| Web 框架 | FastAPI |
| RAG 框架 | LangChain |
| 向量库 | Milvus |
| 关键词检索 | BM25 |
| Embedding | bge-m3 |
| Reranker | bge-reranker-v2-m3 |
| 权限认证 | JWT |
| 容器化 | Docker Compose |
| 文档格式 | Markdown |
| 测试 | requests 集成测试 |

## 4. 项目结构

```text
enterprise-rag/
├── api/                    # FastAPI 接口层
│   ├── auth/               # 登录认证、JWT
│   ├── core/               # 异常处理、日志、中间件
│   ├── routes/             # API 路由
│   ├── schemas/            # Pydantic 请求/响应模型
│   ├── access_control.py   # 权限过滤
│   ├── dependencies.py     # FastAPI 依赖注入
│   └── main.py             # FastAPI 入口
├── chains/                 # RAG Chain 核心逻辑
├── compressor/             # Context Compression
├── data/                   # 原始文档、处理结果、评估报告
├── docs/                   # 项目文档
├── eval/                   # 评估脚本
├── loader/                 # 文档加载
├── retriever/              # 向量检索、BM25、Hybrid Search、Query Router
├── reranker/               # Reranker
├── rewriter/               # Query Rewrite
├── services/               # 业务编排层
├── splitter/               # 文档切分
├── tests/                  # 测试脚本
├── vectorstore/            # Milvus / Chroma 向量库封装
├── docker-compose.yml
└── README.md
```
---
## 5.RAG主链路
用户问题

  ↓

JWT 认证

  ↓
角色权限过滤

  ↓

Query Router 查询分类

  ↓

Query Rewrite 查询改写

  ↓

Hybrid Search 检索
  ├── Dense Vector Search
  └── BM25 Keyword Search
  ↓

RRF 融合排序

  ↓

BGE Reranker 精排

  ↓

Confidence Guard 低置信度判断

  ↓

Context Compression 上下文压缩

  ↓

LLM 生成答案

  ↓

返回答案 + 引用来源 + 耗时信息

---
## 6.API 接口
### 6.1 存活检查
`GET /health`

用于检查 FastAPI 服务是否存活。
### 6.2 就绪检查
`GET /readiness`

用于检查 RAG Chain、Retriever、向量库检索是否可用。
### 6.3 登录接口
`POST /auth/login`

用于获取 JWT Token。
### 6.4 企业知识库问答
`POST /chat`

用于向企业知识库提问。

请求示例：
```aiignore
{
  "question": "员工出差住宿标准是什么？",
  "retrieve_top_k": 5,
  "rerank_top_k": 5,
  "enable_rewrite": true,
  "enable_rerank": true,
  "enable_compression": true
}
```

返回内容包括：
- 答案
- 引用来源
- 当前用户信息
- 权限过滤条件
- 查询路由信息
- 各阶段耗时
- 上下文压缩信息

---
## 7. Docker Compose 启动
`docker compose up -d`

查看容器状态：

`docker compose ps`

查看 API 日志：

`docker compose logs -f api`

详细启动说明见：
`docs/docker_runbook.md`

---
## 8. 健康检查
`Invoke-RestMethod http://127.0.0.1:8000/health`

`Invoke-RestMethod http://127.0.0.1:8000/readiness | ConvertTo-Json -Depth 10`

---

## 9. 集成测试
`python -m tests.test_docker_api`4
测试覆盖：
- /health
- /readiness
- /auth/login
- /chat
- 低置信度拒答
- 权限控制

---
## 10. 最终评估结果

当前评估集结果：

| 指标 | 结果 |
|---|---:|
| 总通过率 | 100.00% |
| Hit@1 | 100.00% |
| Hit@3 | 100.00% |
| Hit@5 | 100.00% |
| Recall@5 | 100.00% |
| Precision@5 | 93.33% |
| MRR | 1.0000 |
| 关键事实覆盖率 | 100.00% |
| 忠实度 Faithfulness | 94.44% |
| 无答案拒答准确率 | 100.00% |
| 权限拦截准确率 | 100.00% |
| 平均响应时间 | 4803.69 ms |
| P50 | 5329.92 ms |
| P95 | 8309.69 ms |

评估报告输出位置：
```
data/eval/final_rag_eval_report.json
data/eval/final_rag_eval_report.md
```

### 11. 项目亮点
- 不只实现基础向量检索，而是加入 Hybrid Search、Reranker、Query Rewrite 和 Context Compression。
- 支持 Metadata 权限过滤，模拟企业内部不同角色访问不同文档的场景。
- 支持低置信度拒答，避免模型在知识库无依据时胡编。
- 返回引用来源，便于答案溯源。
- 提供完整 API 服务，而不是只停留在本地脚本。
- 使用 Docker Compose 部署 Milvus 与 API 服务。
- 提供健康检查、就绪检查和集成测试，具备基础工程化能力。


---


## 12. 后续优化方向

1.接入前端页面，提供可视化问答界面。

2.增加管理后台，支持文档上传和重建索引。

3.支持多租户知识库隔离。

4.增加更细粒度的权限控制。

5.接入 Prometheus / Grafana 监控。

6.增加异步任务队列处理文档入库。













