# Enterprise RAG Knowledge Base

企业级 RAG 知识库问答系统。

本项目用于构建一个企业内部制度知识库，支持对 Markdown 企业制度文档进行加载、清洗、切分、向量化、混合检索、重排和大模型问答。

---

## 1. 项目目标

本项目是一个面向企业内部知识库场景的企业级 RAG 问答系统，目标是模拟真实公司内部制度、流程、权限文档的智能问答能力。

系统面向员工、部门人员和管理员等不同角色，支持在权限范围内查询企业制度文档，并返回可追溯、可解释、低幻觉的问答结果。

典型问答场景包括：

- 员工出差住宿标准是什么？
- 上海出差普通员工住宿上限是多少？
- 员工出差结束后多久内必须提交报销？
- 单次市内打车超过 200 元需要怎么处理？
- FIN-TR-2026-01 是什么制度？
- 出差可以住五星级酒店吗？
- HR 私有薪酬制度中有哪些审批规则？
- 财务预算制度中部门预算调整需要什么流程？
- IT 管理制度中高权限账号如何申请？

系统最终目标：

- 构建一套完整的企业级 RAG 问答链路
- 支持 Markdown 企业制度文档的加载、清洗、切分和向量化
- 支持 Milvus 企业级向量数据库作为主检索后端
- 支持 Chroma 本地开发模式与 Milvus 主方案的配置切换
- 支持 Dense 向量检索与 BM25 关键词检索的 Hybrid Search
- 支持 RRF 融合排序，提升召回稳定性
- 支持 BGE Reranker 精排，提高最终上下文质量
- 支持 Query Rewrite 查询改写，增强模糊问题、口语问题和编号类问题的检索效果
- 支持 Confidence Guard，对知识库无关问题进行低置信度拒答
- 支持 Prompt Guardrail，约束大模型只能基于知识库上下文回答
- 支持 Sources Trace，返回答案来源、文档标题、chunk_id 和检索分数
- 支持 JWT 登录认证
- 支持基于用户角色的文档权限控制，区分公共文档与部门私有文档
- 支持 FastAPI 服务化，对外提供标准化问答接口
- 支持统一错误响应、参数校验、日志管理和 CORS 配置
- 支持离线测试、API 集成测试和 RAG 效果评估
- 后续支持 Docker 化部署，并切换为 Milvus Server 作为生产级向量数据库
---

## 2. 当前 RAG 主链路

当前项目已实现完整企业级 RAG 主链路：

1. Markdown 企业制度文档
2. 文档加载：Markdown → LangChain Document
3. 文本清洗：保留标题、表格、列表结构
4. Markdown 结构化切分：Header Splitter + Recursive Splitter
5. bge-m3 Embedding 向量化
6. 向量数据库持久化：
   - Chroma：本地轻量开发模式
   - Milvus / Milvus Lite：企业级向量数据库主方案
7. Query Rewrite：将用户问题改写为更适合检索的问题
8. Hybrid Retriever：
   - Dense Retriever：Chroma / Milvus 向量检索
   - BM25 Retriever：关键词、编号、结构化字段检索
   - RRF：对 Dense 与 BM25 结果进行排名融合
9. BGE Reranker：对候选文档进行精排
10. Confidence Guard：低置信度问题拒答
11. Prompt Guardrail：约束大模型只能基于知识库上下文回答
12. LLM Answer：生成最终回答
13. Sources Trace：返回文档来源、标题、chunk_id、检索分数等溯源信息
14. FastAPI 服务化：提供登录认证与 `/chat` 问答接口
15. JWT 权限控制：根据用户角色限制可检索的公共 / 部门私有文档

当前主推荐链路：

```text
User Question
↓
Query Rewrite
↓
Hybrid Retriever
    ├── Dense Retriever：Milvus / Chroma
    └── BM25 Retriever
↓
RRF Fusion
↓
BGE Reranker
↓
Confidence Guard
↓
Prompt Guardrail
↓
LLM Answer
↓
Sources Trace
```
当前项目支持通过 .env 中的 VECTOR_DB 切换向量数据库：

可选值：
```
chroma：使用本地 Chroma
milvus：使用 Milvus / Milvus Lite
VECTOR_DB=milvus
```
说明：
- Chroma：早期开发阶段使用，当前作为本地轻量开发与回退方案保留
- Milvus：企业级向量数据库，作为项目主推荐方案
- 项目最终稳定后，计划移除 Chroma，仅保留 Milvus 作为主向量数据库
- BM25 会继续保留，用于关键词、编号、制度条款、结构化字段等精确检索场景



## 3. 技术栈

| 模块 | 技术 |
| --- | --- |
| 开发语言 | Python |
| 文档格式 | Markdown |
| 文档对象 | LangChain Document |
| 文档加载 | 自定义 Markdown Loader |
| 文本清洗 | 自定义 Text Cleaner |
| 文本切分 | MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter |
| Embedding | BAAI/bge-m3 |
| 向量数据库 | Chroma / Milvus / Milvus Lite |
| 向量数据库切换 | VECTOR_DB 配置切换 |
| Dense 检索 | Chroma Retriever / Milvus Retriever |
| 关键词检索 | BM25 + jieba |
| Hybrid Search | Dense + BM25 + RRF |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Query Rewrite | LangChain + LLM |
| 低置信度拒答 | Confidence Guard |
| Prompt 约束 | Prompt Guardrail |
| LLM | DeepSeek / LangChain init_chat_model |
| API 服务 | FastAPI |
| API 文档 | Swagger / OpenAPI |
| 用户认证 | JWT |
| 密码校验 | passlib + bcrypt |
| 权限控制 | Role-Based Access Control |
| 配置管理 | python-dotenv |
| 日志管理 | logging + RotatingFileHandler |
| 跨域处理 | FastAPI CORSMiddleware |
| API 测试 | requests |
| 评估格式 | JSONL |
| 单元测试 | Python 测试脚本 |
| 项目入口 | build_database.py / run_api.py |

---

##  4. 项目结构

```text
enterprise-rag/
├── api/
│   ├── auth/
│   │   ├── dependencies.py          # JWT 当前用户解析与鉴权依赖
│   │   └── users.py                 # Mock 用户表、密码校验、用户角色
│   ├── core/
│   │   ├── exceptions.py            # 全局异常处理与统一错误响应
│   │   └── logging_config.py        # 日志配置
│   ├── routes/
│   │   ├── auth.py                  # 登录接口
│   │   └── chat.py                  # RAG 问答接口
│   ├── schemas/
│   │   └── chat.py                  # Chat API 请求与响应模型
│   ├── access_control.py            # 文档权限控制逻辑
│   ├── dependencies.py              # FastAPI 依赖注入与 RAG Chain 缓存
│   └── main.py                      # FastAPI 应用入口
│
├── archive/
│   └── old_tests/                   # 旧阶段测试脚本归档
│
├── chains/
│   ├── confidence_guard.py          # 低置信度拒答机制
│   └── rag_chain.py                 # 企业级 RAG Chain 主逻辑
│
├── config/
│   └── settings.py                  # 项目统一配置
│
├── data/
│   ├── raw/
│   │   └── markdown/                # 原始 Markdown 企业制度文档
│   ├── processed/
│   │   └── chunks.jsonl             # 切分后的 chunks，供 BM25 使用
│   ├── vectorstore/
│   │   ├── chroma/                  # Chroma 向量库持久化目录
│   │   └── milvus/                  # Milvus Lite 本地数据库文件目录
│   └── eval/                        # 评估结果 JSONL 输出目录
│
├── embeddings/
│   └── bge_embedding.py             # bge-m3 Embedding 初始化封装
│
├── eval/
│   ├── questions.py                 # 统一评估问题集
│   ├── eval_runner.py               # 通用评估执行器
│   ├── baseline_eval.py             # Baseline 评估
│   ├── reranker_eval.py             # Reranker 评估
│   ├── rewrite_reranker_eval.py     # Rewrite + Reranker 评估
│   ├── hybrid_rewrite_reranker_eval.py # Hybrid + Rewrite + Reranker 评估
│   ├── compare_all_eval.py          # 多阶段评估结果对比
│   ├── no_answer_eval.py            # 低置信度拒答评估
│   └── negative_answer_eval.py      # 制度否定类问题评估
│
├── examples/
│   └── api_client_demo.py           # API 客户端调用示例
│
├── loader/
│   └── markdown_loader.py           # Markdown 文档加载器
│
├── pipelines/
│   └── rag_pipeline.py              # RAG Pipeline 构建入口
│
├── reranker/
│   └── bge_reranker.py              # BGE Reranker + Business Boost
│
├── retriever/
│   ├── chroma_retriever.py          # Chroma Dense Retriever
│   ├── milvus_retriever.py          # Milvus Dense Retriever
│   ├── bm25_retriever.py            # BM25 Keyword Retriever
│   └── hybrid_retriever.py          # Dense + BM25 Hybrid Retriever
│
├── rewriter/
│   └── query_rewriter.py            # Query Rewrite 查询改写器
│
├── splitter/
│   └── markdown_splitter.py         # Markdown 文档结构化切分
│
├── tests/
│   ├── test_auth.py                 # 登录认证测试
│   ├── test_access_control.py       # 权限控制测试
│   ├── test_confidence_guard.py     # 低置信度拒答测试
│   ├── test_vector_db_config.py     # 向量数据库配置测试
│   ├── test_milvus_store.py         # Milvus Store 测试
│   ├── test_milvus_retriever.py     # Milvus Retriever 测试
│   ├── test_milvus_retrieve_real.py # Milvus 真实检索测试
│   ├── test_api.py                  # FastAPI 接口集成测试
│   ├── smoke_test.py                # 核心链路快速测试
│   └── run_all_tests.py             # 离线测试统一入口
│
├── utils/
│   └── text_cleaner.py              # 文本清洗工具
│
├── vectorstore/
│   ├── chroma_store.py              # Chroma 构建与加载
│   └── milvus_store.py              # Milvus 构建与加载
│
├── .env.example                     # 环境变量示例
├── .gitignore                       # Git 忽略配置
├── build_database.py                # 构建知识库入口
├── run_api.py                       # FastAPI 启动入口
├── run_rag_demo.py                  # RAG Demo 运行入口
├── requirements.txt                 # 项目依赖
└── README.md                        # 项目说明文档
```
说明：
- data/processed/chunks.jsonl 不建议忽略，因为 BM25 Retriever 依赖该文件
- data/vectorstore/ 可以忽略，因为 Chroma / Milvus 向量库都可以通过 build_database.py 重建
- tests/test_api.py 需要先启动 FastAPI 服务后再运行
- tests/test_milvus_retrieve_real.py 依赖已构建好的 Milvus 数据库，属于手动验证测试
- 使用 Milvus Lite 本地 .db 文件模式时，同一时间不要让多个 Python 进程同时访问该数据库文件

---
## 5. 环境准备

### 5.1 创建虚拟环境

项目建议使用 Python 虚拟环境管理依赖。

在项目根目录执行：

```bash
python -m venv .venv
```
Windows 激活虚拟环境：

`.venv\Scripts\activate`

### 5.2 安装依赖
项目依赖统一维护在 requirements.txt 中。

安装依赖：
`pip install -r requirements.txt`

当前项目主要依赖包括：

| 类型 | 依赖 |
|------|------|
| RAG 框架     | LangChain     |
|  Embedding    | bge-m3 / sentence-transformers     |
|  向量数据库    | Milvus     |
| 关键词检索     | BM25 / jieba     |
| Reranker     |  BGE Reranker    |
| API 服务     |  FastAPI / Uvicorn    |
|  认证鉴权    |  JWT / passlib / bcrypt    |
|  配置管理    |  python-dotenv    |
|  测试与评估    |  requests / JSONL    |
注意：
- 当前项目已完成 FastAPI 服务化，不需要再单独“后续安装 FastAPI”
- 如果修改了依赖，请同步更新 requirements.txt
- Docker 镜像构建也依赖 requirements.txt
---
## 6.配置说明
### 6.1 基础配置:

DeepSeek 大模型 API Key

- DEEPSEEK_API_KEY=your_deepseek_api_key_here

JWT 签名密钥
- JWT_SECRET_KEY=replace_with_a_random_secret_key

JWT 加密算法，默认 HS256
- JWT_ALGORITHM=HS256

Token 有效期，单位分钟
- JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

### 6.2 API 服务配置
```
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true
```
说明：

| 配置项 | 作用 |
|--------|------|
|  API_HOST      |  FastAPI 本地启动 host    |
|  API_PORT      |  FastAPI 服务端口    |
|  API_RELOAD      |  是否启用热重载，本地开发可设为 true    |
Docker 部署时，容器内服务监听：
0.0.0.0:8000

### 6.3 日志配置
LOG_LEVEL=INFO

LOG_DIR=logs

LOG_FILE=logs/app.log

说明：

| 配置项 | 作用 |
|--------|------|
| LOG_LEVEL       |  日志级别    |
| LOG_DIR      |  日志目录    |
| LOG_FILE      |  日志文件路径    |

### 6.4 CORS 配置
CORS_ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

CORS_ALLOW_CREDENTIALS=true

CORS_ALLOW_METHODS=*

CORS_ALLOW_HEADERS=*

说明：

| 配置项  | 作用 |
|---|------|
| CORS_ALLOW_ORIGINS  | 允许跨域访问的前端地址     |
| CORS_ALLOW_CREDENTIALS  | 是否允许携带认证信息     |
| CORS_ALLOW_METHODS  |  允许的 HTTP 方法    |
| CORS_ALLOW_HEADERS  |  允许的请求头    |

### 6.5 低置信度拒答配置
项目已加入低置信度拒答机制，用于减少 RAG 在无关问题上的幻觉回答。

配置位置：

`chains/confidence_guard.py`

接入位置：

`chains/rag_chain.py`

相关配置：

ENABLE_CONFIDENCE_GUARD=true

CONFIDENCE_MIN_OVERLAP_COUNT=1

CONFIDENCE_CHECK_TOP_N=3

说明：

| 配置项 | 作用 |
|--------|------|
|  ENABLE_CONFIDENCE_GUARD      | 是否启用低置信度拒答     |
|  CONFIDENCE_MIN_OVERLAP_COUNT      | 问题与检索结果的最小关键词重叠数     |
| CONFIDENCE_CHECK_TOP_N       | 检查 TopN 个候选文档     |

当问题与知识库内容无关时，系统会返回：
- 知识库中未找到相关信息。

### 6.6 向量数据库配置
当前项目支持 Chroma 与 Milvus 两种向量数据库后端切换。

| 向量数据库  | 用途  |
|---|---|
| Chroma  | 早期开发阶段使用，当前作为本地轻量开发与回退方案保留  |
| Milvus  | 企业级向量数据库，作为项目主推荐方案  |
通过 .env 中的 VECTOR_DB 切换：
```aiignore
# 可选值：
# chroma：使用本地 Chroma
# milvus：使用 Milvus / Milvus Lite / Milvus Server
VECTOR_DB=milvus
```
#### Chroma 配置

使用 Chroma 时：
`VECTOR_DB=chroma`

Chroma 数据默认保存在：
`data/vectorstore/chroma`

#### Milvus Lite 本地文件模式
开发阶段可使用 Milvus Lite 本地文件模式：
```aiignore
VECTOR_DB=milvus

MILVUS_CONNECTION_URI=data/vectorstore/milvus/enterprise_rag.db
MILVUS_COLLECTION_NAME=enterprise_rag
MILVUS_DROP_OLD=true
```
注意：
- Milvus Lite 适合本地开发和快速验证
- Milvus Lite 使用本地 .db 文件
- 同一时间不适合被多个 Python 进程同时访问
- 如果 FastAPI 服务和评估脚本同时访问同一个 .db，可能出现文件锁问题
#### Milvus Server 模式
Docker 部署阶段使用 Milvus Server。
本机 Python 写入 Docker Milvus Server 时：
```aiignore
VECTOR_DB=milvus

MILVUS_CONNECTION_URI=http://127.0.0.1:19530
MILVUS_COLLECTION_NAME=enterprise_rag
MILVUS_DROP_OLD=true
```
API 容器连接 Docker Milvus Server 时，在 docker-compose.yml 中配置：
```aiignore
environment:
  VECTOR_DB: milvus
  MILVUS_CONNECTION_URI: http://milvus:19530
```
区别：
- 本机 Python → Docker Milvus Server：使用 http://127.0.0.1:19530
- API 容器 → Docker Milvus Server：使用 http://milvus:19530

---

## 7. 构建知识库

运行：

```bash
python build_database.py
```

该命令会执行：

1. 读取 `data/raw/markdown` 下的 Markdown 文档
2. 清洗文本
3. 切分 chunks
4. 保存 `data/processed/chunks.jsonl`
5. 构建 `data/vectorstore/chroma` or `data/vectorstore/milvus` 向量库

预期输出示例：

- 原始文档数量：6
- 文本清洗完成
- 切片数量：80
- 切片结果已保存到：`data/processed/chunks.jsonl`
- Chroma 向量库构建完成
- Chroma collection count：80


---

## 8. 评估流程

项目提供四阶段评估。

| 阶段 | 链路 |
| --- | --- |
| Baseline | Chroma Retriever → LLM |
| Reranker | Chroma Retriever → Reranker → LLM |
| Rewrite + Reranker | Query Rewrite → Chroma Retriever → Reranker → LLM |
| Hybrid + Rewrite + Reranker | Query Rewrite → Hybrid Retriever → Reranker → LLM |

### 8.1 运行四阶段评估

分别运行：

```bash
python -m eval.baseline_eval

python -m eval.reranker_eval

python -m eval.rewrite_reranker_eval

python -m eval.hybrid_rewrite_reranker_eval
```

评估结果会保存到 `data/eval/` 目录。

### 8.2 对比四阶段结果

运行：

`python -m eval.compare_all_eval`

用于观察不同链路下 Top1 来源变化。

---

## 9.核心模块说明

| 模块 | 位置 | 作用 |
| --- | --- | --- |
| Query Rewrite | `rewriter/query_rewriter.py` | 将口语化、模糊问题改写为更适合检索的问题，并尽量保留制度编号、FAQ 编号等关键信息 |
| Dense Retriever | `retriever/chroma_retriever.py` / `retriever/milvus_retriever.py` | 基于 bge-m3 Embedding 进行语义检索，支持 Chroma / Milvus 后端切换 |
| BM25 Retriever | `retriever/bm25_retriever.py` | 基于关键词、编号、金额、时间、专有名词进行精确检索 |
| Hybrid Retriever | `retriever/hybrid_retriever.py` | 融合 Dense Retriever 与 BM25 Retriever，并通过 RRF 进行排名融合 |
| Reranker | `reranker/bge_reranker.py` | 使用 BGE Reranker 对召回结果精排，并加入部分业务规则增强排序稳定性 |
| Confidence Guard | `chains/confidence_guard.py` | 判断检索结果与问题的相关性，对知识库无关问题进行低置信度拒答 |
| RAG Chain | `chains/rag_chain.py` | 串联 Query Rewrite、Retriever、Reranker、Confidence Guard、Prompt、LLM 与 Sources Trace |
| Pipeline Builder | `pipelines/rag_pipeline.py` | 统一构建不同 RAG 链路，并根据配置切换 Chroma / Milvus |
| Access Control | `api/access_control.py` | 根据用户角色限制可检索的公共文档与部门私有文档 |
| API Layer | `api/` | 提供登录认证、问答接口、参数校验、统一异常响应和 Swagger 文档 |
---

## 10. FastAPI 服务化

项目已接入 FastAPI，对外提供企业知识库问答接口。

启动方式：

```bash
uvicorn api.main:app --reload
```

或

```bash
python run_api.py
```

启动后访问 Swagger 文档：

`http://127.0.0.1:8000/docs`

API 启动配置可以在 `.env` 中调整：

```env
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true
```

当前接口包括：

- `GET /health`：健康检查接口
- `POST /auth/login`：登录接口，返回 JWT Token
- `POST /chat`：企业知识库问答接口，需要携带 Bearer Token

### API 客户端调用示例

项目提供一个简单的 API 调用示例：

`examples/api_client_demo.py`

运行前先启动 API 服务：

```bash
python run_api.py
```

然后另开终端运行：

```bash
python examples/api_client_demo.py
```

该脚本会自动完成登录、获取 JWT Token、调用 /chat 并打印答案和引用来源。

---

## 11. JWT 登录认证

项目已实现基础 JWT 登录认证流程。

当前使用 Mock 用户表，位置：

`api/auth/users.py`

当前测试账号如下，密码均为：

`123456`

| 用户名 | 角色 | 说明 |
| --- | --- | --- |
| ay | employee | 普通员工 |
| hr_user | hr | HR 用户 |
| finance_user | finance | 财务用户 |
| it_user | it | IT 用户 |
| admin | admin | 管理员 |

登录接口：

`POST /auth/login`

请求体示例：

`{"username": "ay", "password": "123456"}`

返回示例：

`{"access_token": "eyJhbGciOiJIUzI1Ni...", "token_type": "bearer"}`

后续调用 `/chat` 时，需要在请求头中携带：

`Authorization: Bearer <access_token>`

在 Swagger 中，可以点击右上角 `Authorize`，填入登录后返回的 Token。

---

## 12. /chat 问答接口

`/chat` 接口已经改为从 JWT 中解析用户身份，不再从请求体传入 `user_role`。

请求体示例：

```json
{
  "question": "员工出差住宿标准是什么？",
  "retrieve_top_k": 6,
  "rerank_top_k": 3,
  "metadata_filter": {"source": "travel_policy.md"},
  "enable_rewrite": true,
  "enable_rerank": true
}
```

接口内部流程

```text
JWT 认证

↓

角色权限控制

↓

Query Rewrite

↓

Hybrid Retriever(Dense + BM25)

↓

BGE Reranker

↓

LLM 生成回答

↓

返回答案和引用来源
```

主要请求字段：

| 字段 | 说明 |
| --- | --- |
| question | 用户问题 |
| retrieve_top_k | Retriever 召回数量 |
| rerank_top_k | Reranker 保留数量 |
| metadata_filter | metadata 过滤条件 |
| enable_rewrite | 是否启用 Query Rewrite |
| enable_rerank | 是否启用 Reranker |

响应中会返回：

| 字段 | 说明 |
| --- | --- |
| request_id | 请求 ID，用于日志追踪 |
| username | 当前登录用户名 |
| user_id | 当前登录用户 ID |
| user_role | 当前登录用户角色 |
| question | 原始问题 |
| rewritten_question | 改写后的问题 |
| answer | 模型回答 |
| sources | 引用来源 |
| metadata_filter | 最终生效的权限过滤条件 |
| cost_time_seconds | 请求耗时 |

### 12.1 统一成功响应

`/chat` 成功响应采用统一外层结构：

```json
{
  "success": true,
  "data": {
    "request_id": "...",
    "answer": "...",
    "sources": []
  }
}
```

### 12.2 统一错误响应

项目已统一 API 错误响应结构。

错误响应示例：

```json
{
  "success": false,
  "error_code": "AUTH_REQUIRED",
  "message": "未登录，请先传入 Authorization Bearer Token",
  "request_id": null
}
```

常见错误码：

| error_code | 含义 |
| --- | --- |
| AUTH_REQUIRED | 未登录 |
| TOKEN_INVALID | Token 无效或已过期 |
| TOKEN_MISSING_SUB | Token 中缺少用户标识 |
| USER_NOT_FOUND | Token 对应用户不存在 |
| LOGIN_FAILED | 用户名或密码错误 |
| PERMISSION_DENIED | 当前角色无权限访问指定文档 |
| VALIDATION_ERROR | 请求参数校验失败 |
| RAG_EXECUTION_FAILED | RAG 执行失败 |

### 12.3 请求参数校验

| 参数   | 规则 |
| --- | --- |
| question | 必填，不能为空，最长 500 字 |
| retrieve_top_k | 1-20 |
| rerank_top_k | 1-10 |
| rerank_top_k | 不能大于 retrieve_top_k |
| metadata_filter | 可选，支持按 source 过滤 |
| enable_rewrite | boolean |
| enable_rerank | boolean |
| 未定义字段 | 禁止传入，返回 422 |

参数错误时候返回

```json
{
  "success": false,
  "error_code": "VALIDATION_ERROR",
  "message": "请求参数校验失败，请检查请求体格式和字段类型",
  "request_id": null,
  "errors": []
}
```

---

## 13.权限控制

项目已实现基于角色的文档访问控制，用于模拟企业内部知识库中的部门权限隔离。

权限控制位置：

`api/access_control.py`

权限控制依赖文档的 `metadata.source` 字段，因此 `PUBLIC_SOURCES` 和 `DEPARTMENT_SOURCE_MAP` 中配置的文件名必须与 `data/raw/markdown/` 下的 Markdown 文件名完全一致。

---

### 角色权限说明

| 角色 | 权限说明 |
|---|---|
| employee | 普通员工，只能访问公共制度文档 |
| hr | 可访问公共制度文档 + HR 部门私有文档 |
| finance | 可访问公共制度文档 + 财务部门私有文档 |
| it | 可访问公共制度文档 + IT 部门私有文档 |
| admin | 管理员，可访问所有文档 |

---

### 公共文档

普通员工和所有部门角色均可访问以下公共文档：

```text
company_kb.md
leave_policy.md
employee_handbook.md
travel_policy.md
it_security_policy.md
reimbursement_policy.md
```
### 部门私有文档
 
| 部门角色    | 可访问私有文档 |
|---------|---------|
| hr      |  hr_salary_policy.md       |
| finance |  finance_budget_policy.md       |
|    it     |  it_admin_policy.md       |
|   admin      |  所有私有文档       |

---

### 权限访问实例
employee 访问 HR 私有文档

请求：
```
{
  "question": "薪资调整流程是什么？",
  "metadata_filter": {
    "source": "hr_salary_policy.md"
  }
}
```
返回：
```
{
  "success": false,
  "error_code": "PERMISSION_DENIED",
  "message": "当前角色无权访问文档: hr_salary_policy.md",
  "request_id": "..."
}
```
hr_user 访问 HR 私有文档

请求：
```
{
  "question": "薪资调整流程是什么？",
  "metadata_filter": {
    "source": "hr_salary_policy.md"
  }
}
```
返回：
```
  "success": true,
  "data": {
    "user_role": "hr",
    "answer": "...",
    "sources": [
      {
        "source": "hr_salary_policy.md"
      }
    ]
  }
}
```

权限过滤流程：

1. 用户登录获取 JWT。
2. `/chat` 解析 JWT，获得用户角色。
3. 根据角色生成 `effective_metadata_filter`。
4. Retriever 在检索前应用 metadata filter。
5. 如果用户请求无权限文档，返回 `403 Forbidden`。

当用户请求无权限访问的文档时，会返回：

```json
{
  "success": false,
  "error_code": "PERMISSION_DENIED",
  "message": "...",
  "request_id": "..."
}
```
---

## 14. 测试命令

当前项目已有三类基础测试。

### 14.1 认证模块测试

运行：

```bash
python -m tests.test_auth
```

测试内容：

- 用户查询
- 密码校验
- JWT 生成和解析
- Token 用户身份映射
- 过期 Token 校验

---

### 14.2 权限控制测试

运行：

```bash
python -m tests.test_access_control
```

该测试覆盖

1. employee 只能访问公共文档
2. hr 可以访问 HR 私有文档
3. finance 可以访问财务私有文档
4. it 可以访问 IT 私有文档
5. admin 不限制文档访问范围
6. 部门之间不能互相访问私有文档
7. source.$in 会自动过滤无权限文档
8. source.$in 全部无权限时返回PERMISSION_DENIED

### API集成测试
需要先启动 API 服务：
```
python run_api.py
```
 然后另开终端运行
```
python -m tests.test_api
```
该测试覆盖：

1. employee 访问 HR 私有文档返回 PERMISSION_DENIED
2. hr_user 可以访问 HR 私有文档
3. finance_user 可以访问财务私有文档
4. it_user 可以访问 IT 私有文档
5. admin 可以访问所有部门私有文档
---


### 14.3 检索链路 Smoke Test

运行：

```bash
python -m tests.smoke_test
```

测试内容：

- Dense Retriever 是否正常
- BM25 Retriever 是否正常
- Hybrid Retriever 是否正常

---

### 14.4 推荐统一测试命令

核心离线测试，不依赖FastAPI

运行：

```bash
python -m tests.run_all_tests
```

该命令依次运行

- tests.test_auth
- tests.test_access_control
- tests.smoke_test

---

### 14.5 推荐完整检查命令

```bash
python -m tests.run_all_tests

python run_rag_demo.py
```

如果这些都通过，说明认证、权限、检索和 RAG Demo 主链路基本正常。

### API 集成测试

API 测试需要先启动 FastAPI 服务：

```bash
python run_api.py
```

然后另开终端运行

```bash
python -m tests.test_api
```

该测试会检查：

- /health 健康检查

- /auth/login 正确登录

- /auth/login 错误密码返回 `LOGIN_FAILED`

- /chat 未登录返回 `AUTH_REQUIRED`

- /chat 无权限文档访问返回 `PERMISSION_DENIED`

- /chat 带 Token 正常问答

---

目前项目主入口

```text
build_database.py
run_rag_demo.py
run_api.py
```

分别对应：

### 重新构建知识库

```bash
python build_database.py
```

### 本地命令行问答 Demo

```bash
python run_rag_demo.py
```

### 启动 FastAPI 服务

```bash
python run_api.py
```

---

## 15. 日志管理

项目使用 Python 标准 `logging` 进行日志管理。

日志配置位置：

`api/core/logging_config.py`

日志输出位置：

- 控制台
- `logs/app.log`

日志中会记录：

- `request_id`
- `username`
- `user_role`
- `question`
- `metadata_filter`
- `cost_time`
- `error`

日志文件支持自动轮转，单个文件最大 10MB，最多保留 5 个历史文件。

注意：`logs/` 目录不建议提交到 Git 仓库。

.gitignore 中应包含：

- logs/

- *.log


## 16. API客户端调用示例

```text
examples/api_client_demo.py
```

运行前先启动API服务

```bash
python run_api.py
```

然后另开终端运行：

```bash
python examples/api_client_demo.py
```

该脚本会自动完成：

```text
调用 /auth/login 登录
↓
获取 JWT Token
↓
携带 Authorization 请求 /chat
↓
打印答案和引用来源
```

---
## 17.RAG 评估脚本

项目提供多组评估脚本，用于检查不同 RAG 场景下的回答质量。

### 基础问答评估

运行：

```bash
python -m eval.hybrid_rewrite_reranker_eval
```
用于评估完整链路：
```aiignore
Query Rewrite
→ Hybrid Retriever(Dense + BM25)
→ BGE Reranker
→ LLM
```
### 多链路对比评估
```
python -m eval.compare_all_eval
```
用于对比不同 RAG 链路的检索和回答效果

### 低置信度拒答评估
```aiignore
python -m eval.no_answer_eval
```
该脚本用于测试知识库中不存在的问题，例如：
- 公司食堂几点开门？
- 老板今年几岁？
- 公司附近有没有停车场？
- 公司有没有健身房？

评估目标：
1. answer 是否等于：知识库中未找到相关信息。
2. sources 是否为空列表。
3. 系统是否避免根据无关上下文强行回答。

### 制度否定类问题评估
运行:
```aiignore
python -m eval.negative_answer_eval
```
该脚本用于测试知识库中有制度依据、但结论是否定的问题，例如:

- 员工可以报销奶茶吗？ 
- 交通罚款可以报销吗？
- 私人聚餐可以报销吗？
- 电子发票截图可以作为报销凭证吗？

这类问题不应该触发低置信度拒答，而应该根据知识库制度给出明确否定回答，并返回对应 sources。

评估目标：
1. answer 不应该是固定拒答文本。
2. sources 不应该为空。
3. sources 中应包含预期制度文档。
4. answer 中应包含关键业务词和否定表达。
---
## 18.Docker 化部署

当前项目已支持通过 Docker / Docker Compose 启动企业级 RAG 服务。

Docker 化后，系统由以下服务组成：

```text
docker-compose

├── api
│   └── FastAPI + Enterprise-RAG 主服务
│
├── milvus
│   └── Milvus Standalone 向量数据库服务
│
├── etcd
│   └── Milvus 元数据存储
│
└── minio
    └── Milvus 对象存储
    
```
### 整体链路
User

↓

FastAPI Container

↓

Query Rewrite

↓

Hybrid Retriever

    ├── Milvus Dense Retriever
    └── BM25 Retriever
↓

BGE Reranker

↓

Confidence Guard

↓

LLM Answer

↓

Sources Trace

### 1. 启动 Milvus 相关服务
首次启动前，需要确保 Docker Desktop 已正常运行。
启动 Milvus Server 及其依赖服务：
`docker compose up -d etcd minio milvus`

查看服务状态： 

docker compose ps

正常情况下应看到：
- milvus-etcd        healthy
- milvus-minio       healthy
- milvus-standalone  healthy

### 2. 构建知识库到 Docker Milvus Server
当本机执行 build_database.py 写入 Docker 中的 Milvus Server 时，.env 中应配置：
- VECTOR_DB=milvus
- MILVUS_CONNECTION_URI=http://127.0.0.1:19530
- MILVUS_COLLECTION_NAME=enterprise_rag
- MILVUS_DROP_OLD=true

 注意：

本机 Python → Docker Milvus Server：使用 http://127.0.0.1:19530

API 容器 → Docker Milvus Server：使用 http://milvus:19530

执行知识库构建：
`python build_database.py`

构建成功后，应看到类似输出：

- 当前 VECTOR_DB: milvus
- Milvus URI: http://127.0.0.1:19530
- Milvus 向量库构建完成

### 3. 启动 API 服务
启动 FastAPI 容器

`docker compose up -d api`

查看 API 日志：

`docker compose logs -f api`

正常情况下应看到：
- 当前 VECTOR_DB: milvus
- 正在加载 Milvus 向量库...
- Milvus URI: http://milvus:19530
- Application startup complete.
### 4. 验证服务
健康检查：
`curl http://127.0.0.1:8000/health`

访问 Swagger 文档：`http://127.0.0.1:8000/docs`

运行 API 客户端示例：`python examples/api_client_demo.py`

### 5. HuggingFace 模型缓存
Docker 容器启动时需要加载：
- BAAI/bge-m3
- BAAI/bge-reranker-v2-m3

为避免容器重复下载模型，docker-compose.yml 中挂载本机 HuggingFace 缓存：
```
volumes:
  - ${USERPROFILE}/.cache/huggingface:/root/.cache/huggingface
```
同时设置离线模式：
```
environment:
  HF_HOME: /root/.cache/huggingface
  TRANSFORMERS_CACHE: /root/.cache/huggingface
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"
```
如果本机 HuggingFace 缓存不完整，容器可能会启动失败。此时需要先在本机完成模型下载，或者关闭离线模式让容器联网下载。

### 6. PowerShell 中文编码说明
在 PowerShell 中直接调用中文接口时，可能出现响应乱码。

推荐使用：
`python examples/api_client_demo.py`
或者在 Swagger 中测试中文请求。
如果必须在 PowerShell 中测试中文 JSON 请求体，建议使用 UTF-8 bytes 方式发送请求：
```
$bodyObj = @{
  question = "员工出差住宿标准是什么？"
  retrieve_top_k = 10
  rerank_top_k = 5
}

$bodyJson = $bodyObj | ConvertTo-Json -Depth 10
$bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($bodyJson)
```
### 7. Milvus Lite 与 Milvus Server 区别

开发阶段曾使用 Milvus Lite 本地文件模式：
`MILVUS_CONNECTION_URI=data/vectorstore/milvus/enterprise_rag.db`

Docker 部署阶段使用 Milvus Server：
`MILVUS_CONNECTION_URI=http://127.0.0.1:19530`

项目最终推荐使用 Milvus Server 作为主向量数据库方案。
Milvus Lite 本地 .db 文件模式存在文件锁限制，同一时间不适合被多个 Python 进程同时访问。

---
## 19. 当前项目能力总结

当前项目已经具备：

```text
1. Markdown 企业制度文档加载
2. 文本清洗
3. Markdown Header 切分
4. bge-m3 Embedding
5. Chroma 向量库
6. BM25 稀疏检索
7. Hybrid Search 检索融合
8. Query Rewrite 查询改写
9. BGE Reranker 重排
10. RAG 问答链路
11. FastAPI 服务化
12. JWT 登录认证
13. 角色权限控制
14. 部门私有文档权限隔离
15. 统一成功响应
16. 统一错误响应
17. 请求参数校验
18. API 日志追踪
19. API 客户端示例
20. CORS 跨域配置
21. 低置信度拒答机制
22. 自动化测试
23. docker化部署
```

---

## 后续计划

后续可以继续优化：

```text
1. 增加 RAG 命中率和答案质量评估
2. 开发简单前端页面调用 API
3. 支持多轮对话记忆
4. 支持上传文档并自动入库
```
