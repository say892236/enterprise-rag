# config/settings.py

"""
项目统一配置文件。

作用：
1. 集中管理模型名称、向量库路径、top_k 等配置。
2. 避免在多个 test / eval 文件里重复写相同参数。
3. 后续接 FastAPI 时，也可以直接复用这里的配置。
"""
import os
from dotenv import load_dotenv



# ==========================
# Vector Database 配置
# ==========================
load_dotenv()


# Chroma 持久化目录
CHROMA_PERSIST_DIR = "data/vectorstore/chroma"

# Chroma collection 名称
CHROMA_COLLECTION_NAME = "enterprise_rag"

# 当前使用的向量数据库。
# 可选值：
# - chroma
# - milvus
VECTOR_DB = os.getenv(
    "VECTOR_DB",
    "chroma",
).lower()


# ==========================
# Milvus 配置
# ==========================

# Milvus 连接地址。
# 本地 Milvus Lite 可以使用本地 .db 文件路径：
# data/vectorstore/milvus/enterprise_rag.db
#
# 后续 Docker Milvus 服务可以使用：
# http://127.0.0.1:19530
MILVUS_URI = os.getenv(
    "MILVUS_CONNECTION_URI",
    "data/vectorstore/milvus/enterprise_rag.db",
).strip()

# Milvus Collection 名称。
MILVUS_COLLECTION_NAME = os.getenv(
    "MILVUS_COLLECTION_NAME",
    "enterprise_rag",
).strip()

# 构建数据库时是否删除旧 Collection。
MILVUS_DROP_OLD = (
    os.getenv("MILVUS_DROP_OLD", "true").strip().lower() == "true"
)

# ==========================
# 文档切片文件配置
# ==========================

# 已切分好的 chunks 文件，用于 BM25 加载
CHUNKS_FILE = "data/processed/chunks.jsonl"


# ==========================
# Embedding 配置
# ==========================

# 向量模型
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 当前你是 CPU 环境
EMBEDDING_DEVICE = "cpu"


# ==========================
# Reranker 配置
# ==========================

# BGE 重排模型
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# 当前使用 CPU
RERANKER_DEVICE = "cpu"

# Reranker 输入最大长度
RERANKER_MAX_LENGTH = 512


# ==========================
# LLM 配置
# ==========================

LLM_MODEL = "deepseek-v4-flash"
LLM_PROVIDER = "deepseek"
LLM_TEMPERATURE = 0.0


# ==========================
# 检索参数
# ==========================

# Dense 召回数量
DENSE_TOP_K = 10

# BM25 召回数量
BM25_TOP_K = 10

# Hybrid 融合后返回数量
HYBRID_TOP_K = 10

# Reranker 最终保留数量
RERANK_TOP_K = 5


# ==========================
# Hybrid RRF 配置
# ==========================

# Dense 通道权重
DENSE_WEIGHT = 1.0

# BM25 通道权重
BM25_WEIGHT = 1.0

# RRF 平滑参数
RRF_K = 60




# ==========================
# JWT 配置
# ==========================

JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "enterprise-rag-local-dev-secret-key",
)

JWT_ALGORITHM = os.getenv(
    "JWT_ALGORITHM",
    "HS256",
)

JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv(
        "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
        "60",
    )
)

# ==========================
# Logging 配置
# ==========================

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
)

LOG_DIR = os.getenv(
    "LOG_DIR",
    "logs",
)

LOG_FILE = os.getenv(
    "LOG_FILE",
    "logs/app.log",
)


# ==========================
# API Server 配置
# ==========================

API_HOST = os.getenv(
    "API_HOST",
    "127.0.0.1",
)

API_PORT = int(
    os.getenv(
        "API_PORT",
        "8000",
    )
)

#.env 里读取出来的都是字符串，手动转成布尔值
API_RELOAD = os.getenv(
    "API_RELOAD",
    "true",
).lower() == "true"


# ==========================
# CORS 配置
# ==========================

CORS_ALLOW_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

CORS_ALLOW_CREDENTIALS = (
    os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
)

CORS_ALLOW_METHODS = os.getenv(
    "CORS_ALLOW_METHODS",
    "*",
).split(",")

CORS_ALLOW_HEADERS = os.getenv(
    "CORS_ALLOW_HEADERS",
    "*",
).split(",")


# ==========================
# Retrieval Confidence Guard 配置
# ==========================

ENABLE_CONFIDENCE_GUARD = (
    os.getenv("ENABLE_CONFIDENCE_GUARD", "true").lower() == "true"
)

CONFIDENCE_MIN_OVERLAP_COUNT = int(
    os.getenv("CONFIDENCE_MIN_OVERLAP_COUNT", "1")
)

CONFIDENCE_CHECK_TOP_N = int(
    os.getenv("CONFIDENCE_CHECK_TOP_N", "3")
)
