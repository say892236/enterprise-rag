# vectorstore/milvus_store.py

"""
Milvus 向量库封装模块。

作用：
1. 构建 Milvus 向量库。
2. 加载已有 Milvus 向量库。
3. 统一 Milvus 连接参数和 Collection 名称。
4. 为后续 Chroma / Milvus 配置切换做准备。

说明：
- 当前默认使用 Milvus Lite 本地文件模式。
- 后续如果使用 Docker Milvus，只需要修改 MILVUS_URI。
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus

from config.settings import (
    MILVUS_COLLECTION_NAME,
    MILVUS_DROP_OLD,
    MILVUS_URI,
)


def ensure_milvus_uri_parent_dir() -> None:
    """
    确保 Milvus Lite 本地数据库文件所在目录存在。

    例如：
    MILVUS_URI=data/vectorstore/milvus/enterprise_rag.db

    需要确保：
    data/vectorstore/milvus/
    已存在。

    如果 MILVUS_URI 是远程地址，例如：
    http://127.0.0.1:19530

    则不需要创建本地目录。
    """

    # 远程 Milvus 服务地址不需要创建本地目录
    if MILVUS_URI.startswith("http://") or MILVUS_URI.startswith("https://"):
        return

    # 本地 Milvus Lite 文件路径需要创建父目录
    milvus_path = Path(MILVUS_URI)
    milvus_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def get_milvus_connection_args() -> dict:
    """
    获取 Milvus 连接参数。

    LangChain Milvus 使用 connection_args 连接 Milvus。
    当前只配置 uri，后续如果接 Zilliz Cloud 或远程 Milvus，
    可以在这里扩展 token / db_name 等参数。
    """

    return {
        "uri": MILVUS_URI,
    }


def build_milvus_vectorstore(
    documents: List[Document],
    embeddings: Embeddings,
    drop_old: bool | None = None,
) -> Milvus:
    """
    构建 Milvus 向量库。

    参数：
        documents: 已清洗、切分后的 Document 列表。
        embeddings: Embedding 模型。
        drop_old: 是否删除旧 Collection。
                  None 表示使用配置文件中的 MILVUS_DROP_OLD。

    返回：
        Milvus VectorStore 对象。
    """

    if not documents:
        raise ValueError("documents 不能为空，无法构建 Milvus 向量库")

    ensure_milvus_uri_parent_dir()

    if drop_old is None:
        drop_old = MILVUS_DROP_OLD

    print("正在构建 Milvus 向量库...")
    print(f"Milvus URI: {MILVUS_URI}")
    print(f"Milvus Collection: {MILVUS_COLLECTION_NAME}")
    print(f"drop_old: {drop_old}")
    print(f"文档切片数量: {len(documents)}")

    vectorstore = Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args=get_milvus_connection_args(),
        drop_old=drop_old,

        # 允许不同切片保存不同的 metadata 字段，
        # 避免 Schema 只依赖第一条 Document。
        enable_dynamic_field=True,
    )
    print("Milvus 向量库构建完成")

    return vectorstore


def load_milvus_vectorstore(
    embeddings: Embeddings,
) -> Milvus:
    """
    加载已有 Milvus 向量库。

    参数：
        embeddings: Embedding 模型。

    返回：
        Milvus VectorStore 对象。
    """

    ensure_milvus_uri_parent_dir()

    print("正在加载 Milvus 向量库...")
    print(f"Milvus URI: {MILVUS_URI}")
    print(f"Milvus Collection: {MILVUS_COLLECTION_NAME}")

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args=get_milvus_connection_args(),

        # 必须与创建 Collection 时保持一致，
        # 才能在检索结果中正常读取动态 metadata。
        enable_dynamic_field=True,
    )
    print("Milvus 向量库加载完成")

    return vectorstore


def get_milvus_collection_count(
    vectorstore: Milvus,
) -> int | None:
    """
    获取 Milvus Collection 中的实体数量。

    说明：
    不同 langchain-milvus / pymilvus 版本内部属性可能略有差异。
    所以这里使用安全方式读取，失败时返回 None，不影响主流程。
    """

    try:
        collection = getattr(vectorstore, "col", None)

        if collection is None:
            return None

        collection.flush()

        return collection.num_entities

    except Exception:
        return None