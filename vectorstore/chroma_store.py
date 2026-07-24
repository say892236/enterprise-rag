# vectorstore/chroma_store.py

"""
Chroma 向量库封装模块。

作用：
1. 构建 Chroma 向量库。
2. 加载已有 Chroma 向量库。
3. 统一 Chroma collection_name / persist_directory 配置。
"""

import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
)


def reset_chroma_dir(
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> None:
    """
    清理旧的 Chroma 向量库目录。

    重新构建知识库时使用，避免旧数据残留。
    """

    chroma_path = Path(persist_dir)

    if chroma_path.exists():
        shutil.rmtree(chroma_path)

    chroma_path.mkdir(
        parents=True,
        exist_ok=True,
    )


def build_chroma_vectorstore(
    documents: List[Document],
    embeddings: Embeddings,
) -> Chroma:
    """
    构建 Chroma 向量库。

    参数：
        documents: 已清洗、切分后的 Document 列表。
        embeddings: Embedding 模型。

    返回：
        Chroma VectorStore 对象。
    """

    if not documents:
        raise ValueError("documents 不能为空，无法构建 Chroma 向量库")

    print("正在构建 Chroma 向量库...")
    print(f"Chroma persist_directory: {CHROMA_PERSIST_DIR}")
    print(f"Chroma collection_name: {CHROMA_COLLECTION_NAME}")
    print(f"文档切片数量: {len(documents)}")

    # 每次构建前清理旧 Chroma 数据
    reset_chroma_dir(CHROMA_PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    print("Chroma 向量库构建完成")

    return vectorstore


def load_chroma_vectorstore(
    embeddings: Embeddings,
) -> Chroma:
    """
    加载已有 Chroma 向量库。

    参数：
        embeddings: Embedding 模型。

    返回：
        Chroma VectorStore 对象。
    """

    print("正在加载 Chroma 向量库...")
    print(f"Chroma persist_directory: {CHROMA_PERSIST_DIR}")
    print(f"Chroma collection_name: {CHROMA_COLLECTION_NAME}")

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    print("Chroma 向量库加载完成")

    return vectorstore