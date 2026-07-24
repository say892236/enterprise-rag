# build_database.py

"""
企业级 RAG 知识库构建入口。

运行方式：
python build_database.py

完整流程：
1. 读取 data/raw/markdown 下的 Markdown 文档
2. 清洗 Markdown 文本
3. 按 Markdown 标题切分
4. 再按 chunk_size / chunk_overlap 做递归切分
5. 给每个 chunk 添加 metadata
6. 保存 chunks.jsonl，供 BM25 使用
7. 构建 Chroma 向量库，供 Dense Retriever 使用
"""

import json
import shutil
from pathlib import Path
from typing import List
from embeddings.bge_embedding import create_bge_embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNKS_FILE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
)
from config.settings import VECTOR_DB

from vectorstore.milvus_store import build_milvus_vectorstore, get_milvus_collection_count

from vectorstore.chroma_store import build_chroma_vectorstore


# ==========================
# 路径配置
# ==========================

# 原始 Markdown 文档目录
RAW_MARKDOWN_DIR = Path("data/raw/markdown")

# 切片文件保存路径
CHUNKS_OUTPUT_FILE = Path(CHUNKS_FILE)

# Chroma 向量库目录
CHROMA_DIR = Path(CHROMA_PERSIST_DIR)


# ==========================
# 切分参数
# ==========================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_markdown_documents(markdown_dir: Path) -> List[Document]:
    """
    批量加载 Markdown 文档。

    每个 Markdown 文件会转成一个 LangChain Document。

    metadata 中保存：
    - source: 文件名，例如 travel_policy.md
    - file_path: 文件完整路径
    - file_type: markdown
    """

    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown 目录不存在: {markdown_dir}")

    documents = []

    for file_path in markdown_dir.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        doc = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path),
                "file_type": "markdown",
            },
        )

        documents.append(doc)

    if not documents:
        raise ValueError(f"未找到 Markdown 文件: {markdown_dir}")

    return documents


def clean_markdown_text(text: str) -> str:
    """
    清洗 Markdown 文本。

    当前清洗策略：
    1. 去掉 Markdown 引用符号 >
    2. 去掉行尾多余空格
    3. 合并过多空行
    4. 保留标题、表格、列表结构

    注意：
    不要过度清洗 Markdown。
    RAG 文档中，标题、表格、列表通常都是重要结构信息。
    """

    lines = []

    for line in text.splitlines():
        # 去掉引用符号 >
        line = line.lstrip(">")

        # 去掉行尾空格
        line = line.rstrip()

        lines.append(line)

    cleaned_text = "\n".join(lines)

    # 合并连续 3 个以上空行为 2 个空行
    while "\n\n\n" in cleaned_text:
        cleaned_text = cleaned_text.replace("\n\n\n", "\n\n")

    return cleaned_text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    清洗 Document 列表。

    保留原 metadata，只清洗 page_content。
    """

    cleaned_docs = []

    for doc in documents:
        cleaned_doc = Document(
            page_content=clean_markdown_text(doc.page_content),
            metadata=dict(doc.metadata),
        )

        cleaned_docs.append(cleaned_doc)

    return cleaned_docs


def split_markdown_documents(documents: List[Document]) -> List[Document]:
    """
    对 Markdown 文档进行两阶段切分。

    第一阶段：
    使用 MarkdownHeaderTextSplitter 按 # 和 ## 提取标题 metadata。

    第二阶段：
    使用 RecursiveCharacterTextSplitter 控制 chunk_size 和 chunk_overlap。

    最终每个 chunk 会带上：
    - source
    - file_path
    - file_type
    - header_1
    - header_2
    - chunk_index
    - chunk_id
    - char_count
    """

    # 1. 只提取一级标题和二级标题
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    # 2. 控制最终 chunk 大小
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            "。",
            "；",
            "，",
            " ",
            "",
        ],
    )

    all_chunks = []

    for doc in documents:
        source = doc.metadata.get("source")

        # MarkdownHeaderTextSplitter 返回的 Document 会带 header metadata
        header_docs = markdown_splitter.split_text(doc.page_content)

        for header_doc in header_docs:
            # 合并原始文件 metadata 和 header metadata
            merged_metadata = {
                **doc.metadata,
                **header_doc.metadata,
            }

            header_doc.metadata = merged_metadata

        # 对 header_docs 再做递归切分
        split_docs = recursive_splitter.split_documents(header_docs)

        for chunk in split_docs:
            chunk_index = len(all_chunks)

            # 复制 metadata，避免引用污染
            metadata = dict(chunk.metadata)

            metadata["chunk_index"] = chunk_index
            metadata["chunk_id"] = f"{source}:{chunk_index}"
            metadata["char_count"] = len(chunk.page_content)

            final_chunk = Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )

            all_chunks.append(final_chunk)

    return all_chunks


def save_chunks_to_jsonl(chunks: List[Document], output_file: Path) -> None:
    """
    将 chunks 保存成 JSONL。

    这个文件主要供 BM25 使用。

    每一行格式：
    {
        "page_content": "...",
        "metadata": {...}
    }
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_chroma_dir(chroma_dir: Path) -> None:
    """
    删除旧 Chroma 向量库目录。

    为什么要删？
    因为重新构建知识库时，如果不清理旧数据，
    可能出现重复数据、旧 chunk 残留、collection 混乱等问题。
    """

    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    chroma_dir.mkdir(parents=True, exist_ok=True)

def build_vectorstore_by_config(
    documents,
    embeddings,
):
    """
    根据 VECTOR_DB 配置构建向量库。

    支持：
    1. chroma：构建本地 Chroma 向量库。
    2. milvus：构建 Milvus / Milvus Lite 向量库。

    参数：
        documents: 切分后的 Document 列表。
        embeddings: Embedding 模型。

    返回：
        vectorstore 对象。
    """

    print("=" * 100)
    print("开始构建向量库")
    print(f"当前 VECTOR_DB: {VECTOR_DB}")
    print("=" * 100)

    if VECTOR_DB == "chroma":
        return build_chroma_vectorstore(
            documents=documents,
            embeddings=embeddings,
        )

    if VECTOR_DB == "milvus":
        return build_milvus_vectorstore(
            documents=documents,
            embeddings=embeddings,
        )

    raise ValueError(
        f"不支持的 VECTOR_DB 配置：{VECTOR_DB}，可选值：chroma / milvus"
    )


def main():
    """
    知识库构建主流程。

    流程：
    1. 加载原始 Markdown 文档。
    2. 清洗文本。
    3. 切分文档。
    4. 保存 chunks.jsonl。
    5. 加载 Embedding 模型。
    6. 根据 VECTOR_DB 构建 Chroma 或 Milvus 向量库。
    """

    print("=" * 80)
    print("开始构建企业级 RAG 知识库")
    print("=" * 80)

    # 1. 加载原始 Markdown 文档
    raw_docs = load_markdown_documents(RAW_MARKDOWN_DIR)
    print(f"原始文档数量：{len(raw_docs)}")

    # 2. 清洗文本
    cleaned_docs = clean_documents(raw_docs)
    print("文本清洗完成")

    # 3. 文档切分
    chunks = split_markdown_documents(cleaned_docs)
    print(f"切片数量：{len(chunks)}")

    # 4. 保存 chunks.jsonl
    save_chunks_to_jsonl(
        chunks=chunks,
        output_file=CHUNKS_OUTPUT_FILE,
    )
    print(f"切片结果已保存到：{CHUNKS_OUTPUT_FILE}")

    # 5. 加载 Embedding 模型
    # Chroma 和 Milvus 都需要使用同一个 Embedding 模型生成向量
    embeddings = create_bge_embeddings()

    # 6. 根据 VECTOR_DB 构建向量库
    # VECTOR_DB=chroma 时构建 Chroma
    # VECTOR_DB=milvus 时构建 Milvus
    vectorstore = build_vectorstore_by_config(
        documents=chunks,
        embeddings=embeddings,
    )

    print("=" * 80)
    print(f"企业级 RAG 知识库构建完成，当前向量库：{VECTOR_DB}")

    # 7. 安全打印向量库 count
    # Chroma 和 Milvus 的 count 读取方式不同，所以这里做兼容处理
    try:
        if VECTOR_DB == "chroma":
            print("Chroma collection count:", vectorstore._collection.count())

        elif VECTOR_DB == "milvus":
            count = get_milvus_collection_count(vectorstore)
            print("Milvus collection count:", count)

    except Exception as e:
        print(f"向量库 count 读取失败，但不影响构建结果：{e}")

    print("=" * 80)

if __name__ == "__main__":
    main()