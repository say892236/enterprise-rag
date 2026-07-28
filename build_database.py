# build_database.py

"""
企业级 RAG 知识库构建入口。

运行方式：
python build_database.py

完整流程：
1. 读取 data/raw/markdown 下的 Markdown 文档。
2. 清洗 Markdown 文本。
3. 按 Markdown 一级、二级标题切分。
4. 再按 chunk_size / chunk_overlap 递归切分。
5. 为每个 chunk 添加 metadata。
6. 保存 chunks.jsonl，供 BM25 Retriever 使用。
7. 构建 Milvus 向量库，供 Dense Retriever 使用。
"""

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config.settings import CHUNKS_FILE
from embeddings.bge_embedding import create_bge_embeddings
from loader.markdown_loader import MarkdownLoader
from vectorstore.milvus_store import (
    build_milvus_vectorstore,
    get_milvus_collection_count,
)


# ==========================
# 路径配置
# ==========================

# 原始 Markdown 文档目录
RAW_MARKDOWN_DIR = Path("data/raw/markdown")

# 切片结果保存路径
# 该 JSONL 文件主要供 BM25 Retriever 加载使用
CHUNKS_OUTPUT_FILE = Path(CHUNKS_FILE)


# ==========================
# 切分参数
# ==========================

# 单个 chunk 的最大字符数
CHUNK_SIZE = 800

# 相邻 chunk 之间保留的重叠字符数
CHUNK_OVERLAP = 120


def clean_markdown_text(text: str) -> str:
    """
    清洗 Markdown 文本。

    当前清洗策略：
    1. 去掉 Markdown 引用符号 >。
    2. 去掉每行末尾的多余空格。
    3. 合并过多空行。
    4. 保留标题、表格和列表结构。

    注意：
    不要过度清洗 Markdown。
    标题、表格和列表都是 RAG 检索中的重要结构信息。
    """

    cleaned_lines = []

    for line in text.splitlines():
        # 去掉行首 Markdown 引用符号
        line = line.lstrip(">")

        # 去掉行尾多余空格
        line = line.rstrip()

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # 连续三个以上换行统一压缩为两个换行
    while "\n\n\n" in cleaned_text:
        cleaned_text = cleaned_text.replace(
            "\n\n\n",
            "\n\n",
        )

    return cleaned_text.strip()


def clean_documents(
    documents: List[Document],
) -> List[Document]:
    """
    清洗 Document 列表。

    只清洗 page_content，
    原始 metadata 会完整保留。
    """

    cleaned_documents = []

    for document in documents:
        cleaned_document = Document(
            page_content=clean_markdown_text(
                document.page_content,
            ),
            metadata=dict(document.metadata),
        )

        cleaned_documents.append(cleaned_document)

    return cleaned_documents


def split_markdown_documents(
    documents: List[Document],
) -> List[Document]:
    """
    对 Markdown 文档执行两阶段切分。

    第一阶段：
    使用 MarkdownHeaderTextSplitter 按 # 和 ## 切分，
    并提取 header_1、header_2 metadata。

    第二阶段：
    使用 RecursiveCharacterTextSplitter 控制最终
    chunk_size 和 chunk_overlap。

    最终每个 chunk 包含：

    - source
    - file_path
    - file_type
    - header_1
    - header_2
    - chunk_index
    - chunk_id
    - char_count
    """

    # 第一阶段：按 Markdown 一级、二级标题切分
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,

        # 保留标题文本，让标题内容参与 Embedding 和 BM25 检索
        strip_headers=False,
    )

    # 第二阶段：控制最终 chunk 大小
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

    all_chunks: List[Document] = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "unknown",
        )

        # 按 Markdown 标题切分当前文档
        header_documents = markdown_splitter.split_text(
            document.page_content,
        )

        # MarkdownHeaderTextSplitter 生成的新 Document
        # 只包含标题 metadata，需要与原始文件 metadata 合并
        for header_document in header_documents:
            header_document.metadata = {
                **document.metadata,
                **header_document.metadata,
            }

        # 对标题切分结果继续执行递归字符切分
        split_documents = recursive_splitter.split_documents(
            header_documents,
        )

        for split_document in split_documents:
            # 跳过意外产生的空切片
            if not split_document.page_content.strip():
                continue

            # 使用全局递增索引，保证 chunk_index 唯一
            chunk_index = len(all_chunks)

            # 复制 metadata，避免不同 Document 共用同一字典
            metadata = dict(split_document.metadata)

            metadata["chunk_index"] = chunk_index
            metadata["chunk_id"] = (
                f"{source}:{chunk_index}"
            )
            metadata["char_count"] = len(
                split_document.page_content,
            )

            final_chunk = Document(
                page_content=split_document.page_content,
                metadata=metadata,
            )

            all_chunks.append(final_chunk)

    return all_chunks


def save_chunks_to_jsonl(
    chunks: List[Document],
    output_file: Path,
) -> None:
    """
    将 chunks 保存为 JSONL 文件。

    该文件主要供 BM25 Retriever 使用。

    每一行格式：

    {
        "page_content": "...",
        "metadata": {...}
    }
    """

    # 自动创建父目录
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,

                    # 防止 metadata 中出现 Path、日期等
                    # 无法直接 JSON 序列化的对象
                    default=str,
                )
                + "\n"
            )


def build_vectorstore(
    documents: List[Document],
    embedding_model: Embeddings,
):
    """
    构建 Milvus 向量库。

    参数：
        documents:
            已完成切分和 metadata 补充的文档列表。

        embedding_model:
            用于生成 Dense Vector 的 Embedding 模型。

    返回：
        Milvus VectorStore 对象。
    """

    print("=" * 100)
    print("开始构建 Milvus 向量库")
    print("=" * 100)

    return build_milvus_vectorstore(
        documents=documents,
        embeddings=embedding_model,
    )


def main() -> None:
    """
    执行知识库完整构建流程。

    流程：
    1. 加载原始 Markdown 文档。
    2. 清洗文本。
    3. 切分文档。
    4. 保存 chunks.jsonl。
    5. 加载 BGE-M3 Embedding 模型。
    6. 构建 Milvus 向量库。
    7. 检查 Milvus Collection 中的数据数量。
    """

    print("=" * 80)
    print("开始构建企业级 RAG 知识库")
    print("=" * 80)

    # 1. 加载所有原始 Markdown 文档
    loader = MarkdownLoader(
        str(RAW_MARKDOWN_DIR),
    )

    raw_documents = loader.load_directory()

    print(
        f"原始文档数量：{len(raw_documents)}"
    )

    if not raw_documents:
        raise RuntimeError(
            f"未在 {RAW_MARKDOWN_DIR} 中加载到 Markdown 文档"
        )

    # 2. 清洗 Markdown 文本
    cleaned_documents = clean_documents(
        raw_documents,
    )

    print("文本清洗完成")

    # 3. 执行 Markdown 标题切分和递归字符切分
    chunks = split_markdown_documents(
        cleaned_documents,
    )

    print(f"切片数量：{len(chunks)}")

    if not chunks:
        raise RuntimeError(
            "文档切分结果为空，无法构建知识库"
        )

    # 4. 保存 JSONL，供 BM25 Retriever 使用
    save_chunks_to_jsonl(
        chunks=chunks,
        output_file=CHUNKS_OUTPUT_FILE,
    )

    print(
        f"切片结果已保存到：{CHUNKS_OUTPUT_FILE}"
    )

    # 5. 加载 BGE-M3 Embedding 模型
    print("正在加载 BGE-M3 Embedding 模型...")

    embedding_model = create_bge_embeddings()

    print("Embedding 模型加载完成")

    # 6. 构建 Milvus 向量库
    vectorstore = build_vectorstore(
        documents=chunks,
        embedding_model=embedding_model,
    )

    # 7. 验证 Milvus 中实际写入的数据数量
    try:
        collection_count = get_milvus_collection_count(
            vectorstore,
        )

        print(
            "Milvus collection count:",
            collection_count,
        )

        if collection_count != len(chunks):
            print(
                "[WARN] Milvus 数据数量与切片数量不一致："
                f"chunks={len(chunks)}，"
                f"milvus={collection_count}"
            )

    except Exception as exc:
        # Count 检查失败不代表构建一定失败，
        # 因此这里只打印警告，不覆盖构建结果
        print(
            "Milvus collection count 读取失败，"
            f"但不影响本次构建结果：{exc}"
        )

    print("=" * 80)
    print("企业级 RAG 知识库构建完成")
    print(f"切片文件：{CHUNKS_OUTPUT_FILE}")
    print("向量数据库：Milvus")
    print("=" * 80)


if __name__ == "__main__":
    main()