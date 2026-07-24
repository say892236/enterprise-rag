from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class MarkdownSplitter:
    """
    企业 Markdown 文档切分器

    作用：
    先利用 Markdown 标题拿到章节结构，再用递归切分控制 chunk 长度，最后给每个 chunk 加上可追踪的 metadata。

    两阶段切分:
    1. MarkdownHeaderTextSplitter:
       按 Markdown 标题结构切分，并提取标题 metadata

    2. RecursiveCharacterTextSplitter:
       对过长章节继续按长度递归切分

    最终效果:
    - 保留 source / file_path / file_type
    - 新增 header_1 / header_2
    - 新增 chunk_index / chunk_id / char_count
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

        # 第一阶段：按 Markdown 标题切分
        self.header_splitter = MarkdownHeaderTextSplitter(
           # 遇到  # 标题，把标题内容放进 metadata["header_1"]
           # 遇到  ## 标题，把标题内容放进 metadata["header_2"]
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                # ("###", "header_3"),
                # ("####", "header_4"),
            ],
            strip_headers=False,   #不要把标题从正文里删除
        )

        # 第二阶段：对长章节继续切分
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,    #len计算文本长度 后面改成 token
            keep_separator=True,    #切分后保留分隔符,如果不保留，文本可能变得不自然。
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                "，",
                "",
            ],
        )

    # 输入一组原始Document，输出一组 chunk Document
    def split_documents(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        """
        切分 Document 列表，并保留、增强 metadata。
        """
        if not documents:
            return []    #如果没有传任何文档，就直接返回空列表。这样不会报错。

        header_documents: list[Document] = []   #保存第一阶段标题切分后的 Document。

        # 1. 先按 Markdown 标题切分
        for document in documents:
            section_docs = self.header_splitter.split_text(
                document.page_content
            )

            for section_doc in section_docs:
                # 合并原始 metadata 和标题 metadata
                merged_metadata = {
                    **document.metadata,
                    **section_doc.metadata,
                }

                header_documents.append(
                    Document(
                        page_content=section_doc.page_content,
                        metadata=merged_metadata,
                    )
                )

        # 2. 再按 chunk_size 递归切分
        chunks = self.text_splitter.split_documents(
            header_documents
        )

        # 3. 给每个切片补充 chunk 信息
        source_counters: dict[str, int] = {}    #记录每个文件切到第几个 chunk


        #遍历所有切片
        for chunk in chunks:
            #获取来源文件名，从 metadata 里取 source如果没有 source，就用 unknown
            source = str(
                chunk.metadata.get("source", "unknown")
            )
            #如果第一次遇到 source = "travel_policy.md"，
            # source_counters.get("travel_policy.md", 0)
            chunk_index = source_counters.get(source, 0)
            #这个source已经处理了一个chunk下次再遇到它，序号就加1
            source_counters[source] = chunk_index + 1

            chunk.metadata.update(
                {
                    "chunk_index": chunk_index,  #表示这个 chunk 是当前源文件的第几个切片。
                    "chunk_id": f"{source}:{chunk_index}",
                    "char_count": len(chunk.page_content),  #记录当前 chunk 字符数,调试切片质量
                }
            )

        return chunks