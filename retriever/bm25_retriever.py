# retriever/bm25_retriever.py

"""
BM25 Retriever 模块。

作用：
1. 从 data/processed/chunks.jsonl 读取已经切好的 chunks。
2. 使用 jieba 对中文文本分词。
3. 使用 BM25 算法做关键词检索。
4. 返回 LangChain Document 列表，方便后面和 Chroma Retriever 融合。

适合解决：
- 文档编号检索，例如 FIN-TR-2026-01
- 条款编号检索，例如 FAQ-003、3.2、5.1
- 精确金额检索，例如 200 元、600 元、10 个工作日
- 专有名词、部门名、制度编号等关键词匹配
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Retriever:
    """
    企业级 RAG 项目的 BM25 检索器。

    当前版本特点：
    - 从 chunks.jsonl 加载切片数据
    - 使用 jieba 做中文分词
    - 使用 rank-bm25 做关键词检索
    - 返回 LangChain Document
    - 将 bm25_score 写入 metadata，方便调试
    """

    def __init__(
        self,
        chunks_file: str = "data/processed/chunks.jsonl",
        top_k: int = 5,
    ):
        """
        初始化 BM25Retriever。

        参数：
        chunks_file:
            已经切好的 chunks 文件路径。
            这里默认读取 data/processed/chunks.jsonl。

        top_k:
            默认返回前几个结果。
        """

        self.chunks_file = Path(chunks_file)
        self.top_k = top_k

        # 1. 从 jsonl 文件加载 Document 列表
        self.documents = self._load_documents(self.chunks_file)

        # 2. 将每个 Document 的正文分词
        # BM25 不能直接吃原始字符串，它需要 token list
        self.tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in self.documents
        ]

        # 3. 初始化 BM25 模型
        # BM25Okapi 会基于 tokenized_corpus 计算词频、逆文档频率等统计信息
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_documents(self, file_path: Path) -> List[Document]:
        """
        从 chunks.jsonl 加载 Document。

        兼容几种常见 jsonl 格式：

        格式 1：
        {
            "page_content": "...",
            "metadata": {...}
        }

        格式 2：
        {
            "content": "...",
            "metadata": {...}
        }

        格式 3：
        {
            "text": "...",
            "metadata": {...}
        }

        如果你的 chunks.jsonl 字段不同，只需要改这里。
        """

        if not file_path.exists():
            raise FileNotFoundError(f"chunks 文件不存在: {file_path}")

        documents: List[Document] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                # 跳过空行
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"第 {line_number} 行不是合法 JSON: {line}"
                    ) from e

                # 兼容不同字段名
                page_content = (
                    data.get("page_content")
                    or data.get("content")
                    or data.get("text")
                )

                if not page_content:
                    raise ValueError(
                        f"第 {line_number} 行缺少 page_content/content/text 字段"
                    )

                # metadata 推荐单独放在 metadata 字段里
                metadata = data.get("metadata", {})

                # 如果你的 chunks.jsonl 是扁平结构，例如：
                # {"page_content": "...", "source": "...", "header_2": "..."}
                # 那么这里会把除正文以外的字段自动放进 metadata。
                if not metadata:
                    metadata = {
                        key: value
                        for key, value in data.items()
                        if key not in {"page_content", "content", "text"}
                    }

                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=metadata,
                    )
                )

        if not documents:
            raise ValueError(f"chunks 文件为空: {file_path}")

        return documents

    def _tokenize(self, text: str) -> List[str]:
        """
        中文 + 英文 + 数字混合分词。

        处理逻辑：
        1. 转小写，方便英文关键词匹配。
        2. 保留中文、英文、数字。
        3. 使用 jieba 分词。
        4. 去掉空白 token。

        注意：
        BM25 的效果很依赖分词。
        企业项目后面可以加入：
        - 自定义词典
        - 停用词
        - 同义词扩展
        """

        # 英文统一转小写
        text = text.lower()

        # 把一些符号替换为空格，但保留中文、英文、数字、常见编号符号
        # 例如 FIN-TR-2026-01、FAQ-003、3.2 这类编号尽量保留
        text = re.sub(
            r"[^\u4e00-\u9fa5a-z0-9\-\._]+",
            " ",
            text,
        )

        # jieba 分词
        tokens = jieba.lcut(text)

        # 去掉空白 token
        tokens = [
            token.strip()
            for token in tokens
            if token.strip()
        ]

        return tokens

    def match_metadata_filter(metadata: dict, metadata_filter: dict | None) -> bool:
        """
        判断文档 metadata 是否满足过滤条件。

        当前支持：
        1. 精确匹配：
           {"source": "travel_policy.md"}

        2. $in 匹配：
           {"source": {"$in": ["travel_policy.md", "employee_handbook.md"]}}
        """

        if not metadata_filter:
            return True

        for key, expected_value in metadata_filter.items():
            actual_value = metadata.get(key)

            # 支持 {"source": {"$in": [...]}}
            if isinstance(expected_value, dict) and "$in" in expected_value:
                if actual_value not in expected_value["$in"]:
                    return False

            # 支持 {"source": "travel_policy.md"}
            else:
                if actual_value != expected_value:
                    return False

        return True

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        执行 BM25 检索。

        参数：
        query:
            用户问题或改写后的检索问题。

        top_k:
            返回前几个结果。

        metadata_filter:
            metadata 过滤条件。
            例如：
            {"source": "travel_policy.md"}

        返回：
            List[Document]
        """

        k = top_k or self.top_k

        # 1. 对 query 做同样的分词
        tokenized_query = self._tokenize(query)

        # 2. BM25 给语料库里每个 chunk 打分
        scores = self.bm25.get_scores(tokenized_query)

        # 3. 将 document 和 score 绑在一起
        scored_docs = []

        for doc, score in zip(self.documents, scores):
            # 如果传了 metadata_filter，就先做过滤
            if metadata_filter:
                matched = True

                for key, value in metadata_filter.items():
                    if doc.metadata.get(key) != value:
                        matched = False
                        break

                if not matched:
                    continue

            # 复制一份 metadata，避免污染原始 Document
            metadata = dict(doc.metadata)
            metadata["bm25_score"] = float(score)

            scored_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        # 4. BM25 分数越高越相关，所以 reverse=True
        scored_docs.sort(
            key=lambda doc: doc.metadata["bm25_score"],
            reverse=True,
        )

        # 5. 返回 top_k
        return scored_docs[:k]

    def count(self) -> int:
        """
        返回 BM25 当前加载了多少个 chunks。
        """

        return len(self.documents)