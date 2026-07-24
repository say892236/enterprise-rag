# chains/confidence_guard.py

"""
RAG 检索置信度判断模块。

作用：
1. 判断检索结果是否和用户问题相关。
2. 如果明显无关，则提前拒答，避免 LLM 根据无关上下文硬编。
3. 第一版采用关键词重叠规则，不直接依赖 reranker 分数。
"""

import re
from dataclasses import dataclass
from typing import List, Set

import jieba
from langchain_core.documents import Document


# 常见无意义词，避免“公司、员工、什么”等词造成误判
STOP_WORDS = {
    # 通用泛词
    "公司",
    "员工",
    "制度",
    "规定",
    "政策",
    "标准",
    "要求",
    "信息",
    "内容",
    "相关",
    "相关信息",

    # 问句泛词
    "什么",
    "多少",
    "是否",
    "可以",
    "需要",
    "如何",
    "怎么",
    "有没有",
    "有无",
    "这个",
    "那个",
    "一下",
    "请问",
    "如果",
    "进行",
    "管理",

    # 中文语气 / 虚词
    "的",
    "了",
    "吗",
    "呢",
    "是",
    "在",
    "有",
    "和",
    "与",
    "及",
    "或",

    # 时间类泛词
    "时间",
    "工作时间",
    "开放",
    "开放时间",
    "开门",
    "几点",
    "几点钟",
    "什么时候",
}

@dataclass
class ConfidenceResult:
    """
    置信度判断结果。
    """

    passed: bool
    reason: str
    query_tokens: List[str]
    matched_tokens: List[str]
    overlap_count: int

    def to_dict(self) -> dict:
        """
        转成普通 dict，方便写入 RAG 返回结果或日志。
        """

        return {
            "passed": self.passed,
            "reason": self.reason,
            "query_tokens": self.query_tokens,
            "matched_tokens": self.matched_tokens,
            "overlap_count": self.overlap_count,
        }


class RetrievalConfidenceGuard:
    """
    检索置信度保护器。

    判断逻辑：
    1. 没有检索结果：拒答。
    2. 用户问题没有有效关键词：拒答。
    3. 问题关键词和 TopN 文档没有任何重叠：拒答。
    4. 否则允许进入 LLM 生成阶段。
    """

    def __init__(
        self,
        min_overlap_count: int = 1,
        check_top_n: int = 3,
    ):
        """
        初始化置信度保护器。

        参数：
            min_overlap_count: 至少需要多少个关键词命中。
            check_top_n: 检查前几个最终文档。
        """

        self.min_overlap_count = min_overlap_count
        self.check_top_n = check_top_n

    def _tokenize(self, text: str) -> Set[str]:
        """
        对中文问题和文档进行简单分词。

        说明：
        - 使用 jieba 处理中文。
        - 保留英文、数字、编号类 token。
        - 过滤常见停用词和单字噪声。
        """

        if not text:
            return set()

        text = text.lower()

        words = jieba.lcut(text)

        tokens = set()

        for word in words:
            token = word.strip().lower()

            if not token:
                continue

            if token in STOP_WORDS:
                continue

            # 去掉纯标点符号
            if re.fullmatch(r"\W+", token):
                continue

            # 去掉大多数单字中文噪声
            # 但保留英文、数字、编号类内容
            if len(token) == 1 and not re.fullmatch(r"[a-z0-9]", token):
                continue

            tokens.add(token)

        return tokens

    def check(
        self,
        query: str,
        documents: List[Document],
    ) -> ConfidenceResult:
        """
        判断检索结果是否可信。

        参数：
            query: 用于检索的问题，通常是 rewritten_question。
            documents: 最终进入 LLM 的文档，一般是 rerank 后的 final_docs。

        返回：
            ConfidenceResult
        """

        if not documents:
            return ConfidenceResult(
                passed=False,
                reason="没有检索到任何相关文档",
                query_tokens=[],
                matched_tokens=[],
                overlap_count=0,
            )

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return ConfidenceResult(
                passed=False,
                reason="用户问题缺少有效关键词",
                query_tokens=[],
                matched_tokens=[],
                overlap_count=0,
            )

        checked_docs = documents[: self.check_top_n]

        combined_text = "\n".join(
            doc.page_content
            for doc in checked_docs
        )

        document_tokens = self._tokenize(combined_text)

        matched_tokens = sorted(query_tokens & document_tokens)

        if len(matched_tokens) < self.min_overlap_count:
            return ConfidenceResult(
                passed=False,
                reason="问题关键词与检索结果缺少有效重叠",
                query_tokens=sorted(query_tokens),
                matched_tokens=matched_tokens,
                overlap_count=len(matched_tokens),
            )

        return ConfidenceResult(
            passed=True,
            reason="检索结果通过置信度检查",
            query_tokens=sorted(query_tokens),
            matched_tokens=matched_tokens,
            overlap_count=len(matched_tokens),
        )