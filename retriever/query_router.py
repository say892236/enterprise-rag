# retriever/query_router.py

"""
Query Router 查询路由模块。

作用：
1. 根据用户问题判断查询类型。
2. 为不同类型的问题选择不同检索策略。
3. 避免所有问题都无脑走同一套 Rewrite + Hybrid + Reranker。
"""

import re
from dataclasses import dataclass
from typing import Literal


QueryType = Literal[
    "exact_code_query",      # 编号、文件名、制度ID等精确查询
    "structured_query",     # 金额、天数、标准、流程等结构化制度查询
    "semantic_query",       # 普通语义查询
    "unknown_query",        # 明显知识库外问题
]


@dataclass
class QueryRouteResult:
    """
    查询路由结果。
    """

    query_type: QueryType
    enable_rewrite: bool
    enable_rerank: bool
    reason: str


class QueryRouter:
    """
    规则版 Query Router。

    当前先不用 LLM，避免增加额外耗时和不稳定性。
    """

    # 制度编号 / 文档编号 / 文件名特征
    CODE_PATTERNS = [
        r"\b[A-Z]{2,}-[A-Z]{2,}-\d{4}-\d{2}\b",   # 例如 HR-LV-2026-01
        r"\b[A-Z]{2,}-\d{4}-\d{2}\b",             # 例如 RAG-2026-01
        r"\.md\b",                                # 例如 travel_policy.md
    ]

    # 结构化制度查询关键词
    STRUCTURED_KEYWORDS = [
        "多少",
        "多久",
        "几天",
        "几次",
        "金额",
        "标准",
        "上限",
        "下限",
        "额度",
        "比例",
        "流程",
        "审批",
        "提交",
        "报销",
        "住宿",
        "补贴",
        "请假",
        "年假",
        "病假",
        "出差",
    ]

    # 明显知识库外的问题关键词
    OUT_OF_DOMAIN_KEYWORDS = [
        "食堂",
        "天气",
        "股票",
        "电影",
        "外卖",
        "游戏",
        "新闻",
        "汇率",
        "彩票",
    ]

    def route(
        self,
        question: str,
    ) -> QueryRouteResult:
        """
        根据用户问题返回查询路由结果。
        """

        question = question.strip()

        # 1. 编号 / 文件名类问题：优先精确匹配，不强依赖 rewrite
        if self._is_exact_code_query(question):
            return QueryRouteResult(
                query_type="exact_code_query",
                enable_rewrite=False,
                enable_rerank=True,
                reason="问题包含制度编号、文档编号或文件名，适合精确匹配优先。",
            )

        # 2. 明显知识库外问题：仍允许检索，但不做 rewrite，减少无意义改写
        if self._is_unknown_query(question):
            return QueryRouteResult(
                query_type="unknown_query",
                enable_rewrite=False,
                enable_rerank=True,
                reason="问题疑似不属于企业制度知识库，关闭查询改写以减少误召回。",
            )

        # 3. 金额 / 天数 / 流程 / 标准类问题：保留 rewrite + rerank
        if self._is_structured_query(question):
            return QueryRouteResult(
                query_type="structured_query",
                enable_rewrite=True,
                enable_rerank=True,
                reason="问题涉及金额、天数、流程或制度标准，适合 Rewrite + Hybrid + Reranker。",
            )

        # 4. 默认语义问题
        return QueryRouteResult(
            query_type="semantic_query",
            enable_rewrite=True,
            enable_rerank=True,
            reason="普通语义查询，使用默认 Rewrite + Hybrid + Reranker。",
        )

    def _is_exact_code_query(
        self,
        question: str,
    ) -> bool:
        """
        判断是否为编号 / 文件名类精确查询。
        """

        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, question, flags=re.IGNORECASE):
                return True

        return False

    def _is_structured_query(
        self,
        question: str,
    ) -> bool:
        """
        判断是否为结构化制度查询。
        """

        return any(
            keyword in question
            for keyword in self.STRUCTURED_KEYWORDS
        )

    def _is_unknown_query(
        self,
        question: str,
    ) -> bool:
        """
        判断是否明显不属于企业制度知识库。
        """

        return any(
            keyword in question
            for keyword in self.OUT_OF_DOMAIN_KEYWORDS
        )