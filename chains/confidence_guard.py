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

# chains/confidence_guard.py

# 业务领域关键词。
# 用来避免“语义相关但字面重叠少”的问题被误拒答。
DOMAIN_KEYWORDS = {
    "travel": [
        "出差",
        "差旅",
        "住宿",
        "酒店",
        "五星级",
        "城市",
        "一类城市",
        "二类城市",
        "三类城市",
        "交通",
        "机票",
        "火车",
        "高铁",
    ],
    "reimbursement": [
        "报销",
        "发票",
        "费用",
        "审批",
        "付款",
        "提交",
        "凭证",
    ],
    "leave": [
        "请假",
        "年假",
        "病假",
        "事假",
        "调休",
        "假期",
        "申请",
    ],
    "security": [
        "安全",
        "权限",
        "账号",
        "密码",
        "数据",
        "合规",
        "泄露",
    ],
    "hr": [
        "员工",
        "薪资",
        "工资",
        "绩效",
        "职级",
        "入职",
        "离职",
    ],
}


# 文档 source 与业务领域的映射。
# 用来判断检索结果是否属于用户问题对应的业务领域。
SOURCE_DOMAIN_MAP = {
    "travel_policy.md": "travel",
    "reimbursement_policy.md": "reimbursement",
    "leave_policy.md": "leave",
    "it_security_policy.md": "security",
    "hr_salary_policy.md": "hr",
    "employee_handbook.md": "hr",
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

        判断逻辑：
        1. 没有检索结果：拒答。
        2. 用户问题没有有效关键词：拒答。
        3. 问题关键词和 TopN 文档有足够重叠：放行。
        4. 如果关键词重叠不足，但问题和文档属于同一业务领域：放行。
        5. 否则拒答。
        """

        # 1. 没有检索结果，直接拒答
        if not documents:
            return ConfidenceResult(
                passed=False,
                reason="没有检索到任何相关文档",
                query_tokens=[],
                matched_tokens=[],
                overlap_count=0,
            )

        # 2. 提取问题关键词
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return ConfidenceResult(
                passed=False,
                reason="用户问题缺少有效关键词",
                query_tokens=[],
                matched_tokens=[],
                overlap_count=0,
            )

        # 3. 只检查 TopN 个最终文档
        checked_docs = documents[: self.check_top_n]

        combined_text = "\n".join(
            doc.page_content
            for doc in checked_docs
        )

        document_tokens = self._tokenize(combined_text)

        matched_tokens = sorted(query_tokens & document_tokens)
        overlap_count = len(matched_tokens)

        # 判断用户问题与检索文档是否属于同一业务领域
        domain_match = self.has_business_domain_match(
            question=query,
            documents=checked_docs,
        )


        for index, doc in enumerate(checked_docs, start=1):
            metadata = doc.metadata or {}

            print(
                f"doc {index}:",
                "source=", metadata.get("source"),
                "header_1=", metadata.get("header_1"),
                "header_2=", metadata.get("header_2"),
                "rerank_score=", metadata.get("rerank_score"),
            )

            print(
                "content:",
                doc.page_content[:120].replace("\n", " "),
            )

        print("=" * 80)

        # 关键词有足够重叠，直接放行
        if overlap_count >= self.min_overlap_count:
            return ConfidenceResult(
                passed=True,
                reason="检索结果通过关键词重叠检查",
                query_tokens=sorted(query_tokens),
                matched_tokens=matched_tokens,
                overlap_count=overlap_count,
            )

        # 关键词重叠不足，但问题和文档属于同一业务领域，也放行
        if domain_match:
            return ConfidenceResult(
                passed=True,
                reason="问题与检索文档属于同一业务领域，放行低重叠问题",
                query_tokens=sorted(query_tokens),
                matched_tokens=matched_tokens,
                overlap_count=overlap_count,
            )

        # 关键词和业务领域都不匹配，拒答
        return ConfidenceResult(
            passed=False,
            reason="问题关键词与检索结果缺少有效重叠，且业务领域不匹配",
            query_tokens=sorted(query_tokens),
            matched_tokens=matched_tokens,
            overlap_count=overlap_count,
        )

    def get_question_domains(self, question: str) -> set[str]:
        """
        根据问题命中的业务关键词，判断问题属于哪些业务领域。
        """

        matched_domains = set()

        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword in question for keyword in keywords):
                matched_domains.add(domain)

        return matched_domains

    def get_document_domain(self, doc: Document) -> str | None:
        """
        根据 Document metadata.source 判断文档属于哪个业务领域。
        """

        metadata = doc.metadata or {}
        source = metadata.get("source", "")

        # 兼容 source 为文件名或完整路径的情况
        source_name = str(source).replace("\\", "/").split("/")[-1]

        return SOURCE_DOMAIN_MAP.get(source_name)

    def has_business_domain_match(self, question: str, documents: List[Document]) -> bool:
        """
        判断问题和检索文档是否属于同一业务领域。

        作用：
        1. 减少低置信度规则误拒答。
        2. 允许“语义相关但字面重叠少”的问题继续进入 LLM。
        """

        question_domains = self.get_question_domains(question)

        if not question_domains:
            return False

        document_domains = set()

        for doc in documents:
            domain = self.get_document_domain(doc)
            if domain:
                document_domains.add(domain)

        return bool(question_domains & document_domains)