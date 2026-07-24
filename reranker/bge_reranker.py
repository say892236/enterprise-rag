from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document



# 规则配置
@dataclass(frozen=True)
class TopicBoostRule:
    """
    主题加权规则。

    含义：
    如果用户问题命中 query_words，
    并且文档标题命中 header_words，
    就给该 chunk 增加 boost 分。

    配置规则
      ↓
    执行规则
      ↓
    记录加权原因
      ↓
    算最终分
    """

    name: str
    query_words: Tuple[str, ...]
    header_words: Tuple[str, ...]
    boost: float


# 泛制度问题关键词
BROAD_RULE_WORDS = (
    "标准",
    "制度",
    "规则",
    "流程",
    "要求",
    "规定",
)


# 酒店 / 住宿类问题关键词
HOTEL_WORDS = (
    "住宿",
    "酒店",
    "宾馆",
    "五星级",
    "几星级",
)


# 判断类问题关键词
JUDGEMENT_WORDS = (
    "可以",
    "能不能",
    "能否",
    "是否",
    "可不可以",
)


# 通用主题加权规则配置
TOPIC_BOOST_RULES = (
    TopicBoostRule(
        name="住宿类",
        query_words=("住宿", "酒店", "宾馆", "五星级"),
        header_words=("住宿",),
        boost=0.7,
    ),
    TopicBoostRule(
        name="交通类",
        query_words=("交通", "高铁", "飞机", "打车", "出租车", "市内"),
        header_words=("交通",),
        boost=0.7,
    ),
    TopicBoostRule(
        name="报销类",
        query_words=("报销", "发票", "费用", "付款", "咖啡"),
        header_words=("报销", "范围", "费用"),
        boost=0.7,
    ),
    TopicBoostRule(
        name="审批类",
        query_words=("审批", "申请", "负责人"),
        header_words=("审批", "流程"),
        boost=0.7,
    ),
    TopicBoostRule(
        name="补贴类",
        query_words=("补贴", "餐饮", "通讯"),
        header_words=("补贴",),
        boost=0.7,
    ),
    TopicBoostRule(
        name="请假类",
        query_words=("请假", "年假", "病假", "事假"),
        header_words=("请假",),
        boost=0.7,
    ),
    TopicBoostRule(
        name="考勤类",
        query_words=("考勤", "迟到", "打卡", "上班"),
        header_words=("考勤",),
        boost=0.7,
    ),
)


class BGEReranker:
    """
    企业级 BGE Reranker。

    流程：
    1. 使用 bge-reranker-v2-m3 给 query 和 chunk 打语义相关分。
    2. 根据 metadata 和企业规则加减分。
    3. final_score = raw_rerank_score + rerank_boost。
    4. 按 final_score 从高到低排序。
    """

   #模型部分
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        max_length: int = 512,
        use_business_boost: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.use_business_boost = use_business_boost

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _contains_any(text: str, words: Tuple[str, ...]) -> bool:
        """
        判断 text 中是否包含 words 中任意一个词。
        """

        return any(word in text for word in words)

    def _build_passage(self, doc: Document) -> str:
        """
        构造给 Reranker 的 passage。

        注意：
        这里不只传正文，也传 source / header。
        因为企业 RAG 中标题和来源本身就是重要语义。
        """

        source = doc.metadata.get("source", "")
        header_1 = doc.metadata.get("header_1", "")
        header_2 = doc.metadata.get("header_2", "")
        chunk_id = doc.metadata.get("chunk_id", "")

        return f"""
文档来源：{source}
一级标题：{header_1}
二级标题：{header_2}
切片编号：{chunk_id}

正文内容：
{doc.page_content}
"""

    # 业务规则部分
    def _apply_broad_rule_boost(
        self,
        query: str,
        header_2: str,
    ) -> tuple[float, List[str]]:
        """
        泛制度类问题处理。

        例如：
        - 员工出差住宿标准是什么？
        - 报销流程是什么？
        - 请假制度是什么？

        这类问题优先正式制度章节，FAQ 适当降权。
        """

        boost = 0.0
        reasons = []

        is_broad_rule_question = self._contains_any(
            query,
            BROAD_RULE_WORDS,
        )

        if is_broad_rule_question and "FAQ" in header_2:
            boost -= 0.6
            reasons.append("泛制度问题：FAQ 降权 -0.6")

        return boost, reasons

    def _apply_hotel_judgement_boost(
        self,
        query: str,
        header_2: str,
    ) -> tuple[float, List[str]]:
        """
        酒店 / 住宿判断类问题处理。

        例如：
        - 出差可以住五星级酒店吗？
        - 出差能不能住酒店？
        - 酒店超标能报销吗？

        这类问题更适合优先参考正式住宿标准章节。
        """

        boost = 0.0
        reasons = []

        is_hotel_question = self._contains_any(
            query,
            HOTEL_WORDS,
        )

        is_judgement_question = self._contains_any(
            query,
            JUDGEMENT_WORDS,
        )

        if not is_hotel_question:
            return boost, reasons

        if "住宿标准" in header_2:
            boost += 1.5
            reasons.append("酒店/住宿问题：住宿标准章节加权 +1.5")

        if "FAQ" in header_2 and is_judgement_question:
            boost -= 0.4
            reasons.append("酒店判断问题：FAQ 降权 -0.4")

        return boost, reasons

    def _apply_topic_boost(
        self,
        query: str,
        header_2: str,
    ) -> tuple[float, List[str]]:
        """
        通用主题匹配加权。

        例如：
        - 问题里有“住宿”，标题里有“住宿” → 加权。
        - 问题里有“报销”，标题里有“报销范围” → 加权。
        """

        for rule in TOPIC_BOOST_RULES:
            query_hit = self._contains_any(
                query,
                rule.query_words,
            )

            header_hit = self._contains_any(
                header_2,
                rule.header_words,
            )

            if query_hit and header_hit:
                reason = f"主题匹配：{rule.name} +{rule.boost}"
                return rule.boost, [reason]

        return 0.0, []

    def _business_boost(
        self,
        query: str,
        doc: Document,
    ) -> tuple[float, List[str]]:
        """
        企业业务加权入口。

        返回：
        - boost: 总加权分
        - reasons: 加权原因列表，方便调试
        """

        header_2 = doc.metadata.get("header_2") or ""

        total_boost = 0.0
        all_reasons = []

        boost, reasons = self._apply_broad_rule_boost(
            query=query,
            header_2=header_2,
        )
        total_boost += boost
        all_reasons.extend(reasons)

        boost, reasons = self._apply_hotel_judgement_boost(
            query=query,
            header_2=header_2,
        )
        total_boost += boost
        all_reasons.extend(reasons)

        boost, reasons = self._apply_topic_boost(
            query=query,
            header_2=header_2,
        )
        total_boost += boost
        all_reasons.extend(reasons)

        return total_boost, all_reasons

    def _compute_raw_scores(
        self,
        query: str,
        documents: List[Document],
    ) -> List[float]:
        """
        使用 BGE Reranker 计算原始相关性分数。
        """

        pairs = [
            [
                query,
                self._build_passage(doc),
            ]
            for doc in documents
        ]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            scores = self.model(
                **inputs,
                return_dict=True,
            ).logits.view(-1).float()

        return scores.cpu().tolist()


    #对外入口
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
    ) -> List[Document]:
        """
        对 Retriever 召回的 documents 进行重排。
        """

        if not documents:
            return []

        raw_scores = self._compute_raw_scores(
            query=query,
            documents=documents,
        )

        reranked_docs = []

        for doc, raw_score in zip(documents, raw_scores):
            boost = 0.0
            boost_reasons = []

            if self.use_business_boost:
                boost, boost_reasons = self._business_boost(
                    query=query,
                    doc=doc,
                )

            final_score = raw_score + boost

            doc.metadata["raw_rerank_score"] = raw_score
            doc.metadata["rerank_boost"] = boost
            doc.metadata["rerank_boost_reasons"] = boost_reasons
            doc.metadata["rerank_score"] = final_score

            reranked_docs.append(doc)

        reranked_docs.sort(
            key=lambda doc: doc.metadata["rerank_score"],
            reverse=True,
        )

        return reranked_docs[:top_k]