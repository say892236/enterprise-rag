# compressor/context_compressor.py

"""
Context Compressor 上下文压缩模块。

作用：
1. 从 Reranker 排序后的文档片段中提取和问题最相关的句子 / 表格行 / 列表项。
2. 减少送入 LLM 的上下文长度。
3. 尽量保留制度问答中的金额、天数、审批、标准等关键信息。

当前版本：
- 使用规则压缩，不额外调用 LLM。
- 如果压缩结果太少，自动回退到原文，避免丢失关键信息。
"""

import re
from typing import List

from langchain_core.documents import Document


class ContextCompressor:
    """
    规则版上下文压缩器。
    """

    # 企业制度问答里的强相关关键词
    BUSINESS_KEYWORDS = [
        "标准",
        "金额",
        "上限",
        "下限",
        "额度",
        "比例",
        "审批",
        "流程",
        "提交",
        "报销",
        "住宿",
        "酒店",
        "出差",
        "差旅",
        "补贴",
        "请假",
        "年假",
        "病假",
        "事假",
        "加班",
        "权限",
        "安全",
        "合规",
        "数据",
        "账号",
    ]

    # 常见城市关键词，用于保留“上海属于一类城市”这类判断依据
    CITY_KEYWORDS = [
        "北京",
        "上海",
        "广州",
        "深圳",
        "杭州",
        "南京",
        "成都",
        "重庆",
        "武汉",
        "西安",
        "苏州",
        "天津",
    ]

    # 明显和当前制度问答无关的后勤 / 外部话题
    OUT_OF_DOMAIN_KEYWORDS = [
        "食堂",
        "天气",
        "股票",
        "电影",
        "外卖",
        "游戏",
        "新闻",
        "彩票",
    ]

    def __init__(
        self,
        max_chars_per_doc: int = 350,
        min_chars_to_keep: int = 80,
    ):
        """
        初始化压缩器。

        参数：
        - max_chars_per_doc：每个文档片段最多保留多少字符。
        - min_chars_to_keep：压缩后少于该长度时，回退使用原文前 max_chars_per_doc 字符。
        """

        self.max_chars_per_doc = max_chars_per_doc
        self.min_chars_to_keep = min_chars_to_keep

    def compress_documents(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        压缩文档列表。

        返回新的 Document 列表，不修改原始 documents。
        """

        compressed_docs = []

        for doc in documents:
            compressed_content = self.compress_text(
                query=query,
                text=doc.page_content,
            )

            # 保留原 metadata，同时标记该片段已被压缩
            metadata = dict(doc.metadata or {})
            metadata["compressed"] = True
            metadata["original_char_count"] = len(doc.page_content)
            metadata["compressed_char_count"] = len(compressed_content)

            compressed_docs.append(
                Document(
                    page_content=compressed_content,
                    metadata=metadata,
                )
            )

        return compressed_docs

    def compress_text(
        self,
        query: str,
        text: str,
    ) -> str:
        """
        压缩单个文本片段。

        策略：
        1. 按行和句子切分文本。
        2. 保留包含问题关键词或业务关键词的内容。
        3. 保留 Markdown 表格行。
        4. 如果压缩结果太少，则回退到原文截断。
        """

        text = text.strip()

        if not text:
            return ""

        query_keywords = self._extract_query_keywords(query)

        units = self._split_text_units(text)

        selected_units = []

        for unit in units:
            if self._should_keep_unit(
                unit=unit,
                query_keywords=query_keywords,
            ):
                selected_units.append(unit)

        compressed_text = "\n".join(selected_units).strip()

        # 如果压缩后内容太少，说明规则没抓住重点，回退原文前 max_chars_per_doc 字符
        if len(compressed_text) < self.min_chars_to_keep:
            compressed_text = text[: self.max_chars_per_doc]

        # 限制每个 doc 最长字符数，避免上下文膨胀
        return compressed_text[: self.max_chars_per_doc]

    def _extract_query_keywords(
        self,
        query: str,
    ) -> List[str]:
        """
        从用户问题中提取简单关键词。

        当前策略：
        - 保留长度 >= 2 的中文连续片段。
        - 保留英文、数字、制度编号。
        """

        keywords = []

        # 中文片段
        chinese_parts = re.findall(r"[\u4e00-\u9fa5]{2,}", query)
        keywords.extend(chinese_parts)

        # 英文 / 数字 / 编号
        english_parts = re.findall(r"[A-Za-z0-9_\-\.]{2,}", query)
        keywords.extend(english_parts)

        # 追加企业制度常见关键词
        for keyword in self.BUSINESS_KEYWORDS:
            if keyword in query:
                keywords.append(keyword)

        # 追加城市关键词
        # 例如问题里有“上海”，必须保留下文中的“上海属于一类城市”
        for keyword in self.CITY_KEYWORDS:
            if keyword in query:
                keywords.append(keyword)

        # 去重但保留顺序
        deduped_keywords = []

        for keyword in keywords:
            if keyword not in deduped_keywords:
                deduped_keywords.append(keyword)

        return deduped_keywords

    def _split_text_units(
        self,
        text: str,
    ) -> List[str]:
        """
        将文本切分成候选单元。

        兼容：
        - Markdown 表格行
        - 列表项
        - 普通中文句子
        """

        units = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            # Markdown 表格行、列表项直接作为一个单元
            if line.startswith("|") or line.startswith("-") or line.startswith("*"):
                units.append(line)
                continue

            # 普通段落再按中文标点切分
            sentence_parts = re.split(r"[。；;！!？?]", line)

            for sentence in sentence_parts:
                sentence = sentence.strip()
                if sentence:
                    units.append(sentence)

        return units

    def _should_keep_unit(
            self,
            unit: str,
            query_keywords: List[str],
    ) -> bool:
        """
        判断一个文本单元是否应该保留。
        """

        # 1. 明显无关话题优先过滤。
        # 例如当前问的是差旅报销，就不应该保留“食堂开放时间”。
        for keyword in self.OUT_OF_DOMAIN_KEYWORDS:
            if keyword in unit and keyword not in query_keywords:
                return False

        # 2. Markdown 表格行通常包含制度标准，优先保留。
        if unit.startswith("|"):
            return True

        # 3. 包含问题关键词则保留。
        # 例如 query 里有“上海”，则保留“上海属于一类城市”。
        for keyword in query_keywords:
            if keyword and keyword in unit:
                return True

        # 4. 保留城市分类依据。
        # 例如“一类城市 / 二类城市 / 三类城市”通常是金额标准的前置判断条件。
        if re.search(r"[一二三四五六七八九十]+类城市", unit):
            return True

        # 5. 包含业务强关键词则保留。
        for keyword in self.BUSINESS_KEYWORDS:
            if keyword in unit:
                return True

        # 6. 包含金额、百分比、天数等规则信息则保留。
        if re.search(r"\d+(\.\d+)?\s*(元|天|日|%|％|小时|工作日|次|晚)", unit):
            return True

        return False