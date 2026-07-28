# api/schemas/chat.py

"""
Chat 接口请求和响应结构。
"""


from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

class ChatRequest(BaseModel):
    """
    /chat 请求体。

    负责接收用户问题，以及控制检索、重排、metadata 过滤等参数。
    """

    # 禁止传入未定义字段
    # 例如传入 unknown_field 时，接口会直接返回 422
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                # 示例 1：正常访问公共文档
                {
                    "question": "员工出差住宿标准是什么？",
                    "session_id": "请先调用 /chat/session/new 获取",
                    "retrieve_top_k": 6,
                    "rerank_top_k": 3,
                    "enable_rewrite": True,
                    "enable_rerank": True,
                    "enable_compression": True,
                    "metadata_filter": {
                        "source": "差旅住宿与交通标准.md",
                    },
                },

                # 示例 2：知识库无答案问题
                {
                    "question": "公司食堂几点开门？",
                    "session_id": "请先调用 /chat/session/new 获取",
                    "retrieve_top_k": 6,
                    "rerank_top_k": 3,
                    "enable_rewrite": True,
                    "enable_rerank": True,
                    "enable_compression": True,
                    "metadata_filter": None,
                },

                # 示例 3：HR 私有文档权限测试
                #
                # employee 用户调用时应返回 PERMISSION_DENIED；
                # hr 或 admin 用户调用时应正常访问。
                {
                    "question": "公司的薪资调整流程是什么？",
                    "session_id": "请先调用 /chat/session/new 获取",
                    "retrieve_top_k": 6,
                    "rerank_top_k": 3,
                    "enable_rewrite": True,
                    "enable_rerank": True,
                    "enable_compression": True,
                    "metadata_filter": {
                        "source": "薪酬体系与调薪管理制度.md",
                    },
                },
            ]
        },
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="用户问题，不能为空，最长 500 字",
        examples=["员工出差住宿标准是什么？"],
    )

    retrieve_top_k: int = Field(
        default=10,
        ge=1,
        le=20,
        description="初始检索返回的候选文档数量，范围 1-20",
    )

    rerank_top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Reranker 重排后保留的文档数量，范围 1-10",
    )

    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="metadata 过滤条件，例如 {'source': 'travel_policy.md'}",
    )

    enable_rewrite: bool = Field(
        default=True,
        description="是否启用查询改写",
    )

    enable_rerank: bool = Field(
        default=True,
        description="是否启用 Reranker 重排",
    )

    enable_compression: bool = Field(
        default=True,
        description="是否启用 Context Compression 上下文压缩",
    )
    session_id: str = Field(
        default="default",
        description="当前聊天会话 ID",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """
        校验用户问题。

        处理：
        1. 去掉前后空格。
        2. 防止传入全空白字符串。
        """

        question = value.strip()

        if not question:
            raise ValueError("question 不能为空")

        return question

    @model_validator(mode="after")
    def validate_top_k(self):
        """
        校验 retrieve_top_k 和 rerank_top_k 的关系。

        rerank_top_k 不应该大于 retrieve_top_k。
        因为重排是在初始检索结果里再筛选。
        """

        if self.rerank_top_k > self.retrieve_top_k:
            raise ValueError("rerank_top_k 不能大于 retrieve_top_k")

        return self


class SourceItem(BaseModel):
        """
        RAG 引用来源信息。

        page_content / content 用于：
        1. 前端展示引用片段
        2. 评估 Faithfulness 忠实度
        3. 调试检索结果是否真正支持答案
        """

        # 基础溯源信息
        source: Optional[str] = None
        file_path: Optional[str] = None
        file_type: Optional[str] = None
        header_1: Optional[str] = None
        header_2: Optional[str] = None
        chunk_id: Optional[str] = None
        chunk_index: Optional[int] = None

        # 文档正文
        page_content: Optional[str] = None
        content: Optional[str] = None
        content_preview: Optional[str] = None

        # Dense 检索分数
        score: Optional[float] = None
        milvus_score: Optional[float] = None
        dense_rank: Optional[int] = None
        dense_rrf_score: Optional[float] = None

        # BM25 检索分数
        bm25_score: Optional[float] = None
        bm25_rank: Optional[int] = None
        bm25_rrf_score: Optional[float] = None

        # Hybrid / Reranker
        hybrid_score: Optional[float] = None
        retrieval_sources: Optional[List[str]] = None
        raw_rerank_score: Optional[float] = None
        rerank_boost: Optional[float] = None
        rerank_boost_reasons: Optional[List[str]] = None
        rerank_score: Optional[float] = None


class QueryRouteInfo(BaseModel):
    """
    Query Router 查询路由信息。

    用于说明本次问题被识别为什么类型，
    以及最终启用了哪些检索增强策略。
    """

    query_type: str = Field(description="查询类型，例如 structured_query / semantic_query / exact_code_query")
    enable_rewrite: bool = Field(description="本次请求最终是否启用 Query Rewrite")
    enable_rerank: bool = Field(description="本次请求最终是否启用 Reranker")
    enable_compression: bool = Field(
        default=True,
        description="是否启用 Context Compression 上下文压缩",
    )
    reason: str = Field(description="Query Router 判断原因")


class RAGTimingInfo(BaseModel):
    """
    RAG 链路内部耗时信息。

    用于分析一次问答中各阶段的耗时占比，
    方便定位性能瓶颈。
    """

    route_seconds: float = Field(description="Query Router 路由耗时")
    rewrite_seconds: float = Field(description="Query Rewrite 耗时")
    retrieve_seconds: float = Field(description="Retriever 检索耗时")
    rerank_seconds: float = Field(description="Reranker 重排耗时")
    confidence_seconds: float = Field(description="Confidence Guard 置信度判断耗时")
    compression_seconds: float = Field(description="Context Compression 上下文压缩耗时")
    context_build_seconds: float = Field(description="上下文拼接耗时")
    llm_seconds: float = Field(description="LLM 生成耗时")
    total_chain_seconds: float = Field(description="RAG Chain 内部总耗时")


class ContextCompressionInfo(BaseModel):
    """
    Context Compression 上下文压缩信息。

    用于观察压缩前后上下文长度变化。
    """

    enabled: bool = Field(description="本次是否启用上下文压缩")
    original_chars: int = Field(description="压缩前上下文总字符数")
    compressed_chars: int = Field(description="压缩后上下文总字符数")
    compression_ratio: float = Field(description="压缩后字符数 / 压缩前字符数")
    doc_count: int = Field(description="参与构造上下文的文档片段数量")


class ChatResponse(BaseModel):
    """
    用户问答响应体。
    """

    request_id: str = Field(description="请求 ID，用于日志追踪")

    username: str = Field(description="当前登录用户名")
    user_id: int = Field(description="当前登录用户 ID")
    user_role: str = Field(description="当前登录用户角色")

    question: str = Field(description="原始问题")
    rewritten_question: Optional[str] = Field(default=None, description="改写后的问题")
    answer: str = Field(description="模型回答")

    sources: List[SourceItem] = Field(description="引用来源列表")

    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="本次请求最终生效的 metadata 过滤条件",
    )

    query_route: Optional[QueryRouteInfo] = Field(
        default=None,
        description="本次请求的 Query Router 路由结果",
    )

    timings: Optional[RAGTimingInfo] = Field(
        default=None,
        description="RAG 链路内部各阶段耗时",
    )

    compression: Optional[ContextCompressionInfo] = Field(
        default=None,
        description="Context Compression 上下文压缩信息",
    )

    cost_time_seconds: float = Field(description="本次请求总耗时，单位：秒")


class ChatAPIResponse(BaseModel):
    """
    /chat 成功响应外层结构。

    统一成功响应格式：
    {
        "success": true,
        "data": {...}
    }
    """

    success: bool = Field(
        default=True,
        description="请求是否成功",
    )

    data: ChatResponse = Field(
        description="问答结果数据",
    )