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
    # 例如传了 unknown_field，会直接返回 422
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
               # 正常公共文档问答
                {
                    "question": "员工出差住宿标准是什么？",
                    "retrieve_top_k": 6,
                    "rerank_top_k": 3,
                    "metadata_filter": {
                        "source": "travel_policy.md"
                    },
                    "enable_rewrite": True,
                    "enable_rerank": True,
                },
                # 无答案问题拒答
                {
                    "question": "公司食堂几点开门？",
                    "retrieve_top_k": 6,
                    "rerank_top_k": 3,
                    "metadata_filter": None,
                    "enable_rewrite": True,
                    "enable_rerank": True,
                },
               # 私有文档权限测试
                {
                    "question": "薪资调整流程是什么？",
                    "retrieve_top_k": 6,
                    "rerank_top_k": 3,
                    "metadata_filter": {
                        "source": "hr_salary_policy.md"
                    },
                    "enable_rewrite": True,
                    "enable_rerank": True,
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
    单条引用来源。
    """

    source: Optional[str] = Field(default=None, description="来源文件")
    header_1: Optional[str] = Field(default=None, description="一级标题")
    header_2: Optional[str] = Field(default=None, description="二级标题")
    chunk_id: Optional[str] = Field(default=None, description="chunk ID")

    content_preview: Optional[str] = Field(
        default=None,
        description="引用片段预览",
    )

    retrieval_sources: Optional[List[str]] = Field(
        default=None,
        description="检索来源，例如 dense / bm25",
    )

    chroma_score: Optional[float] = Field(default=None, description="Chroma 原始分数")
    milvus_score: Optional[float] = Field(default=None,description="Milvus Dense 检索分数",)
    bm25_score: Optional[float] = Field(default=None, description="BM25 原始分数")
    hybrid_score: Optional[float] = Field(default=None, description="Hybrid 融合分数")
    rerank_score: Optional[float] = Field(default=None, description="Reranker 最终分数")



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