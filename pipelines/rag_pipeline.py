# pipelines/rag_pipeline.py

"""
RAG Pipeline 构建模块。

作用：
1. 统一创建不同阶段的 RAG Chain。
2. 避免 eval / test / FastAPI 中重复写初始化代码。
3. 后续改模型、路径、top_k，只需要改 config/settings.py。
"""

from retriever.milvus_retriever import MilvusRetriever
from retriever.bm25_retriever import BM25Retriever
from retriever.hybrid_retriever import HybridRetriever

from reranker.bge_reranker import BGEReranker
from rewriter.query_rewriter import QueryRewriter
from chains.rag_chain import EnterpriseRAGChain
from chains.confidence_guard import RetrievalConfidenceGuard
from config.settings import (
    ENABLE_CONFIDENCE_GUARD,
    CONFIDENCE_MIN_OVERLAP_COUNT,
    CONFIDENCE_CHECK_TOP_N,

)

from config.settings import (

    CHUNKS_FILE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    RERANKER_MODEL_NAME,
    RERANKER_DEVICE,
    RERANKER_MAX_LENGTH,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    DENSE_TOP_K,
    BM25_TOP_K,
    HYBRID_TOP_K,
    DENSE_WEIGHT,
    BM25_WEIGHT,
    RRF_K,
)


def build_dense_retriever():
    """
    构建 Dense Retriever。


    使用 MilvusRetriever

    注意：
    这里只切换 Dense Retriever。
    BM25 Retriever 不变，Hybrid Search 仍然是：
    Dense Retriever + BM25 Retriever
    """

    print("=" * 100)
    print("正在构建 Dense Retriever")
    print("=" * 100)


    return MilvusRetriever()


def build_bm25_retriever() -> BM25Retriever:
    """
    创建 BM25 Retriever。

    负责：
    - 编号检索
    - 金额检索
    - 条款号检索
    - 精确关键词检索
    """

    return BM25Retriever(
        chunks_file=CHUNKS_FILE,
        top_k=BM25_TOP_K,
    )


def build_hybrid_retriever() -> HybridRetriever:
    """
    创建 Hybrid Retriever。

    链路：
    Dense Retriever + BM25 Retriever → RRF 融合
    """

    dense_retriever = build_dense_retriever()
    bm25_retriever = build_bm25_retriever()

    return HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        final_top_k=HYBRID_TOP_K,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        rrf_k=RRF_K,
    )


def build_reranker() -> BGEReranker:
    """
    创建 BGE Reranker。

    负责：
    - 对 Retriever 召回结果重新排序
    - 加入 Business Boost 企业规则加权
    """

    return BGEReranker(
        model_name=RERANKER_MODEL_NAME,
        device=RERANKER_DEVICE,
        max_length=RERANKER_MAX_LENGTH,
        use_business_boost=True,
    )


def build_rewriter() -> QueryRewriter:
    """
    创建 Query Rewriter。

    负责：
    - 将口语化问题改写成更适合检索的问题
    - 保留编号类问题原始意图
    """

    return QueryRewriter(
        llm_model=LLM_MODEL,
        llm_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
    )


def build_baseline_rag_chain() -> EnterpriseRAGChain:
    """
    构建 Baseline RAG Chain。

    链路：
    Chroma Retriever → LLM
    """

    return EnterpriseRAGChain(
        retriever=build_dense_retriever(),
        reranker=None,
        rewriter=None,
        llm_model=LLM_MODEL,
        llm_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
    )


def build_reranker_rag_chain() -> EnterpriseRAGChain:
    """
    构建 Reranker RAG Chain。

    链路：
    Milvus Retriever → Reranker → LLM
    """

    return EnterpriseRAGChain(
        retriever=build_dense_retriever(),
        reranker=build_reranker(),
        rewriter=None,
        llm_model=LLM_MODEL,
        llm_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
    )


def build_rewrite_reranker_rag_chain() -> EnterpriseRAGChain:
    """
    构建 Rewrite + Reranker RAG Chain。

    链路：
    Query Rewrite → Milvus Retriever → Reranker → LLM
    """

    return EnterpriseRAGChain(
        retriever=build_dense_retriever(),
        reranker=build_reranker(),
        rewriter=build_rewriter(),
        llm_model=LLM_MODEL,
        llm_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
    )


def build_hybrid_rag_chain() -> EnterpriseRAGChain:
    """
    构建当前最完整的企业 RAG Chain。

    链路：
    Query Rewrite
    → Hybrid Retriever(Dense + BM25)
    → BGE Reranker
    → LLM

    """
    confidence_guard = build_confidence_guard()
    
    return EnterpriseRAGChain(
        retriever=build_hybrid_retriever(),
        reranker=build_reranker(),
        rewriter=build_rewriter(),
        llm_model=LLM_MODEL,
        llm_provider=LLM_PROVIDER,
        temperature=LLM_TEMPERATURE,
        confidence_guard=build_confidence_guard(),
    )

def build_confidence_guard():
    """
    构建低置信度拒答保护器。
    """

    if not ENABLE_CONFIDENCE_GUARD:
        return None

    return RetrievalConfidenceGuard(
        min_overlap_count=CONFIDENCE_MIN_OVERLAP_COUNT,
        check_top_n=CONFIDENCE_CHECK_TOP_N,
    )