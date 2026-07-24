# embeddings/bge_embedding.py

"""
BGE Embedding 初始化模块。

作用：
1. 统一创建 bge-m3 Embedding 模型。
2. 避免在 vectorstore、retriever 等地方重复写 HuggingFaceEmbeddings。
3. 后续如果要切换模型、设备，只需要改这里或 config/settings.py。
"""

from langchain_huggingface import HuggingFaceEmbeddings


def create_bge_embeddings(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """
    创建 bge-m3 Embedding 模型。

    参数：
    model_name:
        Embedding 模型名称，当前使用 BAAI/bge-m3。

    device:
        模型运行设备。
        当前你的 torch 没有 CUDA，所以使用 cpu。

    返回：
        LangChain 可用的 HuggingFaceEmbeddings 实例。
    """

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            # bge 系列通常建议开启归一化
            # 这样向量相似度计算更稳定
            "normalize_embeddings": True,
        },
    )