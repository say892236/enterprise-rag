# tests/test_context_compressor.py

"""
Context Compressor 测试脚本。
"""

import sys
from pathlib import Path

from langchain_core.documents import Document


# 把项目根目录加入 Python 模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from compressor.context_compressor import ContextCompressor


def main():
    compressor = ContextCompressor(
        max_chars_per_doc=350,
        min_chars_to_keep=80,
    )

    query = "普通员工去上海出差，酒店最多能报多少？"

    doc = Document(
        page_content="""
公司差旅制度适用于所有正式员工。员工出差前需要提交出差申请。
上海、北京、广州、深圳属于一类城市。
住宿标准如下：
| 城市类别 | 普通员工 | 经理 | 总监及以上 |
| 一类城市 | 600元/晚 | 750元/晚 | 950元/晚 |
| 二类城市 | 450元/晚 | 600元/晚 | 800元/晚 |
员工应优先选择协议酒店。超过标准的住宿费用需要审批。
公司食堂开放时间为内部后勤安排，不属于差旅制度。
""",
        metadata={
            "source": "travel_policy.md",
            "header_2": "住宿标准",
        },
    )

    compressed_docs = compressor.compress_documents(
        query=query,
        documents=[doc],
    )

    compressed_doc = compressed_docs[0]

    print("=" * 80)
    print("原始长度:", len(doc.page_content))
    print("压缩后长度:", len(compressed_doc.page_content))
    print("metadata:", compressed_doc.metadata)
    print("-" * 80)
    print(compressed_doc.page_content)


if __name__ == "__main__":
    main()