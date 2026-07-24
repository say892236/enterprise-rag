from loader.markdown_loader import MarkdownLoader
from splitter.markdown_splitter import MarkdownSplitter
from utils.text_cleaner import TextCleaner


def main() -> None:
    # 1. 加载原始文档
    loader = MarkdownLoader("data/raw/markdown")
    documents = loader.load_directory()

    print(f"原始文档数量：{len(documents)}")

    # 2. 清洗文档
    cleaner = TextCleaner()

    for document in documents:
        document.page_content = cleaner.clean(
            document.page_content
        )

    # 3. 切分文档
    splitter = MarkdownSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    print(f"切片数量：{len(chunks)}")

    # 4. 基础检查
    assert documents, "没有加载到原始文档"
    assert chunks, "没有生成任何切片"

    assert all(
        chunk.metadata.get("source")
        for chunk in chunks
    ), "存在缺少 source 的切片"

    assert all(
        "chunk_id" in chunk.metadata
        for chunk in chunks
    ), "存在缺少 chunk_id 的切片"

    for chunk in chunks[:10]:
        print("=" * 70)
        print("chunk_id:", chunk.metadata.get("chunk_id"))
        print("source:", chunk.metadata.get("source"))
        print("header_1:", chunk.metadata.get("header_1"))
        print("header_2:", chunk.metadata.get("header_2"))
        print("char_count:", chunk.metadata.get("char_count"))
        print("正文：")
        print(chunk.page_content[:500])
        print()

if __name__ == "__main__":
    main()