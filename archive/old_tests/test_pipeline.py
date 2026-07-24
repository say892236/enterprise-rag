from loader.markdown_loader import MarkdownLoader
from utils.text_cleaner import TextCleaner



# 1. 加载原始 Markdown

loader = MarkdownLoader(
    "../data/raw/markdown"
)


docs = loader.load_directory()



print(
    "原始文档数量:",
    len(docs)
)



# 2. 清洗

cleaner = TextCleaner()



for doc in docs:

    doc.page_content = cleaner.clean(
        doc.page_content
    )



# 3. 查看结果


for doc in docs:

    print("=" * 50)

    print(
        "文件:",
        doc.metadata["source"]
    )

    print(
        doc.page_content[:300]
    )