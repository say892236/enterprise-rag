from pathlib import Path
from langchain_core.documents import Document


class MarkdownLoader:
    """
    企业级 Markdown 文档加载器

    功能:                        目的：读取 Markdown 文件
                                               ↓
                                 转换成 LangChain Document
    1. 加载单个 Markdown 文件
    2. 批量加载 Markdown 目录
    3. 自动生成 Metadata
    """


    def __init__(self, directory_path: str):

        self.directory_path = Path(directory_path)


     # 读取单个.md文件
    def load_file(self, file_path: Path):

        content = file_path.read_text(
            encoding="utf-8"
        )


        metadata = {
            "source": file_path.name,
            "file_path": str(file_path),
            "file_type": "markdown"
        }


        return Document(
            page_content=content,
            metadata=metadata
        )


    def load_directory(self):


        #检查目录是否存在
        if not self.directory_path.exists():
            raise FileNotFoundError(
                f"目录不存在: {self.directory_path}"
            )

        #用来存放所有读取出来的 Document
        documents = []


        # 遍历所有 md 文件
        for file_path in self.directory_path.glob("*.md"):

            doc = self.load_file(file_path)

            documents.append(doc)


        return documents



if __name__ == "__main__":


    loader = MarkdownLoader(
        "../data/raw/markdown"
    )


    docs = loader.load_directory()


    print(
        f"加载文档数量: {len(docs)}"
    )


    for doc in docs:

        print("-" * 50)

        print(
            "文件:",
            doc.metadata["source"]
        )

        print(
            doc.page_content[:100]
        )