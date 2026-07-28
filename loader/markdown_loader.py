from pathlib import Path
from langchain_core.documents import Document
import frontmatter


class MarkdownLoader:
    """
    企业级 Markdown 文档加载器

    功能：

    Markdown 文件
        ↓
    Front Matter解析
        ↓
    LangChain Document

    自动保留企业知识库 metadata：
    - department
    - allowed_roles
    - security_level
    - document_type
    """


    def __init__(self, directory_path: str):

        self.directory_path = Path(directory_path)


    def load_file(self, file_path: Path):
        """
        加载单个 Markdown 文件

        处理：
        1. 解析 YAML metadata
        2. 分离正文
        3. 创建 Document
        """


        # 读取 Markdown
        post = frontmatter.load(
            file_path
        )


        # YAML metadata
        metadata = {

            # 文件基础信息
            "source": file_path.name,

            "file_path": str(file_path),

            "file_type": "markdown",


            # 企业知识库 metadata
            **post.metadata
        }


        return Document(

            # 只保存正文
            page_content=post.content,

            metadata=metadata
        )


    def load_directory(self):

        if not self.directory_path.exists():

            raise FileNotFoundError(
                f"目录不存在: {self.directory_path}"
            )


        documents = []


        for file_path in self.directory_path.rglob("*.md"):

            doc = self.load_file(
                file_path
            )

            documents.append(doc)


        return documents



if __name__ == "__main__":


    loader = MarkdownLoader(
        "data/raw/markdown"
    )


    docs = loader.load_directory()


    print(
        f"加载文档数量: {len(docs)}"
    )


    for doc in docs:

        print("-" * 50)

        print(
            doc.metadata
        )

        print(
            doc.page_content[:100]
        )