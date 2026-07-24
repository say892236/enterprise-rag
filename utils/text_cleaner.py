import re


class TextCleaner:
    """
    企业 Markdown 文档清洗器。

    功能:
    1. 清理 Markdown 引用符号
    2. 清理行尾多余空白
    3. 合并多个空行
    4. 保留标题、表格、列表结构
    """

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")

        # 1. 清理 Markdown 引用符号 >
        text = re.sub(
            r"^>\s?",
            "",
            text,
            flags=re.MULTILINE
        )

        # 2. 删除每一行末尾多余空格或 Tab
        text = re.sub(
            r"[ \t]+$",
            "",
            text,
            flags=re.MULTILINE
        )

        # 3. 合并多个空行，最多保留一个空行
        text = re.sub(
            r"\n\s*\n+",
            "\n\n",
            text
        )

        # 4. 去除整篇文本首尾空白
        text = text.strip()

        return text