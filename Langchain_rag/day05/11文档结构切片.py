# 文档结构切片适用于,科技说明,技术服务,或者是类似的结构鲜明的Md或者Md转换的pdf
# 对于结构松散的散文,诗歌,小说等不适合

# 1.markdown的splitter用法
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 定义要切分的标题层级
headers_to_split_on = [
    ("#","一级标题"),
    ("##","二级标题"),
    ("###","三级标题")
]

# 创建切片器
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers = True   #   是否过滤掉标题内容
)

# 执行切片
document = """
# Foo
## Bar
Hi This is Jack
Hi This is Lucy
## Baz
Hi This is Tom
"""

chunks = splitter.split_text(document)
print(chunks)
#
# for chunk in chunks:
#     print(f"元数据:{chunk.metadata}")
#     print(f"内容:{chunk.page_content}\n")