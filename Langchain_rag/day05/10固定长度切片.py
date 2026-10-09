import numpy as np

# 1.固定长度切片
# 固定长度切片是最底层的切片逻辑,即我们使用的切割器其他策略都不可实施的时候,会自动降级成固定长度
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size = 500,    # 每个切片的字符数
    chunk_overlap = 50,  # 重叠字符数
    separator="",        # 分隔符为空,代表按字符均匀切割
    length_function=len  # 使用字符长度计数
)

with open('Langchain_rag/data/百度千帆.txt','r',encoding='utf-8') as f:
    text = f.read()

# 有两种切分方法:split_text(针对普通文章即字符串) 以及  split_documents(针对 Document 类)
chunks = splitter.split_text(text)

chunk_lengths = [len(chunk) for chunk in chunks]

# 基本统计量
print(f"切片长度:{len(chunks)}")
print(f"最小长度:{np.min(chunk_lengths)} 字符")
print(f"最大长度:{np.max(chunk_lengths)} 字符")
print(f"平均长度:{np.mean(chunk_lengths)} 字符")
print(f"第一个切片的内容: \n{chunks[0]}")

# 2.循环递归切片
# 更高级策略的兜底策略
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,     # 目标切片长度(字符数)
    chunk_overlap = 50,   # 重叠窗口(兜底用)
    separators = [
        "\n\n",
        "\n",
        "。","！", "？",
        "；",
        "，"
    ],
    length_function= len
)

chunks = splitter.split_text(text)

chunk_lengths = [len(chunk) for chunk in chunks]

# 基本统计量
print(f"切片长度:{len(chunks)}")
print(f"最小长度:{np.min(chunk_lengths)} 字符")
print(f"最大长度:{np.max(chunk_lengths)} 字符")
print(f"平均长度:{np.mean(chunk_lengths)} 字符")
print(f"第一个切片的内容: \n{chunks[0]}")