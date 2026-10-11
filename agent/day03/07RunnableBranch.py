from langchain_core.runnables import RunnableBranch, RunnableLambda

# 使用 RunnableBranch + lambda  实现根据字符串长度进行不同处理
# 这里的 x 可以理解为输入的文本
processor = RunnableBranch(
    # case1 : 长度 <= 20,直接返回原文
    (lambda x : len(x) <=20,RunnableLambda(lambda x:x)), # 可以替换成RunnablePassthrough

    # case2 : 长度在 20~50 之间,返回前 10个 + ... + 后10个
    (lambda x: 20 < len(x) <= 50,RunnableLambda(lambda x : x[:10] + "..." + x[-10:])),

    # 默认分支 : 长度 > 50 ,返回前 20个 + ... + 后20个
    RunnableLambda(lambda x: x[:20] + "..." + x[-20:])
)

texts = [
        "Hello World", # 短文本（≤20）
        "This is a medium length text for testing.", # 中等长度（20~50）
        "This is a very long text used to test RunnableBranch.", # 长文本（>50）
]
#
# for text in texts:
#     print(processor.invoke(text))

print(processor.batch(texts))