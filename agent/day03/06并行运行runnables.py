from langchain_core.runnables import RunnableLambda,RunnableParallel

# 1.显式写法
chain = RunnableParallel({
    "sub_1" : RunnableLambda(lambda x: x - 1),
    "sub_2" : RunnableLambda(lambda x: x + 1)
})
res = chain.invoke(5)
print(res)

# 2.隐式写法
# 把一个字典放在管道符 "|"右边的时候,langchain会自动将其转换成 RunnableParallel
# 所以也就是,只要并行运行的多个组件,不是chain里面的第一个组件,就都可以进行隐式写法
chain = (RunnableLambda(lambda x: x ** 2)
        |{# type : ignore
            "sub_1" : RunnableLambda(lambda x: x - 1),
            "sub_2" : RunnableLambda(lambda x: x + 1)
        })
print(chain.invoke(5))
'''
{'sub_1': 4, 'sub_2': 6}
{'sub_1': 24, 'sub_2': 26}
'''