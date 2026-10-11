from langchain_core.runnables import RunnableLambda

chain = RunnableLambda(lambda x : x-1) | RunnableLambda(lambda x : x ** 2)

res = chain.invoke(10)
print(res)
