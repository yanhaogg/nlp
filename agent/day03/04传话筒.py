from langchain_core.runnables import RunnableLambda,RunnablePassthrough

chain = (RunnableLambda(lambda x:x+1)
        | RunnablePassthrough()      # 假装努力,但是实际上没有任何作用,只是进行传递数据
        | RunnableLambda(lambda x:x * 2)
         )

print(chain.invoke(4))