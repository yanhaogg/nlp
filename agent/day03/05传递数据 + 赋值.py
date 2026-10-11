from langchain_core.runnables import RunnablePassthrough,RunnableAssign

# data = {'name': '李雷','age' : 18}
# data = {'name': '李雷','age' : 18, 'greeting': 欢迎 李雷 来到北京大学!}
def add_greeting(data):
    return f"欢迎 {data['name']} 来到北京大学!"
# data['greeting'] = f"欢迎 {data['name']} 来到北京大学!"

# chain = RunnablePassthrough()| RunnableAssign(mapper={"greeting":add_greeting})
# 直接修改原始传递的字典 , 一般放在最开头 , 数据进入prompt之前
chain = RunnablePassthrough.assign(greeting = add_greeting) # 把两个组件合并成一个组件

res = chain.invoke({'name': '李雷','age' : 18})
print(res)

# 11 : 14