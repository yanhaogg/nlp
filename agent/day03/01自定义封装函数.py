from langchain_core.runnables import RunnableLambda

# 1.定义一个函数 : 对任意一个数字 , +6
def add_six(number):
    return number + 6

# 使用RunnableLambda 包装成 Runnable
add_six_runnable = RunnableLambda(add_six) # 参数是函数的签名,不能是具体的函数

# 调用
res = add_six_runnable.invoke(10)
print(res)

# 2.封装  lambda 函数
# lambda 函数,lambda 后面跟着对应的参数, : 后面协商对参数的处理方式
add_six_runnable = RunnableLambda(lambda x : x + 6)

res = add_six_runnable.invoke(10)
print(res)