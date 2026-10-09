# 1.访问本地文件的核心抓手(真正负责落地)
def read_file(file_path):
    """读取文件内容并返回"""
    with open(file_path,'r',encoding='utf-8') as f:
        content = f.read()
    return content
# print(read_file('../data/data.txt'))

# 2.发送指令(下达调用上面函数的命令)
# 函数注册表(用来告诉 LLM 我们现在有哪些函数)
functions = {
    "read_file" : read_file # 后面的是函数签名
}

# 调用函数(LLM 发送的实际指令)
json_input = {
    "name" : "read_file",
    "arguments" : {
        "file_path" : "../data/data.txt"
    }
}

# 根据JSON结构执行函数调用(把指令翻译成实际调用函数的语句)
if json_input["name"] in functions:
    res = functions[json_input["name"]](**json_input["arguments"])
    # functions[json_input["name"]] : 相当于调用函数本身
    # (**json_input["arguments"])相当于把参数拆包,并把键值对作为参数,传入函数
    print(res)
else:
    print(f"函数 {json_input['name']} 未找到")