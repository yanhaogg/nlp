import re

text = "小坤今年18岁,他喜欢唱跳RAP打篮球。"

# 示例1 : 捕获并提取数字和单位
res= re.search(r'(\d+)岁',text)
print(res)
print(res.group(0)) # 完整的匹配结果
print(res.group(1)) # 捕获组的第一组结果

# 示例2 : 同时捕获两个部分
res = re.search(r'(\w+)喜欢(\w+)',text)
print(res.group(0)) # 完整的匹配结果
print(res.group(1)) # 捕获组的第一组结果
print(res.group(2)) # 捕获组的第二组结果

# 核心用途
text = "atten- tion 以及  pro- cess"
# res = re.sub(r'-\s','',text)
# print(res)
cleaned = re.sub(r'(\w+)-\s*(\w+)',r'\1\2',text)#\s是空格

print(cleaned)
