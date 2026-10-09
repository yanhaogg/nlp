import re
# 1.页码清除
text = "第1页\n人工智能是未来的方向\n第2页\n机器学习是其分支"
"第 X 页"

# 删除类似 第1页  第 12 页  Page 1的页码
res = re.sub(r'(第\s*\d+\s*页|Page\s*\d+)',"",text)
# | 在正则表达式中代表 "或" 的意思

print(res)

# 2.去除页眉页脚
import re

text = """
XX公司内部培训资料
第1页
人工智能是未来方向
版权所有 © 2026
XX公司内部培训资料
第2页
机器学习是其分支
版权所有 © 2026
"""

# (1)如果内容完全一样,重复出现,直接匹配原内容
# (2)如果相同中带有变化,固定同样的内容,使用符号表达变化的内容

# 匹配每页开头的固定页眉 以及 页码,结尾的固定页脚
res = re.sub(r'(XX公司内部培训资料\s*|第\s*\d+\s*页|版权所有 © 2026\s*)',"",text)
print(res)

# 3.换行切断修复
text = "This is an atten- tion test.\nAnother exam- ple here."

res = re.sub(r'(\w+)-\s*\n*\s*(\w+)',r'\1\2',text)
print(res)