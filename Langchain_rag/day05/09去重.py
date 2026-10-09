text = '''
人工智能是未来方向。
机器学习是重要分支。
人工智能是未来方向。
深度学习正在崛起。
'''

import hashlib
# 去重要放在基础清洗和标准清洗之后,因为清洗之前有很多 "\n"
def deduplication(text):
    paragraphs = [p.strip() for p in text.split("\n")]
    seen = {} # 查重字典
    cleaned = [] # 去重后的干净列表

    for para in paragraphs:
        # 特征提取 : MD5哈希 + 前150字符
        key = hashlib.md5(para[:150].encode('utf-8')).hexdigest()
        if key not in seen:
            seen[key] = para
            cleaned.append(para)
    return "\n".join(cleaned)

res = deduplication(text)
print(res)

