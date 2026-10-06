import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from model.client import *
import json
import requests
import os
from dotenv import load_dotenv
load_dotenv()




# ====================== 1. 准备数据 ======================
documents = [
    "请假流程:员工因病、事、年假等原因需要请假时,应提前至少1个工作日通过企业微信或OA系统提交请假申请。申请需注明请假类型、起止时间、请假事由,并上传相关证明材料(如病假需提供医院诊断证明)。部门主管在收到申请后需在24小时内审批,超过3天以上的请假需由部门经理或以上领导审批。请假期间如需延长,须提前办理续假手续。未经批准擅自离岗的,按旷工处理。",
    "调休流程:员工因加班产生的调休需在加班后30天内申请使用。申请调休时需在系统中选择对应的加班记录,并填写调休时间。调休申请需由部门主管审批,原则上应在工作安排允许的情况下进行。调休最长可累积至当年年底,逾期未使用的调休将自动清零。调休期间按正常出勤计算工资,但需确保所在团队工作不受影响。紧急情况下可临时申请调休,但需及时补交申请单。",
    
    "年终奖申请:符合条件的正式员工可在每年12月1日至12月15日期间通过HR系统提交年终奖申请。申请需填写个人年度工作总结、绩效自评及下一年度工作计划。年终奖发放标准根据员工职级、绩效考核结果、公司整体业绩及个人贡献综合评定。部门主管需在12月20日前完成对下属的年终奖评级推荐,人力资源部最终审核后于次年1月底前统一发放。未在规定时间内提交申请的员工视为自动放弃本年度年终奖。",
    
    "离职流程:员工提出离职需提前30天以书面形式(或通过OA系统)向直属主管提交离职申请。主管收到申请后应在3个工作日内与员工进行离职面谈,了解离职原因并进行挽留。员工需在离职前完成工作交接、归还公司财产(如工牌、电脑、钥匙等)、结清借款及报销。人力资源部负责办理社保、公积金、工资结算及离职证明开具等手续。未按规定办理离职手续或存在未结清事项的,公司有权暂缓办理离职手续。"
]

def embeddings(prompts):#list
    url = "https://qianfan.baidubce.com/v2/embeddings"
    
    payload = json.dumps({
        "model": "embedding-v1",
        "input": prompts
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '+ os.getenv("BAIDU_QIANFAN_API_KEY")
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    return [res['embedding'] for res in response.json()['data']]
# ====================== 2.向量化 ======================


#print("正在加载 sentence-transformers 模型，请稍候...")
#embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 支持中文
# 对所有文档进行向量化（只计算一次）
print("正在对文档进行向量化...")
#database = embeddings.encode(documents, convert_to_numpy=True)
database = embeddings(documents)

# print(len(database))
# print(database[0])
# print(len(database[0]))

question = input("输入你的问题:")
#query = embeddings.encode([question], convert_to_numpy=True)[0]
query=embeddings([question])

import numpy as np
def cosine_similarity(vec1, vec2):
    # 计算公式 cos = (vec1·vec2)/(||vec1||·||vec2||)
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    # 计算L2范数（自动处理零向量）
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    # 计算余弦相似度
    return dot_product / (norm1 * norm2)

res = []
for i in range(len(database)):
    res.append({"score":cosine_similarity(query, database[i]), "sentence":documents[i]})
print(res)

top_k = 2
final_res = sorted(res, key=lambda x: x["score"], reverse=True)[:top_k]
# for idx, item in enumerate(final_res):
#     print(f'{idx + 1}. {item["sentence"]}')


system_template = """
你是一个AI助理，你的专长是人工智能的知识解答。请严格遵守以下要求：
1. 回答必须完全基于提供的文档内容，不得臆造信息
2. 直接回应问题，避免无关内容
"""
human_template = """根据以下文档内容，回答问题：
{context}
问题：{query}
答案：
"""

from langchain_core.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])
#question = input("我不想干了，我应该怎么操作")
messages = chat_prompt.format_messages(context=final_res, query=question)
messages=convert(messages)
response = llm.generate_response(messages)
print(response)