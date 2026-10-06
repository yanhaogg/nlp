from model.client import *

sys_role0 = "you are a helpful assistant"

sys_role1 = "你是一个专业的美食评论家，擅长分析菜品的色香味口感，并给出星级评分"
sys_role2 = "你是一个外地的游客"
sys_role3 = "你是北京本地人，说话带儿音，要地道"

prompt = "请评价一下北京特色食物-豆汁儿,生成字数控制在200字以内"

for role in [sys_role0, sys_role1, sys_role2, sys_role3]:
    #print(f"System role: {role}")
    res=llm.generate_response(prompt=prompt, temperature=1.0, top_p=0.95, max_tokens=1024, stop=None, system_role=role)
    print(f"扮演的角色：{role},\n生成的内容：{res}")
   # print(res)
    print("\n")

