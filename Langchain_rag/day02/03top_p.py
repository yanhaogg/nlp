from model.client import *



prompt = f'''
Task:随机推荐一种颜色
Constraint:只生成颜色名称，不要添加任何解释或额外文字
Few-shot:红色
'''

for p in [0.1, 0.5, 0.9]:
    print("="*30)
    print(f"Top-p={p}:")
    for i in range(10):
        response = llm.generate_response(prompt=prompt, top_p=p,temperature=1.5)
        print(response)