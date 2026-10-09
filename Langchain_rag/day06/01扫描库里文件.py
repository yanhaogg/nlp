from pathlib import Path
import json
folder = Path('Langchain_rag/pdf_data')
# 这里扫描所有的json文件,然后取所有文件的前缀
json_stems = {p.stem for p in folder.glob("*.json")}
with_json = set()
without_json = set()
for pdf in folder.glob("*.pdf"):
    (with_json if pdf.stem in json_stems else without_json).add(pdf.stem)
result = {"1":sorted(with_json),"0":sorted(without_json),"2":[]}

print(f'一共{len(result["0"])}篇论文没有json配对文件')
print(f'一共{len(result["1"])}篇论文有json配对文件')
with open('Langchain_rag/day06/scan.json','w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=4)
