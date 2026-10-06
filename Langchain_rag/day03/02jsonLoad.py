import json
with open("item.json",'r',encoding='utf-8') as f:
    data=json.load(f)

print(data[0]['price'])