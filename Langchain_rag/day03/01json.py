item = [
  {
    "id": 1,
    "product": "iPhone 16",
    "price": 7999,
    "inStock": True
  },
  {
    "id": 2,
    "product": "AirPods",
    "price": 1299,
    "inStock": False
  }
]

import json

with open('item.json','w',encoding='utf-8') as f:
    json.dump(item,f,ensure_ascii=False,#正常显示中文
              indent=4)#排版，缩进长度为4格