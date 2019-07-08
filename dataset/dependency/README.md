# WiseKB
WiseKB project > datasets > dependency 

## Decode usage example code (python 3.6.x)  
```python
import ast
import json
f = open('src_food.txt','r',encoding='utf8')
lines = f.readlines()
f.close()
for line in lines:
    json_data = json.dumps(ast.literal_eval(line.strip()))
    jsonStr = json.loads(json_data)
    for item in jsonStr:
        print(item)
        """ 아래처럼 결과가 나옴
        {'id': 0.0, 'text': '학관에', 'head': 2.0, 'label': 'NP_AJT', 'mod': [], 'weight': 0.287726}
        {'id': 1.0, 'text': '새로', 'head': 2.0, 'label': 'AP', 'mod': [], 'weight': 0.756773}
        {'id': 2.0, 'text': '생긴', 'head': 3.0, 'label': 'VP_MOD', 'mod': [0.0, 1.0], 'weight': 0.421279}
        {'id': 3.0, 'text': '식당', 'head': 4.0, 'label': 'NP_AJT', 'mod': [2.0], 'weight': 0.194773}
        {'id': 4.0, 'text': '가봤어?', 'head': -1.0, 'label': 'VP', 'mod': [3.0], 'weight': 0.0152167}
        """
```

## description 
<table>
<tr>
<th>  </th>
<th>tag</th>
<th>mean</th>
</tr>
<tr>
<th rowspan="6"> dependecy </th>
<td>id</td>
<td>어절의 ID (출현 순서)</td>
</tr>
<tr>
<td>text</td>
<td>의존구문 텍스트</td>
</tr>
<tr>
<td>head</td>
<td>부모 어절의 ID</td>
</tr>
<tr>
<td>label</td>
<td>의존관계</td>
</tr>
<tr>
<td>mod</td>
<td>자식 어절들의 ID </td>
</tr>
<tr>
<td>weight</td>
<td>의존구문 분석 결과 신뢰도</td>
</tr>

</table>
