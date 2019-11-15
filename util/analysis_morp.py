# -*- coding:utf-8 -*-
import urllib3
from typing import List, TextIO
import json
from util.modules import *
from tqdm import tqdm
'''
reference :http://aiopen.etri.re.kr/guide_wiseNLU.php
'''
'''__________ CONFIG AREA START __________ '''
def run(_config):
    log('i', "Start analysis_morp.py")
    '''__________ CONFIG AREA START __________'''
    _config = _config['preprocess']
    accessKey = _config['access_key']
    inputFileName: str = _config['input_file']
    outputFileName: str = _config['output_file']

    openApiURL :str = "http://aiopen.etri.re.kr:8000/WiseNLU"
    analysisCode :str = "morp" #morp, dparse
    requestJson = {
        "access_key": accessKey, # from CONFIG file
        "argument": {
            "text": '',
            "analysis_code": analysisCode
        }
    }

    '''__________ CONFIG AREA END   __________ '''
    textList: List[str] = [] # input txt에 있는 data 저장 할 list
    result  : str = '' # 반환값 저장 변수
    outputF : TextIO = open(outputFileName, "w", encoding='utf-8-sig') #저장할 파일 file output stream

    try:
        inputF: TextIO = open(inputFileName, 'r', encoding='utf-8-sig')
        for line in inputF.readlines():
            if line.strip() != '':
                textList.append(line.strip())
        inputF.close()

    except OSError:
        print('cannot input open')
        exit(-1)
    ### Make http pool & request
    http = urllib3.PoolManager()
    for text in tqdm(textList):
        requestJson['argument']['text'] = text
        response = http.request(
            "POST",
            openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            body=json.dumps(requestJson)
        )
        jsonStr = json.loads(str(response.data, "utf-8"))
        result = ""
        for sentence in jsonStr['return_object']['sentence']:
            for morp in sentence['morp']:
                result += "{}/{} ".format(morp['lemma'],morp['type'])
            outputF.write('{}\n'.format(result))

    outputF.close()
    log('s','End')