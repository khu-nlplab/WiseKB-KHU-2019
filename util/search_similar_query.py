#-*- encoding:utf8 -*-
from util.modules import *
import operator
import os
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from typing import List, TextIO, Dict
from tqdm import tqdm

def extract_morp_str(_morpStr: str) -> str:
    tmp: str = ""
    splitList = _morpStr.split(' ')
    for morp in splitList:
        try: tmp += morp.split('/')[1] + " "
        except: pass
    return tmp.strip()

def calc_jaccard(_answer,_sAnswer,_mode='cnt'):
    aSet = set(_answer.split(' '))
    saSet = set(_sAnswer.split(' '))
    unionSet = aSet | saSet
    interSet = aSet & saSet
    if _mode == 'cnt':
        score = (len(unionSet) - len(interSet)) / len(unionSet)
    elif _mode == 'mean':
        score1 = (len(unionSet) - len(interSet)) / len(unionSet)
        unionLen = 0
        interLen = 0
        for i in unionSet: unionLen += len(i)
        for i in interSet: interLen += len(i)
        score2 = (unionLen - interLen) / unionLen
        score = (score1 + score2)/2
    else:
        unionLen = 0
        interLen = 0
        for i in unionSet: unionLen += len(i)
        for i in interSet: interLen += len(i)
        score = (unionLen - interLen) / unionLen
    return score

def run(_config):
    log('i', "Start similar_query.py")
    '''__________ CONFIG AREA START __________'''
    _config = _config['preprocess']
    model = _config['model']
    qFileName = _config['data_src_file']
    qmFileName = _config['morp_src_file']
    aFileName = _config['data_tgt_file']
    amFileName = _config['morp_tgt_file']

    similarQueryNum = int(_config['similar_query_num'])
    jaccard_threshold = float(_config['jaccard_threshold'])
    inputFileName = _config['similar_input_file']
    outputFileName = _config['similar_output_file']
    '''__________ CONFIG AREA END   __________'''
    if model == 'W':
        # Load dataset data
        QList: List[str] = get_data_from_txt(qFileName)
        QMList: List[str] = get_data_from_txt(qmFileName)
        AList: List[str] = get_data_from_txt(aFileName)
        AMList: List[str] = get_data_from_txt(amFileName)
        tuneAMList: List[str] = []
        # Preprocess
        for line in AMList:
            tuneAMList.append(extract_morp_str(line))

        # Set index directory
        indexdir = './ndx'
        if not os.path.exists(indexdir): os.makedirs(indexdir)
        # Set schema
        schema = Schema(title=TEXT(stored=True),
                        path=ID(stored=True),
                        content=NGRAMWORDS(minsize=2, maxsize=3, stored=True),
                        idx=ID(stored=True))
        ix = create_in(indexdir, schema)
        writer = ix.writer()
        # Build DB
        for fori in range(len(AList)):
            writer.add_document(title=u"{}".format(AList[fori]),
                                path=u"/{}".format(_config['domain']),
                                content=u"{}".format(tuneAMList[fori]),
                                idx=u"{}".format(fori))
        writer.commit()

        # Search
        querys: List[str] = get_data_from_txt(inputFileName)
        outF = open(outputFileName, 'w', encoding='utf-8')

        # Search
        with ix.searcher() as searcher:
            qp = QueryParser("content", ix.schema)
            for query in tqdm(querys):
                query = query.strip()
                scoreDict: Dict[int, float] = {}


                queryIdx = QList.index(query)
                answer = AList[queryIdx]
                # Get a morp sequence
                aMorpStr = extract_morp_str(AMList[ AList.index(answer) ])
                splitMorp = aMorpStr.split(' ')


                # bi-gram weight sum
                for fori in range(len(splitMorp) - 1):
                    user_q = qp.parse(u'{}'.format(splitMorp[fori] + " " + splitMorp[fori + 1]))
                    results = searcher.search(user_q, limit=similarQueryNum)
                    for r in results:
                        try: scoreDict[r['idx']] += r.score
                        except: scoreDict[r['idx']] = r.score

                # Tri-gram weight sum
                for fori in range(len(splitMorp) - 2):
                    user_q = qp.parse(u'{}'.format(splitMorp[fori] + " " + splitMorp[fori + 1] + " " + splitMorp[fori + 2]))
                    results = searcher.search(user_q, limit=similarQueryNum)
                    for r in results:
                        try: scoreDict[r['idx']] += r.score
                        except: scoreDict[r['idx']] = r.score
                sortedScore = sorted(scoreDict.items(), key=operator.itemgetter(1), reverse=True)

                cnt = 0
                jaccardDict: Dict[int, float] = {}
                for item in sortedScore:
                    if cnt == similarQueryNum: break
                    top = sortedScore.pop(0)
                    idx = int(top[0])
                    if int(idx) == queryIdx: pass
                    else:
                        am = extract_morp_str(AMList[ AList.index(answer) ])
                        sam = extract_morp_str(AMList[ idx ])
                        score = calc_jaccard(am, sam, _config['jaccard_method'])  # mean, cnt
                        jaccardDict[idx] = score
                        cnt += 1

                sortedJaccard = sorted(jaccardDict.items(), key=operator.itemgetter(1))
                for item in sortedJaccard:
                    idx, score = item[0], item[1]
                    if float(score) <= jaccard_threshold:
                        outF.write(
                            "{}:{}:{}:{}:{}".format(queryIdx, idx, AList[queryIdx], AList[idx], str(score)[:5]) + '\n')
        log('s', 'End')
    elif model == 'ME':
        pass


