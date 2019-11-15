import operator
import os
from util.modules import *
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from typing import List, TextIO, Dict
from tqdm import tqdm
def extract_morp_str(_morpStr: str) -> str:
    """
    :param _morpStr: morp long string
    :return: 품사만 남은 string
    """
    tmp: str = ""
    splitList = _morpStr.split(' ')
    for morp in splitList:
        try:
            tmp += morp.split('/')[1] + " "
        except:  # 공백인 경우, 형태소분석결과가 없음
            pass
    return tmp.strip()
def run(_config):
    '''__________ CONFIG AREA START __________'''
    log('i', "Start make_testset.py")
    _config = _config['test']
    domain = _config['test_domain']
    VOCAB_SRC_FILE = _config['search_vocab_file']
    Q_FILE_NAME: str = _config['test_src_file']
    Q_MORP_FILE_NAME: str = _config['test_morp_src_file']
    A_MORP_FILE_NAME: str = _config['test_morp_tgt_file']
    AMList: List[str] = get_data_from_txt(A_MORP_FILE_NAME)
    SIMILAR_COUNT = 1 # 30개까지 뽑음 유사쿼리

    INPUT_FILE = _config['test_input_morp_file']
    OUTPUT_FILE = _config['test_output_file']
    srcVocab = get_data_from_txt(VOCAB_SRC_FILE)
    '''__________ CONFIG AREA END   __________'''

    # Load dataset data
    QList: List[str] = get_data_from_txt(Q_FILE_NAME)
    QMList: List[str] = get_data_from_txt(Q_MORP_FILE_NAME)
    tuneQMList :List[str] = []

    # Set index directory
    indexdir = './ndx'
    if not os.path.exists(indexdir):
        os.makedirs(indexdir)

    # Set schema
    schema = Schema(title=TEXT(stored=True),
                    path=ID(stored=True),
                    content=NGRAMWORDS(minsize=2, maxsize=3, stored=True),
                    idx=ID(stored=True))
    # create schema
    ix = create_in(indexdir, schema)

    # Define the writer for search Inverted
    writer = ix.writer()

    # Preprocess
    for fori in range( len(QMList) ):
        tuneQMList.append(extract_morp_str(QMList[fori]))

    # Build DB
    for fori in range( len(QMList) ):
        writer.add_document(title=u"{}".format(QList[fori]),
                            path=u"/{}".format(domain),
                            content=u"{}".format(tuneQMList[fori]),
                            idx=u"{}".format(fori))
    writer.commit()
    querys: List[str] = get_data_from_txt(INPUT_FILE)
    outF = open(OUTPUT_FILE,'w',encoding='utf8')

    # Search
    with ix.searcher() as searcher:
        qp = QueryParser("content", ix.schema)
        for query in tqdm(querys):
            query = query.strip()
            dataDict: Dict[int, float] = {}
            splitMorp = query.split(' ') # A Morp 쪼갬
            # bi-gram weight sum
            for fori in range(len(splitMorp) - 1):
                user_q = qp.parse(u'{}'.format(splitMorp[fori]+" "+splitMorp[fori+1]))
                results = searcher.search(user_q, limit=SIMILAR_COUNT)
                for r in results:
                    try:
                        dataDict[r['idx']] += r.score
                    except:
                        dataDict[r['idx']] = r.score
            # Tri-gram weight sum
            for fori in range(len(splitMorp) - 2):
                user_q = qp.parse(u'{}'.format(splitMorp[fori]+" "+splitMorp[fori+1]+" "+splitMorp[fori+2]))
                results = searcher.search(user_q, limit=SIMILAR_COUNT)
                for r in results:
                    try:
                        dataDict[r['idx']] += r.score
                    except:
                        dataDict[r['idx']] = r.score

            sortedTmp = sorted(dataDict.items(), key=operator.itemgetter(1), reverse=True)

            top = sortedTmp.pop(0)
            idx = int( top[0] )
            # word 2 vec
            sqMorp = QMList[idx]
            saMorp = AMList[idx]

            result = ""
            for stem in query.split(' '):
                try:
                    idx = srcVocab.index(stem)
                except:
                    idx = srcVocab.index('<unk>')
                result += str(idx + 3) + ' '
            result = result[:-1] + '||'

            for stem in sqMorp.split(' '):
                idx = srcVocab.index(stem)
                result += str(idx + 3) + ' '
            result = result[:-1] + '|'

            for stem in saMorp.split(' '):
                idx = srcVocab.index(stem)
                result += str(idx + 3) + ' '
            result = result[:-1] + '\n'
            outF.write(result)
        outF.close()
    log('s', 'End')