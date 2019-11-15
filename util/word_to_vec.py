#-*- encoding:utf8 -*-
from util.modules import *
from typing import List, TextIO, Dict
from tqdm import tqdm
def run(_config):
    log('i', "Start word_to_vec.py")
    '''__________ CONFIG AREA START __________'''
    _config = _config['preprocess']
    qFileName = _config['data_src_file']
    qmFileName = _config['morp_src_file']
    aFileName = _config['data_tgt_file']
    amFileName = _config['morp_tgt_file']

    inputFileName = _config['similar_query_file']
    outputFileName = _config['word2vec_output_file']
    vocabFileName = _config['vocab_file']
    vocabPadding = int( _config['vocab_padding'] )
    '''__________ CONFIG AREA END __________'''
    # Load dataset data
    QList: List[str] = get_data_from_txt(qFileName)
    QMList: List[str] = get_data_from_txt(qmFileName)
    AList: List[str] = get_data_from_txt(aFileName)
    AMList: List[str] = get_data_from_txt(amFileName)
    vocab = get_data_from_txt(vocabFileName)

    lines = get_data_from_txt(inputFileName)
    outF = open(outputFileName, 'w', encoding='utf8')

    for line in tqdm(lines):
        line = line.strip()
        qIdx, sqIdx = int(line.split(":")[0]), int(line.split(":")[1])

        query, answer = QList[qIdx], AList[qIdx]
        squery, sanswer = QList[sqIdx], AList[sqIdx]

        qMorp, aMorp = QMList[qIdx], AMList[qIdx]
        sqMorp, saMorp = QMList[sqIdx], AMList[sqIdx]

        result = ""
        for stem in qMorp.split(' '):
            idx = vocab.index(stem)
            result += str(idx + vocabPadding) + ' '
        result = result[:-1] + '|'

        for stem in aMorp.split(' '):
            idx = vocab.index(stem)
            result += str(idx + vocabPadding) + ' '
        result = result[:-1] + '|'

        for stem in sqMorp.split(' '):
            idx = vocab.index(stem)
            result += str(idx + vocabPadding) + ' '
        result = result[:-1] + '|'

        for stem in saMorp.split(' '):
            idx = vocab.index(stem)
            result += str(idx + vocabPadding) + ' '
        result = result[:-1] + '\n'
        outF.write(result)
    outF.close()
    log('s', 'End')