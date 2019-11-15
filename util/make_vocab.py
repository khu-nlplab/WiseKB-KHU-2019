#-*- encoding:utf8 -*-
from util.modules import *
import operator

# Main function
def run(_config):
    log('i',"Start make_vocab.py")
    _config = _config['preprocess']
    srcLines = get_data_from_txt(_config['morp_src_file'])
    tgtLines = get_data_from_txt(_config['morp_tgt_file'])

    dataDict = {}
    for line in srcLines:
        line = line.strip()
        tokens = line.split(" ")
        for token in tokens:
            try: dataDict[str(token)] += 1
            except: dataDict[str(token)] = 1
    for line in tgtLines:
        line = line.strip()
        tokens = line.split(" ")
        for token in tokens:
            try: dataDict[str(token)] += 1
            except: dataDict[str(token)] = 1
    sortedData = sorted(dataDict.items(), key=operator.itemgetter(1), reverse=True)

    with open(_config['vocab_output_file'], 'w', encoding='utf-8') as ofstream_vocab:
        ofstream_vocab.write('<unk>' + '\n')
        for item in sortedData:
            ofstream_vocab.write(item[0].strip() + '\n')
    log('s', "Complete make the vocab file : {} ".format(_config['vocab_output_file']))
    log('s','End')