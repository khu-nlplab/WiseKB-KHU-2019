import random
from util.modules import *
def run(_config):
    log('i', "Start split_dataset.py")
    '''__________ CONFIG AREA START __________'''
    _config = _config['preprocess']
    inputFileName = _config['word2vec_file']
    trainFileName = _config['train_output_file']
    validFileName = _config['valid_output_file']
    trainRatio = float( _config['train_ratio'] )
    validRatio = float( _config['valid_ratio'] )
    '''__________ CONFIG AREA END __________'''
    lines = get_data_from_txt(inputFileName)
    total = int(len(lines))
    train = int( total * trainRatio)
    valid = int( total * validRatio)
    while True:
        # OK : train + valid < total
        # but train + valid > total  -> X
        if train + valid > total:
            valid -= 1
        else:
            break
    tF = open(trainFileName, 'w', encoding='utf-8')
    vF = open(validFileName, 'w', encoding='utf-8')
    for _ in range(train):
        line = random.choice(lines)
        lines.remove(line)
        tF.write(line.strip() + '\n')
    tF.close()
    for _ in range(valid):
        line = random.choice(lines)
        lines.remove(line)
        vF.write(line.strip() + '\n')
    vF.close()
    log('s', 'End')