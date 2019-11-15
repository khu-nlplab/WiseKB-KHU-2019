#-*- encoding:utf8 -*-
import yaml
import util.make_vocab as make_vocab
import util.search_similar_query as search_similar_query
import util.word_to_vec as word_to_vec
import util.split_dataset as split_dataset
import util.make_testset as make_testset
import util.analysis_morp as analysis_morp
if __name__ == "__main__":
    # Get config
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    useFlag = str(config['use'])
    if useFlag == 'pre':
        analysis_morp.run(config)
        make_vocab.run(config)
        search_similar_query.run(config)
        word_to_vec.run(config)
        split_dataset.run(config)
    elif useFlag == '1':
        analysis_morp.run(config)
    elif useFlag == '2':
        make_vocab.run(config)
    elif useFlag == '3':
        search_similar_query.run(config)
    elif useFlag == '4':
        word_to_vec.run(config)
    elif useFlag == '5':
        split_dataset.run(config)
    elif useFlag == '6':
        make_testset.run(config)

