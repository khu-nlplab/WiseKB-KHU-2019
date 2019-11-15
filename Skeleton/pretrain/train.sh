NMT_DIR=..
python3 ${NMT_DIR}/train.py \
    -model_type rg \
    -config config.yml \
    -nmt_dir ${NMT_DIR} \
    -src_vocab ../data/vocab_src_food \
    -tgt_vocab ../data/vocab_src_food \
    -train_file ../data/f75p3s_train \
    -valid_file ../data/f75p3s_valid

