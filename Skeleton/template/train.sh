NMT_DIR=..
DOMAIN=weather
python3 ${NMT_DIR}/template.py \
    -config config.yml \
    -nmt_dir ${NMT_DIR} \
    -src_vocab ../data/vocab_src_${DOMAIN} \
    -tgt_vocab ../data/vocab_src_${DOMAIN} \
    -train_file ../data/${DOMAIN}70_0806ms_train \
    -valid_file ../data/${DOMAIN}70_0806ms_valid \
    -mode train  \
