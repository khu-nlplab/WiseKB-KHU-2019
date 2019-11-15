NMT_DIR=..
DOMAIN=food
python3 ${NMT_DIR}/joint_train.py \
    -model_type JNT \
    -config config.yml \
    -nmt_dir ${NMT_DIR} \
    -src_vocab ../data/vocab_src_all_1113 \
    -tgt_vocab ../data/vocab_tgt_all_1113 \
    -train_file ../data/shop_vec_train \
    -valid_file ../data/shop_vec_valid
