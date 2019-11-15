NMT_DIR=..
MODEL_DIR=1113_Shop_60
TGT_DIR=1113_S60
NUM=20
DOMAIN=weather
python3 ${NMT_DIR}/translate.py \
    -test_file ../data/shop_vec_test \
    -model_type JNT \
    -tgt_out ./${TGT_DIR}/soft_out${NUM} \
    -model ./out/${MODEL_DIR}/checkpoint_epoch${NUM}.pkl \
    -src_vocab ../data/vocab_src_all_1113 \
    -tgt_vocab ../data/vocab_tgt_all_1113
