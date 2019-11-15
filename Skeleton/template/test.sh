NMT_DIR=..
NUM=15
DOMAIN=food
python3 ${NMT_DIR}/template.py \
    -config config.yml \
    -nmt_dir ${NMT_DIR} \
    -src_vocab ../data/vocab_src_${DOMAIN} \
    -tgt_vocab ../data/vocab_src_${DOMAIN} \
    -model ./out/0813f/checkpoint_epoch${NUM}.pkl \
    -test_file ../data/0813f_TEST_data \
    -out_file ./0813f_${NUM} \
    -mode test
python clean.py TMP > final_output
