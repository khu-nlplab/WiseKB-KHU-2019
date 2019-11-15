NMT_DIR=..
python3 ${NMT_DIR}/translate.py \
        -test_file  ../data/f75p3s_test\
        -model_type rg \
        -tgt_out ablation_response \
        -model ./out/0722/checkpoint_epoch9.pkl  \
        -src_vocab ../data/vocab_src_food \
        -tgt_vocab ../data/vocab_src_food


