python final_step.py
NMT_DIR=..
python3 ${NMT_DIR}/template.py \
      -config ../template/config.yml \
      -nmt_dir ${NMT_DIR} \
      -src_vocab ../data/vocab_src_food \
      -tgt_vocab ../data/vocab_src_food \
      -model ../template/out/0722/checkpoint_epoch8.pkl \
      -test_file ../data/f75p3s_test \
      -out_file ./in_tem \
      -mode test

python3 ../read.py douban in_tem > output_skeleton

python3 ${NMT_DIR}/translate.py \
    -test_file ./in_tem \
    -model_type rg \
    -tgt_out ./output_case \
    -model ./hard.pkl \
    -src_vocab ../data/douban/vocab_src \
    -tgt_vocab ../data/douban/vocab_tgt
