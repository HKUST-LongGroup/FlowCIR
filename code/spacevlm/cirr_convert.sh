CUDA_VISIBLE_DEVICES=0 python3 split_caps_tinyllama.py \
  -i /export/zqhe/retrieval_data/data/cirr/captions/cap.rc2.val.json \
  -o /export/zqhe/retrieval_data/data/cirr/captions/cap.rc2.val.posneg.json \
  --device cuda \
  --batch_size 16 \
  --only_when_neg_or_change
