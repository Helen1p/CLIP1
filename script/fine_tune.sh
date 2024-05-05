CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
--mode fine_tune \
--frozen_layers True \
--load_from_clip False \
> ./log_fine_tune.txt 2>&1 &