CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
--mode train \
> ./log_train.txt 2>&1 &