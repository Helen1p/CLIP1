CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
--mode test \
> ./log_test.txt 2>&1 &