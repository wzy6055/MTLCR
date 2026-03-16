#!/bin/bash

#cd /224010098/workspace/JiTCR
#export CUDA_VISIBLE_DEVICES=0,1
#accelerate launch train.py --config "config/changsha256_jit_B16_200k.yml"

#cd /224010098/workspace/JiTCR
#for f in config/ablation_b4/*.yml; do
#    echo "Running $f"
#    accelerate launch train.py --config "$f"
#done

cd /224010098/workspace/JiTCR
for f in config/ablation_repa/*_repa_200k.yml; do
    echo "Running $f"
    accelerate launch train_adahdit_repa.py --config "$f"
done

