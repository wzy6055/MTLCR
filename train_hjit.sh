#!/bin/bash

#cd /224010098/workspace/JiTCR
#export CUDA_VISIBLE_DEVICES=0,1
#accelerate launch train.py --config "config/changsha256_jit_B16_200k.yml"

#cd /224010098/workspace/JiTCR
#echo "Running config/vaihingen_thick_hjit_B2_vpvl_200k.yml"
#accelerate launch train_hjit.py --config config/vaihingen_thick_hjit_B2_vpvl_200k.yml

#echo "Running config/vaihingen_thick_hhjit_B2_xpvl_200k.yml"
#accelerate launch train_hjit.py --config config/vaihingen_thick_hhjit_B2_xpvl_200k.yml

echo "Running config/ablation_hjit/vaihingen_thick_hjit_B4_xpvl_200k.yml"
accelerate launch train_hjit.py --config config/ablation_hjit/vaihingen_thick_hjit_B2_xpvl_200k.yml

echo "Running config/ablation_hjit/vaihingen_thick_hjit_L4_xpvl_200k.yml"
accelerate launch train_hjit.py --config config/ablation_hjit/vaihingen_thick_hjit_L4_xpvl_200k.yml

