#!/usr/bin/env bash

accelerate launch train_mt.py --config config/vaihingen_thick_adahdit_dinojoint_biparallel_seg_B2_100k.yml
accelerate launch train_mt.py --config config/vaihingen_thick_adahdit_dinojoint_bisequential_seg_B2_100k.yml
