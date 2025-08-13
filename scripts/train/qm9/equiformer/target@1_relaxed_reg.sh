#!/bin/bash

# Loading the required module
source /etc/profile
# module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
# source activate th102_cu113_tgconda

conda activate equiformer_new


torchrun --standalone --nproc_per_node=1 main_qm9_reg.py \
    --output-dir 'models/qm9/equiformer_relaxed/se_l2/target@1/' \
    --model-name 'graph_attention_transformer_nonlinear_l2_relaxed' \
    --input-irreps '5x0e' \
    --target 1 \
    --data-path 'datasets/qm9' \
    --feature-type 'one_hot' \
    --batch-size 128 \
    --radius 5.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp \
    --eq-reg-non 0.01 \
    --eq-reg-eq 0.0 \
    --eq-reg-power 2 \
    --eq-reg-where 'relaxed-linear'
