#!/bin/bash

python main.py \
    --task $1 \
    --model $2 \
    --model_size $3 \
    --pe $4 \
    --rwse_steps $5 \
    --rrwp_steps $5 \
    --lpe_num_eigvals $5 \
    --seed $6 \
    --root $7 \
    --learning_rate $8 \
    --path $9 \
    --save_checkpoint 

