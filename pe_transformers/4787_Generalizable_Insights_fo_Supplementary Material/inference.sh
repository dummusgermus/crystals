#!/bin/bash

python inference.py \
    --inference_task $1 \
    --task $2 \
    --checkpoint $4 \
    --shots $3 \
    --model_size $5 \
    --pe $6 \
    --rwse_steps $7 \
    --rrwp_steps $7 \
    --lpe_num_eigvals $7 \
    --seed $8 \
    --root $9 \
    --extrapolate_start 20 \
    --extrapolate_end 257