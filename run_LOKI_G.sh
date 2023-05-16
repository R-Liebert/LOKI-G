#!/bin/bash

# Run the Python script
python3 LOKI-G.py \
    --demonstration_path "./data" \
    --num_outputs 6 \
    --hard_switch_iter 18 \
    --random_sample_switch_iter \
    --il_epochs 10 \
    --rl_epochs 10 \
    --env_file "/path/to/your/env.py"
