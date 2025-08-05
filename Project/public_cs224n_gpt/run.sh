#!/bin/bash

if [ "$1" = "last" ]; then
    python3 classifier.py --epochs 10 --fine-tune-mode last-linear-layer --use_gpu --batch_size 64 --hidden_dropout_prob 0.3 --lr 1e-3
elif [ "$1" = "all" ]; then
    python3 classifier.py --epochs 10 --fine-tune-mode full-model --use_gpu --batch_size 64 --hidden_dropout_prob 0.3 --lr 1e-5
else
    echo "unkown arg '$1', must be in { last-linear-layer, full-model }"
fi
