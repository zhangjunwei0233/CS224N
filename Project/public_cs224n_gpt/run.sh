#!/bin/bash

# Sentiment Classification Tasks
if [ "$1" = "last" ]; then
    echo "Running sentiment classification with last-linear-layer fine-tuning..."
    python3 classifier.py --epochs 10 --fine-tune-mode last-linear-layer --use_gpu --batch_size 64 --hidden_dropout_prob 0.3 --lr 1e-3
elif [ "$1" = "all" ]; then
    echo "Running sentiment classification with full-model fine-tuning..."
    python3 classifier.py --epochs 10 --fine-tune-mode full-model --use_gpu --batch_size 64 --hidden_dropout_prob 0.3 --lr 1e-5

# Paraphrase Detection Tasks
elif [ "$1" = "para" ]; then
    echo "Running paraphrase detection with GPT-2 base model..."
    python3 paraphrase_detection.py --epochs 10 --use_gpu --batch_size 8 --lr 1e-5 --model_size gpt2
elif [ "$1" = "para-medium" ]; then
    echo "Running paraphrase detection with GPT-2 medium model..."
    python3 paraphrase_detection.py --epochs 10 --use_gpu --batch_size 4 --lr 1e-5 --model_size gpt2-medium
elif [ "$1" = "para-large" ]; then
    echo "Running paraphrase detection with GPT-2 large model..."
    python3 paraphrase_detection.py --epochs 10 --use_gpu --batch_size 2 --lr 1e-5 --model_size gpt2-large

# Debug/Validation Modes (Small datasets, fewer epochs)
elif [ "$1" = "debug-para" ]; then
    echo "Running paraphrase detection DEBUG mode (small dataset, few epochs)..."
    python3 paraphrase_detection.py --epochs 2 --use_gpu --batch_size 8 --lr 1e-5 --model_size gpt2 \
        --max_train_size 1000 --max_dev_size 500 --max_test_size 500
elif [ "$1" = "debug-sentiment" ]; then
    echo "Running sentiment classification DEBUG mode (reduced epochs)..."
    python3 classifier.py --epochs 2 --fine-tune-mode full-model --use_gpu --batch_size 32 --hidden_dropout_prob 0.3 --lr 1e-5

else
    echo "Unknown argument '$1'"
    echo "Available options:"
    echo "  SENTIMENT CLASSIFICATION:"
    echo "    last         - Sentiment classification (last-linear-layer fine-tuning)"
    echo "    all          - Sentiment classification (full-model fine-tuning)"
    echo "  PARAPHRASE DETECTION:"
    echo "    para         - Paraphrase detection (GPT-2 base)"
    echo "    para-medium  - Paraphrase detection (GPT-2 medium)"
    echo "    para-large   - Paraphrase detection (GPT-2 large)"
    echo "  DEBUG/VALIDATION MODES:"
    echo "    debug-para   - Paraphrase detection (100 train, 50 dev/test, 2 epochs)"
    echo "    debug-sentiment - Sentiment classification (2 epochs, smaller batch)"
fi
