#!/bin/bash

MODEL_NAME="${1:-"game_npc_vicuna_base"}"
MODEL="models/$MODEL_NAME/ggml-q5_0.bin"

echo "MODEL: $MODEL"

# Check if the second command-line argument is set
if [ -n "$2" ]; then
    TEXT_FILE_NAME=$2
    echo "TEXT_FILE_NAME: $TEXT_FILE_NAME"
    python tools/langchain_qa.py --model_path $MODEL --file_path $TEXT_FILE_NAME --chain_type stuff
else
    python tools/langchain_qa.py --model_path $MODEL  --chain_type stuff
fi
