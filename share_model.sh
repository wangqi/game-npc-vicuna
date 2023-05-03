#!/bin/bash

echo "Share base model"
python ./tools/share_model.py --model_path models/game_npc_vicuna_base \
  --tokenizer_path config/tokenizer.model \
  --model_name wangqi776/game_npc_vicuna_base

echo "Share fine-tuned model"
python ./tools/share_model.py --model_path models/game_npc_vicuna \
  --tokenizer_path config/tokenizer.model \
  --model_name wangqi776/game_npc_vicuna
