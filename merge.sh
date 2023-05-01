BASE_MODEL="models/game_npc_vicuna_base"
LORA_PATH="lora_out/final"
LORA_TOKEN="config"
MODEL_PATH="models/game_npc_vicuna"

echo "Merge the game_npc_vicuna_base model with fine-tuned weight to $LORA_PATH"
CUDA_VISIBLE_DEVICES=0 python tools/merge.py --base_model $BASE_MODEL --lora_model $LORA_PATH \
 --lora_token $LORA_TOKEN --output_dir $MODEL_PATH
