BASE_MODEL="decapoda-research/llama-7b-hf"
LORA_PATH="lora_out/final"
LORA_TOKEN="config"
MODEL_PATH="models/game_npc_vicuna"

python tools/merge.py --base_model $BASE_MODEL --lora_model $LORA_PATH --lora_token $LORA_TOKEN --output_dir $MODEL_PATH
