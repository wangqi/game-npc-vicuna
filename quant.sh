MODEL="models/game_npc_vicuna_base"
QUANT_MODEL="models/game_npc_vicuna_gptq"

echo "Quantized the $MODEL to 8bit and 128 groupsize to $QUANT_MODEL"
CUDA_VISIBLE_DEVICES=0 python tools/gptq/llama.py $MODEL c4 --wbits 8 --true-sequential --groupsize 128 \
  --save_safetensors $QUANT_MODEL/game_npc_vicuna-4bit-128g.safetensors \
  --save $QUANT_MODEL/game_npc_vicuna-4bit-128g.pt
