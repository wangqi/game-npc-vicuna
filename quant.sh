MODEL="ggml-f16.bin"
QUANT_PATH="models/game_npc_vicuna_huntress"
QUANT_TYPE="q4_1"

echo "Quantized the $QUANT_PATH/$MODEL using $QUANT_TYPE"
#CUDA_VISIBLE_DEVICES=0 python tools/gptq/llama.py $MODEL c4 --wbits 8 --true-sequential --groupsize 128 \
#  --save_safetensors $QUANT_MODEL/game_npc_vicuna-4bit-128g.safetensors \
#  --save $QUANT_MODEL/game_npc_vicuna-4bit-128g.pt

# quantize the model to 4-bits (using q4_1 method)
./tools/quantize $QUANT_PATH/$MODEL $QUANT_PATH/ggml_$QUANT_TYPE.bin $QUANT_TYPE
