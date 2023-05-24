#!/bin/bash

MODEL_NAME="${1:-"game_npc_vicuna_base"}"
OUT_TYPE="f16"
QUANT_TYPE="q4_1"

CONSOLIDATED_PTH="^consolidated\.([0-9]+)\.pth$"
PYTORCH_MODEL="^pytorch_model-(.*)\.bin$"

# Search for files matching the regex pattern in the current directory
MODEL_FILE=$(find . -type f -regex "CONSOLIDATED_PTH" | head -n 1)

if [ -e "models/$MODEL_NAME/$MODEL_FILE" ]; then
    echo python tools/convert_ggml.py --outtype $OUT_TYPE --outfile models/$MODEL_NAME/ggml-$OUT_TYPE.bin models/$MODEL_NAME/$MODEL_FILE
    echo python tools/convert_ggml.py --outtype $OUT_TYPE --outfile models/$MODEL_NAME/ggml-$OUT_TYPE.bin models/$MODEL_NAME/$MODEL_FILE
else
    MODEL_FILE=$(find . -type f -regex "PYTORCH_MODEL" | head -n 1)
    echo python tools/convert_ggml.py --outtype $OUT_TYPE --outfile models/$MODEL_NAME/ggml-$OUT_TYPE.bin models/$MODEL_NAME/$MODEL_FILE
    echo python tools/convert_ggml.py --outtype $OUT_TYPE --outfile models/$MODEL_NAME/ggml-$OUT_TYPE.bin models/$MODEL_NAME/$MODEL_FILE
fi

echo "Quantize the model models/$MODEL_NAME/ggml-$OUT_TYPE.bin to $QUANT_TYPE"
echo ./tools/quantize models/$MODEL_NAME/ggml-$OUT_TYPE.bin models/$MODEL_NAME/ggml-$QUANT_TYPE.bin
echo ./tools/quantize models/$MODEL_NAME/ggml-$OUT_TYPE.bin models/$MODEL_NAME/ggml-$QUANT_TYPE.bin $QUANT_TYPE

#echo "Quantized the models/$MODEL_NAME/$MODEL using $QUANT_TYPE"
#echo python tools/gptq/llama.py models/$MODEL_NAME/ c4 --wbits 4 --true-sequential --groupsize 128
#CUDA_VISIBLE_DEVICES=0 python tools/gptq/llama.py models/$MODEL_NAME/ c4 --wbits 4 --true-sequential --groupsize 128 \
#  --save_safetensors models/$MODEL_NAME/gptq-4bit-128g.safetensors \
#  --save $QUANT_MODEL/game_npc_vicuna-4bit-128g.pt
