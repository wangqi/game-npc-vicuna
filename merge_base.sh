# Assign the first command line argument to a variable or use a default value
BASE_MODEL="${1:-"openaccess-ai-collective/wizard-mega-13b"}"
LORA_WEIGHT="${2:-"ziqingyang/chinese-llama-plus-lora-13b,ziqingyang/chinese-alpaca-plus-lora-13b"}"
MODEL_PATH="${3:-"models/wizard-mega_chinese-alpaca-plus-lora-13b"}"
OUT_TYPE="f16"
QUANT_TYPE="q4_1"

mkdir -p $MODEL_PATH
echo "The BASE_MODEL is: $BASE_MODEL"
echo "The LORA_WEIGHT is: $LORA_WEIGHT"
echo "The output model path is: $MODEL_PATH, the output type is: $OUT_TYPE"

echo "Merge the base '$BASE_MODEL' with '$LORA_WEIGHT' LoRA weight"
echo python tools/merge/merge_llama_with_chinese_lora.py --base_model $BASE_MODEL --lora_model $LORA_WEIGHT --output_dir $MODEL_PATH
python tools/merge/merge_llama_with_chinese_lora.py --base_model $BASE_MODEL --lora_model $LORA_WEIGHT --output_dir $MODEL_PATH

echo python tools/convert_ggml.py --outtype $OUT_TYPE --outfile $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/consolidated.00.pth
python tools/convert_ggml.py --outtype $OUT_TYPE --outfile $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/consolidated.00.pth

echo "quantize the model ggml-$OUT_TYPE.bin $QUANT_TYPE"
echo tools/quantize $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/ggml-$QUANT_TYPE.bin $QUANT_TYPE
tools/quantize $MODEL_PATH/ggml-$OUT_TYPE.bin $MODEL_PATH/ggml-$QUANT_TYPE.bin $QUANT_TYPE

echo "Compress the model to 4bit 128 group safetensors format. Need to install gptq.llama first"
echo "Refer to https://github.com/0cc4m/GPTQ-for-LLaMa"
echo python -m gptq.llama $MODEL_PATH c5 --wbits 4 --true-sequential --act-order --save_safetensors $MODEL_PATH/gptq-4bit-128g.safetensors
CUDA_VISIBLE_DEVICES=0 python -m gptq.llama $MODEL_PATH c4 --wbits 4 --true-sequential --act-order \
  --save_safetensors $MODEL_PATH/gptq-4bit-128g.safetensors
