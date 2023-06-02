DATA_PATH="./data/data.json"
#DATA_PATH="./data/merge.json"
OUTPUT_PATH="lora_out"
MODEL_PATH="models/game_npc_vicuna_base"
# lora_checkpoint="./checkpoints/chinese-vicuna-lora-8b-belle-and-guanaco"
TEST_SIZE=0.3
TOKENIZER_PATH="config/chinese_llama_alpaca/"

python tools/finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE \
--prompt_path data/train_tpl.txt \
--target_models q_proj,k_proj,v_proj \
--tokenizer_path $TOKENIZER_PATH

mkdir -p $OUTPUT_PATH/final/
cp lora_out/adapter_config.json $OUTPUT_PATH/final/
cp lora_out/adapter_model.bin $OUTPUT_PATH/final/
