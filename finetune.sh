# Suitable for 4 GPUs
TOT_CUDA="0,1,2,3"
CUDAs=(${TOT_CUDA//,/ })
echo "CUDAs: $CUDAs"
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="./data/data.json"
OUTPUT_PATH="lora_out"
MODEL_PATH="models/game_npc_vicuna_base"
TEST_SIZE=0.3
TOKENIZER_PATH="config/chinese_llama_alpaca/"

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT tools/finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE \
--prompt_path data/train_tpl.txt \
--micro_batch_size 8 \
--batch_size 64 \
--target_models q_proj,k_proj,v_proj \
--tokenizer_path $TOKENIZER_PATH \
--offload_dir /data/offload_dir

mkdir -p $OUTPUT_PATH/final/
cp adapter_config.json $OUTPUT_PATH/final/
cp adapter_model.bin $OUTPUT_PATH/final/
