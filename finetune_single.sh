DATA_PATH="./data/data.json"
#DATA_PATH="./data/merge.json"
OUTPUT_PATH="lora_out"
MODEL_PATH="models/game_npc_vicuna_base"
# lora_checkpoint="./checkpoints/chinese-vicuna-lora-8b-belle-and-guanaco"
from_data_beginning=True

python tools/finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size 0.3 \
--prompt_path data/train_tpl.txt \
--target_models q_proj,k_proj,v_proj,down_proj,gate_proj,up_proj
# --resume_from_checkpoint $lora_checkpoint \
# --ignore_data_skip $from_data_beginning

mkdir -p $OUTPUT_PATH/final/
cp lora_out/adapter_config.json $OUTPUT_PATH/final/
cp lora_out/adapter_model.bin $OUTPUT_PATH/final/
