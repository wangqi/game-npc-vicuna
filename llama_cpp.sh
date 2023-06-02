#!/bin/bash

MODEL_NAME="${1:-"game_npc_vicuna_base"}"
MODEL="models/$MODEL_NAME/ggml-q4_1.bin"
PROMPT_FILE_NAME="${2:-"prompt_en.txt"}"
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-./data/$PROMPT_FILE_NAME}
USER_NAME="${3:-"### User"}"
AI_NAME="${4:-"### AI"}"

echo "MODEL: $MODEL"
echo "PROMPT_TEMPLATE: $PROMPT_TEMPLATE"
echo "USER_NAME: $USER_NAME"
echo "AI_NAME: $AI_NAME"

# Adjust to the number of CPU cores you want to use.
N_THREAD="${N_THREAD:-8}"
# Number of tokens to predict (made it larger than default because we want a long interaction)
N_PREDICTS="${N_PREDICTS:-2048}"

# Note: you can also override the generation options by specifying them on the command line:
# For example, override the context size by doing: ./chatLLaMa --ctx_size 1024
GEN_OPTIONS="${GEN_OPTIONS:---ctx_size 512 --temp 0.7 --top_k 40 --top_p 0.5 --repeat_last_n 256 --batch_size 1024 --repeat_penalty 4.0}"

DATE_TIME=$(date +%H:%M)
DATE_YEAR=$(date +%Y)

TEMP_PROMPT_FILE=$(mktemp -t llamacpp_prompt.XXXXXXX.txt)

sed -e "s/\[\[USER_NAME\]\]/$USER_NAME/g" \
    -e "s/\[\[AI_NAME\]\]/$AI_NAME/g" \
    -e "s/\[\[DATE_TIME\]\]/$DATE_TIME/g" \
    -e "s/\[\[DATE_YEAR\]\]/$DATE_YEAR/g" \
     $PROMPT_TEMPLATE > $TEMP_PROMPT_FILE

echo "Generating with prompt: $TEMP_PROMPT_FILE"
cat $TEMP_PROMPT_FILE

# shellcheck disable=SC2086 # Intended splitting of GEN_OPTIONS
./tools/llama $GEN_OPTIONS \
  --interactive-first \
  --model "$MODEL" \
  --threads "$N_THREAD" \
  --n_predict "$N_PREDICTS" \
  --color --interactive \
  --file $TEMP_PROMPT_FILE \
  --reverse-prompt "$USER_NAME:"
  "$@"

#usage: ./tools/llama [options]
#options:
#  -h, --help            show this help message and exit
#  -i, --interactive     run in interactive mode
#  --interactive-first   run in interactive mode and wait for input right away
#  -ins, --instruct      run in instruction mode (use with Alpaca models)
#  --multiline-input     allows you to write or paste multiple lines without ending each in '\'
#  -r PROMPT, --reverse-prompt PROMPT
#                        halt generation at PROMPT, return control in interactive mode
#                        (can be specified more than once for multiple prompts).
#  --color               colorise output to distinguish prompt and user input from generations
#  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)
#  -t N, --threads N     number of threads to use during computation (default: 8)
#  -p PROMPT, --prompt PROMPT
#                        prompt to start generation with (default: empty)
#  -e                    process prompt escapes sequences (\n, \r, \t, \', \", \\)
#  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)
#  --prompt-cache-all    if specified, saves user input and generations to cache as well.
#                        not supported with --interactive or other interactive options
#  --random-prompt       start with a randomized prompt.
#  --in-prefix STRING    string to prefix user inputs with (default: empty)
#  --in-suffix STRING    string to suffix after user inputs with (default: empty)
#  -f FNAME, --file FNAME
#                        prompt file to start generation.
#  -n N, --n-predict N   number of tokens to predict (default: -1, -1 = infinity)
#  --top-k N             top-k sampling (default: 40, 0 = disabled)
#  --top-p N             top-p sampling (default: 0.9, 1.0 = disabled)
#  --tfs N               tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
#  --typical N           locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
#  --repeat-last-n N     last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)
#  --repeat-penalty N    penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)
#  --presence-penalty N  repeat alpha presence penalty (default: 0.0, 0.0 = disabled)
#  --frequency-penalty N repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)
#  --mirostat N          use Mirostat sampling.
#                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.
#                        (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
#  --mirostat-lr N       Mirostat learning rate, parameter eta (default: 0.1)
#  --mirostat-ent N      Mirostat target entropy, parameter tau (default: 5.0)
#  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS
#                        modifies the likelihood of token appearing in the completion,
#                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',
#                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'
#  -c N, --ctx-size N    size of the prompt context (default: 512)
#  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)
#  --no-penalize-nl      do not penalize newline token
#  --memory-f32          use f32 instead of f16 for memory key+value
#  --temp N              temperature (default: 0.8)
#  -b N, --batch-size N  batch size for prompt processing (default: 512)
#  --perplexity          compute perplexity over the prompt
#  --keep                number of tokens to keep from the initial prompt (default: 0, -1 = all)
#  --mlock               force system to keep model in RAM rather than swapping or compressing
#  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)
#  -ngl N, --n-gpu-layers N
#                        number of layers to store in VRAM
#  --mtest               compute maximum memory usage
#  --verbose-prompt      print prompt before generation
#  --lora FNAME          apply LoRA adapter (implies --no-mmap)
#  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter
#  -m FNAME, --model FNAME
#                        model path (default: models/7B/ggml-model.bin)