from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def share(model_path, model_name):
    print(f"Share model @'%s' with name '%s' to huggingface." % (model_path, model_name))
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.push_to_hub(model_name, use_auth_token=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/game_npc_vicuna_base",
                        help="Give the model path that will be uploaded to Huggingface")
    parser.add_argument("--model_name", type=str, default="game_npc_vicuna_base",
                        help="Specify the target model name in Huggingface hub")
    args = parser.parse_args()
    share(args.model_path, args.model_name)
