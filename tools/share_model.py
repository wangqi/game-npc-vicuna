import os
import argparse
from transformers import AutoModel, AutoTokenizer

def main(args):
    # Load your model and tokenizer
    model = AutoModel.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model.push_to_hub(args.model_name)
    tokenizer.push_to_hub(args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your model directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to your tokenizer directory")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to upload")

    args = parser.parse_args()
    main(args)
