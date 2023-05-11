import streamlit as st
import json, os

from langchain import LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
import torch


# Load AI models from Hugging Face
def load_models(config_file):
    device = "cpu"
    if torch.cuda.is_available():
        print("using GPU")
        device = 0

    with open(config_file, 'r') as f:
        model_configs = json.load(f)

    models = []
    for model_config in model_configs:
        model_name = model_config.get('name', "model_rating")
        print("-------------------- Load Model ----------------------")
        print("model_name: ", model_name)
        model_id = model_config['model_id']
        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map={"": device})

        lora_id = model_config.get('lora_id', None)
        if lora_id:
            tokenizer = AutoTokenizer.from_pretrained(lora_id)
            pipe = pipeline(model_name, model=lora_id, tokenizer=tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            pipe = pipeline(model_name, model=model, tokenizer=tokenizer)

        local_llm = HuggingFacePipeline(pipeline=pipe)

        models.append({'name': model_name, 'model': local_llm})

    return models


# Call ChatGPT 4 API to rate answers

# Call OpenAI API to rate answers
def rate_answers(answers):
    api_key = os.environ.get("OPENAI_API_KEY")
    api = OpenAIAPI(api_key)

    ratings = []
    for answer in answers:
        prompt = f"Rate the following answer on a scale of 1 to 10 (10 being the highest): {answer}"
        response = api.complete(prompt, max_tokens=5)
        rating = int(response["choices"][0]["text"].strip())
        ratings.append(rating)

    return ratings


# Main function
def main():
    config_file = "test/config.json"
    models = load_models(config_file)

    st.title("AI Model Comparison")

    question = st.text_input("Enter your question:")
    answers = []

    if question:
        for model in models:
            llm_chain = LLMChain(prompt=question, llm=model)
            answer = llm_chain.run(question)
            answers.append(answer)

        # Display answers in a table
        st.write("AI Model Answers:")
        cols = st.columns(len(models))

        for i, answer in enumerate(answers):
            cols[i].write(answer)

        # Button to rate answers
        if st.button("Rate them"):
            ratings = rate_answers(answers)

            # Display ratings below the answers
            st.write("Ratings:")
            rating_cols = st.columns(len(models))

            for i, rating in enumerate(ratings):
                rating_cols[i].write(rating)


if __name__ == "__main__":
    main()
