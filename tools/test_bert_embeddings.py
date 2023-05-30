from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from tools.embeddings import BertEmbeddings
import os, textwrap, sys


sys.path.append("./") # go to parent dir

ROOT_PATH = "."
# SENTENCE_EMBEDDING = "sentence-transformers/distiluse-base-multilingual-cased-v1"
SENTENCE_EMBEDDING = "shibing624/text2vec-base-chinese"
GGML_MODEL_PATH = ROOT_PATH + "/models/game_npc_vicuna_huntress/ggml-q4_1.bin"

file_path = ROOT_PATH + "/data/data.txt"
ggml_db_path = ROOT_PATH + "/db/ggml"
chatgpt_db_path = ROOT_PATH + "/db/chatgpt"

CHAT_MODEL = 'gpt-4'
# CHAT_MODEL = 'gpt-3.5-turbo'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with open(file_path, 'r', encoding="utf-8") as f:
    file_content = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
text_docs = text_splitter.create_documents([file_content])
print("total docs: ", len(text_docs))

ggml_persisted = True


def load_chroma_retriever(ggml_persisted=False):
    if not ggml_persisted:
        ggml_embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING)
        print("create ggml embeddings")
        ggml_db = Chroma.from_documents(documents=text_docs, embedding=ggml_embeddings, persist_directory=ggml_db_path)
        print("create ggml db")
        ggml_db.persist()
    else:
        print("Load ChromaDB from persisted path")
        ggml_embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING)
        ggml_db = Chroma(persist_directory=ggml_db_path, embedding_function=ggml_embeddings)
    return ggml_db


def load_faiss_retriever(ggml_persisted=False):
    if not ggml_persisted:
        # ggml_embeddings = BertEmbeddings()
        ggml_embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING)
        print("create ggml based on FAISS and embeddings:", SENTENCE_EMBEDDING)
        ggml_db = FAISS.from_documents(documents=text_docs, embedding=ggml_embeddings)
        ggml_db.save_local(folder_path=ggml_db_path)
        print("create ggml db")
    else:
        print("Load FAISS from persisted path and embeddings:", SENTENCE_EMBEDDING)
        # ggml_embeddings = BertEmbeddings()
        ggml_embeddings = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING)
        ggml_db = FAISS.load_local(folder_path=ggml_db_path, embeddings=ggml_embeddings)
    return ggml_db


def load_chatgpt_retriever(chatgpt_persisted=True):
    if not chatgpt_persisted:
        openai_embeddings = OpenAIEmbeddings()
        chatgpt_db = Chroma.from_documents(text_docs, openai_embeddings, persist_directory=chatgpt_db_path)
        chatgpt_db.persist()
    else:
        print("Load ChromaDB from persisted path")
        openai_embeddings = OpenAIEmbeddings()
        chatgpt_db = Chroma(persist_directory=chatgpt_db_path, embedding_function=openai_embeddings)
    return chatgpt_db

def query_retriever(query, retriever):
    # Load background docs if possible
    background_doc_list = retriever.get_relevant_documents(query=query)
    if len(background_doc_list) > 0:
        background = background_doc_list[0]
        return "\n".join(textwrap.wrap(background.page_content, width=80))
    return ""


ggml_db = load_chroma_retriever(ggml_persisted=False)
# ggml_db = load_faiss_retriever(ggml_persisted=True)
ggml_retriever = ggml_db.as_retriever()

chatgpt_db = load_chatgpt_retriever(chatgpt_persisted=True)
chatgpt_retriever = chatgpt_db.as_retriever()

# Test
query = "你认识艾丽娜吗？"
print("shibing624/text2vec-base-chinese")
print(query_retriever(query, ggml_retriever))
print("chatgpt")
print(query_retriever(query, chatgpt_retriever))
