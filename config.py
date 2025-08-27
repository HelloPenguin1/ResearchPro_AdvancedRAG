from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
hf_embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    encode_kwargs = {'normalize_embeddings':True},
)

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)