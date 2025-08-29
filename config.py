from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
hf_embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs = {'normalize_embeddings':True},
)

llm = ChatGroq(model="llama-3.1-8b-instant", 
               groq_api_key=groq_api_key)

hf_reranker_encoder = "cross-encoder/ms-marco-MiniLM-L-6-v2"
