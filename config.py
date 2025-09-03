from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import HypotheticalDocumentEmbedder
import torch

import os
from dotenv import load_dotenv
load_dotenv()


## for increased efficiency
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)


groq_api_key = os.getenv("GROQ_API_KEY")

##############################################################################################

hf_embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs = {'normalize_embeddings':True},
)

llm = ChatGroq(model="openai/gpt-oss-20b", 
               groq_api_key=groq_api_key)

llm_summarize = ChatGroq(model="llama-3.1-8b-instant", 
               groq_api_key=groq_api_key)


##############################################################################################


hf_reranker_encoder = "cross-encoder/ms-marco-MiniLM-L-6-v2"

##############################################################################################

hyde_base_embedding =  HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    encode_kwargs = {'normalize_embeddings':True},
)

hyde_embedding = HypotheticalDocumentEmbedder.from_llm(llm = llm, 
                                              base_embeddings = hyde_base_embedding,
                                              prompt_key="sci_fact")





