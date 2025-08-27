from fastapi import FastAPI, UploadFile, File
from typing import Annotated
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from config import llm, hf_embeddings
from document_process import DocumentProcessor
from rag_pipeline import RAG_Pipeline


class QueryRequest(BaseModel): 
    query: str


#initializing fastapi
app = FastAPI(
    title="RAG API v2",
    description="A simple RAG (Retrieval Augmented Generation) API using modularity",
    version="1.2.0"
)

app.state.vectorstore = None

#Instantiate classes
document_processor = DocumentProcessor(hf_embeddings)
rag_pipeline = RAG_Pipeline(llm, app.state.vectorstore)


@app.get("/")
async def root():
    return {
        "message": "Welcome to the RAG API pra ctice",
        "endpoints": {
            "POST /upload_file": "Upload a document for processing",
            "POST /query": "Query the uploaded documents"
        }
    }

    
## API endpoint for uplaoding docs
@app.post('/upload_file')
async def upload_file(file: Annotated[UploadFile, File(description="Upload a text document to process")]):
    text = await file.read()
    text=text.decode('utf-8', errors = "ignore")

    chunks = document_processor.process_text(text)
    rag_pipeline.update_vectorstore(document_processor.vectorstore)

    return {"Message": "File uploaded successfully",
            "Chunks": chunks}
    

## API endpoint for querying the retriever
@app.post('/query')
async def query_rag(query: QueryRequest):
    result = rag_pipeline.query(query.query)
    return {"response": result}


@app.delete('/delete')
async def deletevectorstore():
    rag_pipeline.vectorstore = None
    document_processor.vectorstore = None
    return {"Message": "Vectorstore cleared"}
