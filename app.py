from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Annotated
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from config import llm, hyde_embedding, llm_summarize
from document_process import DocumentProcessor
from rag_pipeline import RAG_Pipeline
from postRetrievalReranker import ReRanker_Model
from config import hf_reranker_encoder
import os

class QueryRequest(BaseModel): 
    query: str


#initializing fastapi
app = FastAPI(
    title="RAG API v2",
    description="A simple RAG (Retrieval Augmented Generation) API using modularity",
    version="1.2.0"
)


#Instantiate classes
document_processor = DocumentProcessor()
rag_pipeline = RAG_Pipeline(llm)
reranker = ReRanker_Model(hf_reranker_encoder)





@app.get("/")
async def root():
    return {
        "message": "Welcome to the Advanced Research Assistant",
        "endpoints": {
            "POST /upload_file": "Upload a document for processing",
            "POST /query": "Query the uploaded documents"
        }
    }



    
## API endpoint for uplaoding docs
@app.post('/upload_file')
async def upload_file(file: Annotated[UploadFile, File(description="Upload a text document to process")]): 
    temp_file_path = None
    try:
        if not os.path.exists("temp"):
            os.makedirs("temp")

        temp_file_path = os.path.join("temp", f"temp_{file.filename}")
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Load and process document
        docs = document_processor.load_pdf(temp_file_path)
        chunks = document_processor.process_pdf(docs)

        # Create retrievers
        syntactic_retriever = document_processor.syntactic_retriever(chunks)
        semantic_retriever = document_processor.create_parent_retriever(docs, hyde_embedding)
        hybrid_retriever = rag_pipeline.create_hybrid_retriever(syntactic_retriever, semantic_retriever)
        
        compression_retriever = reranker.create_compression_retriever(hybrid_retriever)
        rag_pipeline.set_compression_retriever(compression_retriever)
       
       
        # Update vectorstore
        if document_processor.vectorstore:
            rag_pipeline.update_vectorstore(document_processor.vectorstore)
        else:
            raise HTTPException(status_code=500, detail="Vectorstore initialization failed")

        # Verify state
        if not rag_pipeline.vectorstore or not rag_pipeline.hybrid_retriever:
            raise HTTPException(status_code=500, detail="Failed to initialize retriever")

        # Cleanup
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return {"message": f"File uploaded and retriever initialized successfully."}
    
    except HTTPException as he:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise he
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    

## API endpoint for querying the retriever
@app.post('/query')
async def query_rag(query: QueryRequest):
    try:
        result = rag_pipeline.query(query.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")



@app.delete('/delete')
async def deletevectorstore():
    rag_pipeline.vectorstore = None
    document_processor.vectorstore = None
    rag_pipeline.hybrid_retriever = None
    rag_pipeline.compression_retriever = None  

    return {"Message": "Vectorstore cleared"}
