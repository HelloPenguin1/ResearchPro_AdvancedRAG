from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from config import hf_embeddings

from mutimodal_processor import MultimodalProcessor

class DocumentProcessor:
    def __init__(self):
        self.vectorstore = None
        self.multimodal_processor = MultimodalProcessor()
        self.processed_docs = []

    def load_and_process_pdf(self, filepath: str):

        self.processed_docs = self.multimodal_processor.load_and_process(filepath)
        print(f"Generated {len(self.processed_docs)} enriched documents.")
        
        return self.processed_docs

    def create_retrievers(self, docs):
        """
        Creates Semantic (FAISS) and Syntactic (BM25) retrievers
        """
        # 1. Semantic Retriever (Vector Search)
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(docs, hf_embeddings)
        semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # 2. Syntactic Retriever (Keyword Search)
        print("Creating BM25 retriever...")
        syntactic_retriever = BM25Retriever.from_documents(
            documents=docs,
            preprocess_func=lambda text: text.lower().split()
        )
        syntactic_retriever.k = 4

        return semantic_retriever, syntactic_retriever




    def get_statistics(self) -> dict:
        return {
            "processed_documents": len(self.processed_docs),
            "vectorstore_ready": self.vectorstore is not None
        }