from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever


class RAG_Pipeline:
    def __init__(self, llm , vectorstore=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.hybrid_retriever = None
        self.compression_retriever = None

    def update_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore

    def create_hybrid_retriever(self, syntactic_retriever, semantic_retriever):
        self.hybrid_retriever = EnsembleRetriever(retrievers = [syntactic_retriever, semantic_retriever],
                                             weights = [0.6,0.4])
        return self.hybrid_retriever
    
    def set_compression_retriever(self, compression_retriever):
        self.compression_retriever = compression_retriever
    

    def query(self, question: str):
        if not self.vectorstore or not self.hybrid_retriever:
            return "No documents uploaded or hybrid retriever not initialized"
 
        chain = RetrievalQA.from_llm(llm = self.llm,
                                     retriever = self.compression_retriever )
        return chain.run(question)      
    



    