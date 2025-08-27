from langchain.chains import RetrievalQA

class RAG_Pipeline:
    def init(self, llm , vectorstore=None):
        self.llm = llm
        self.vectorstore = vectorstore

    def update_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore

    def query(self, question: str):
        if not self.vectorstore:
            return "No documents uploaded"

        retriever = self.vectorstore.as_retriever()  
        chain = RetrievalQA.from_llm(llm = self.llm,
                                     retriever = retriever)
        return chain.run(question)      