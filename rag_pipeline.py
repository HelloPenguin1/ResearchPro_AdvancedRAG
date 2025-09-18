from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory


class RAG_Pipeline:
    def __init__(self, llm , vectorstore=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.hybrid_retriever = None
        self.compression_retriever = None
        self.conversational_rag = None

        self.reformulation_prompt = self.create_reformulation_prompt()
        self.answer_prompt  = self.create_answer_prompt()


    def create_reformulation_prompt(self):
        reform_sys_prompt = """
        You are a research question reformulator.
        Given the conversation history and the latest user query, rewrite the query 
        into a clear, self-contained research question. 

        Guidelines:
        - Preserve the user’s intent.
        - Expand abbreviations or vague references (e.g., "it", "they") using chat history.
        - Do NOT answer the question.
        - If the question is already standalone, return it unchanged.
        """

        return ChatPromptTemplate.from_messages([
            ("system", reform_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])


    def create_answer_prompt(self):
        answer_sys_prompt = """
        You are an expert research assistant specialized in providing accurate, 
        detailed, and well-structured answers based on retrieved documents.

        Use the following rules:
        1. Base your answer ONLY on the provided context. If the context does not 
           contain enough information, clearly say so.
        2. Provide a complete, well-structured explanation (not just a short answer).
        3. When applicable, summarize key points, compare perspectives, and provide 
           nuanced insights.
        4. Maintain an academic, professional tone suitable for research use.
        5. Do NOT fabricate references or information.

        Context:
        {context}
        """

        return ChatPromptTemplate.from_messages([
            ("system", answer_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    
    def update_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore

    def create_hybrid_retriever(self, syntactic_retriever, semantic_retriever):
        self.hybrid_retriever = EnsembleRetriever(retrievers = [syntactic_retriever, semantic_retriever],
                                             weights = [0.6,0.4])
        return self.hybrid_retriever
    
    def set_compression_retriever(self, compression_retriever):
        self.compression_retriever = compression_retriever
    


    def create_rag_chain(self, retriever):
        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            retriever,
            self.reformulation_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            self.answer_prompt
        )

        rag_pipeline = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        return rag_pipeline
    
    
    def create_conversational_chain(self, rag_chain, get_session_history_func):
        self.conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_session_history_func,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        return self.conversational_rag


    def query(self, question: str, session_id: str) -> str:
        if not self.conversational_rag:
            return "Conversational chain not initialized"
        
        try:
            response = self.conversational_rag.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )
            return response['answer']
        except Exception as e:
            return f"Error processing query: {str(e)}"