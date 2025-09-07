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
            Given a chat history and a recent user question which might 
            reference context in the chat history, formulate a standalone
            question which can be understood without the chat history.
            DO NOT answer the question. Just reformulate the question if needed
            else return as it is.
        """

        return ChatPromptTemplate.from_messages([
            ("system", reform_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])


    def create_answer_prompt(self):
        answer_sys_prompt = """
            You are an expert research assistant for question answering tasks.
            Make sure to answer the questions as accurately as possible without 
            leaving any details using the following retrieved context.
            Give a complete answer to the question.

            Context: {context} 
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