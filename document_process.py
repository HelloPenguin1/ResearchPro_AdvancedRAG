from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS


class DocumentProcessor:
    def __init__(self):
        # self.embeddings = embeddings
        self.vectorstore = None




    def load_pdf(self, filepath:str):
        loader = UnstructuredPDFLoader(
                    file_path=filepath,
                    mode = "elements",
                    strategy = "fast",
                    languages=["eng"],
                    pdf_infer_table_structure = True,
                    include_page_breaks = True
                )
        docs = loader.load()
        return docs
    




    def process_pdf(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size = 800, 
                                                  chunk_overlap = 100,
                                                  separators=[
                                                      "\n\n",
                                                      "\n",
                                                      ". ",
                                                      " ",
                                                      ""
                                                  ])
        chunks = splitter.split_documents(docs)
        return chunks
    



    def syntactic_retriever(self, chunks):

        def simple_process(text):
            return text.lower().split()


        syntactic_retriever = BM25Retriever.from_documents(documents=chunks,
                                                          preprocess_func= simple_process)
        return syntactic_retriever
    
    


    def semantic_retriever(self, chunks, embeddings):
       self.vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
       semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
       

       return semantic_retriever

    

        