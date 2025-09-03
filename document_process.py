from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore


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
        self.chunks = splitter.split_documents(docs)
        return self.chunks
    



    def syntactic_retriever(self, chunks):
        def simple_process(text):
            return text.lower().split()


        syntactic_retriever = BM25Retriever.from_documents(documents=chunks,
                                                          preprocess_func= simple_process)
        return syntactic_retriever
    
    
    def create_parent_retriever(self, docs, embeddings):
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 100)

        self.vectorstore = FAISS.from_texts(texts=["init"], embedding=embeddings)

        parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore = InMemoryStore(),
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )

        parent_retriever.add_documents(docs)
        return parent_retriever

    

        