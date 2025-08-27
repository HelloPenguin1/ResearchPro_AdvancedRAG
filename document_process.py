from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

class DocumentProcessor:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None

    def process_text(self, text:str):
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
        chunks = splitter.split_text(text)
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)

        return len(chunks)

        