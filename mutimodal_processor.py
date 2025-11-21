from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from config import llm_summarize
import base64

class MultimodalProcessor:
    def __init__(self):
        self.llm = llm_summarize

    def load_and_process(self, filepath: str) -> list[Document]:
        # 1. Partition with hi_res strategy (Essential for tables)
        print("Partitioning document (this may take a while)...")
        elements = partition_pdf(
            filename=filepath,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True, 
            languages=["eng"]
        )
        
        # 2. Smart Chunking (Respects document structure)
        print("Chunking by title...")
        chunks = chunk_by_title(
            elements,
            max_characters=3000,
            new_after_n_chars=2400,
            combine_text_under_n_chars=500
        )
        
        # 3. AI Summarization of Tables/Images
        print("Summarizing mixed content...")
        documents = self._enrich_chunks_with_summaries(chunks)
        
        return documents

    def _enrich_chunks_with_summaries(self, chunks) -> list[Document]:
        processed_docs = []
    
        for chunk in chunks:
            # Detect content types in this chunk
            content_data = self._analyze_chunk_content(chunk)
            
            # If chunk has tables or images, generate a summary
            if content_data['tables'] or content_data['images']:
                enhanced_text = self._generate_ai_summary(
                    content_data['text'],
                    content_data['tables'],
                    content_data['images']
                )
            else:
                enhanced_text = content_data['text']
                

            doc = Document(
                page_content=enhanced_text,
                metadata={
                    "source": "pdf",
                    "has_tables": len(content_data['tables']) > 0,
                    "original_tables": content_data['tables'], 
                    "page_number": chunk.metadata.page_number
                }
            )
            processed_docs.append(doc)
            
        return processed_docs
    
    
    
    
    

    def _analyze_chunk_content(self, chunk):
        data = {'text': chunk.text, 'tables': [], 'images': []}
        
        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
            for element in chunk.metadata.orig_elements:
                if element.category == 'Table':
                    html = getattr(element.metadata, 'text_as_html', element.text)
                    data['tables'].append(html)
                        
        return data
    
    

    

    def _generate_ai_summary(self, text, tables, images) -> str:        
        # Construct context for the LLM
        context_str = f"TEXT CONTENT:\n{text}\n\n"
        for i, table in enumerate(tables):
            context_str += f"TABLE {i+1} DATA (HTML):\n{table}\n\n"

        prompt = f"""
        You are an expert technical writer. I have a document chunk containing text and data tables.
        
        1. Summarize the surrounding text.
        2. Analyze the tables and convert them into natural language statements (e.g. "The accuracy increased from 50% to 90%"). 
        3. Keep specific numbers, metric names, and entities intact so they are searchable.
        
        CONTENT TO PROCESS:
        {context_str}
        
        OUTPUT (Searchable Summary):
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"Summary failed: {e}")
            return text