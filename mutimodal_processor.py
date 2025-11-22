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
   
        print("Fast scan to detect table/image pages...")
        fast_scan = partition_pdf(
            filename=filepath,
            strategy="fast",
            infer_table_structure=False,
            extract_image_block_types=None,
            languages=["eng"]
        )
        pages_with_tables = set()

        # Detect which pages need hi_res
        for el in fast_scan:
            category = getattr(el, "category", None)
            text = getattr(el, "text", "") or ""
            page = getattr(el.metadata, "page_number", None)

            if page is None:
                continue

            if (
                category in ("Table", "Image") or
                "table" in text.lower() or
                "figure" in text.lower()
            ):
                pages_with_tables.add(page)

        print(f"Detected table/image pages: {sorted(list(pages_with_tables))}")
        
        

        # Step 2: If no tables/images → use fast output only
        if not pages_with_tables:
            print("No complex elements detected. Using fast scan for all pages.")
            elements = fast_scan
        else:
            print("Running hi_res selectively on visual pages...")
            hi_res_elements = partition_pdf(
                    filename=filepath,
                    strategy="hi_res",
                    infer_table_structure=True,
                    extract_image_block_types=["Table", "Image"],
                    extract_image_block_to_payload=False,
                    languages=["eng"],
                    page_range=",".join(str(p) for p in pages_with_tables)
                )
           

            # Merge
            elements = []
            for el in fast_scan:
                page = getattr(el.metadata, "page_number", None)
                if page in pages_with_tables:
                    continue
                elements.append(el)

            elements.extend(list(hi_res_elements))

        # Step 3: Chunking
        print("Chunking by title...")
        chunks = chunk_by_title(
            elements,
            max_characters=3000,
            new_after_n_chars=2400,
            combine_text_under_n_chars=500
        )

        # Step 4: Convert to Document objects 
        processed_docs = self._enrich_chunks_with_summaries(chunks)

        return processed_docs
    
    
    

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
        You are an expert research analyst and technical writer specializing in precise data interpretation and document summarization.

        TASK: Create a comprehensive, searchable summary that integrates text and tabular data with precision and clarity.

        GUIDELINES FOR SUMMARIZATION:

        TEXT ANALYSIS:
        - Preserve the core arguments, findings, and conclusions from the text
        - Extract and highlight the primary research question or objective
        - Identify and include key methodologies and approaches mentioned
        - Retain important qualifications, limitations, and caveats
        - Use precise academic language; avoid over-simplification

        TABLE ANALYSIS:
        - Convert tabular data into clear, grammatically correct statements
        - FOR EACH TABLE: Identify the key metrics, dimensions (rows/columns), and comparison points
        - Express relationships and trends explicitly (e.g., "metric X increased by Y% from 2020 to 2021")
        - Extract specific numerical values and preserve them exactly as shown
        - Note any patterns, outliers, or significant findings visible in the data
        - If the table shows comparisons (A vs B), state the differences clearly

        INTEGRATION REQUIREMENTS:
        - Connect table findings with supporting textual context from the surrounding paragraphs
        - Clearly mark which data points come from tables vs. text for traceability
        - Use phrases like "According to Table X..." or "The data shows..." for attribution
        - Maintain logical flow between text summaries and table insights

        CRITICAL CONSTRAINTS:
        1. ACCURACY: Never invent, extrapolate, or assume data not explicitly shown
        2. SPECIFICITY: Include ALL numerical values, percentages, and measurements exactly as presented
        3. SEARCHABILITY: Use domain-specific terms, metric names, and entity names that appear in the original
        4. COMPLETENESS: Don't omit important columns, rows, or comparisons from tables
        5. CLARITY: Explain technical metrics and abbreviations found in tables

        FORMAT YOUR OUTPUT AS:
        [INTEGRATED SUMMARY]
        [Summary of key findings from text]
        [Detailed table insights with numerical data]
        [Connections between text and tabular evidence]
        [/INTEGRATED SUMMARY]

        CONTENT TO PROCESS:
        {context_str}

        OUTPUT (Comprehensive Searchable Summary):
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"Summary failed: {e}")
            return text