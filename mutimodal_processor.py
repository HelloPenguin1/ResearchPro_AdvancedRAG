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
            filename=filepath,  strategy="fast", infer_table_structure=False, extract_image_block_types=None, languages=["eng"]
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
        processed_docs = self._convert_chunks_without_summary(chunks)

        return processed_docs
    
    
    

    def _convert_chunks_without_summary(self, chunks) -> list[Document]:
        """
        Simple conversion: extract text + table HTML, no AI summarization.
        """
        processed_docs = []

        for chunk in chunks:
            text = chunk.text or ""

            # Extract table HTML if present
            tables = []
            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
                for element in chunk.metadata.orig_elements:
                    if getattr(element, "category", None) == "Table":
                        html = getattr(element.metadata, "text_as_html", element.text)
                        tables.append(html)

            doc = Document(
                page_content=text,
                metadata={
                    "source": "pdf",
                    "has_tables": len(tables) > 0,
                    "original_tables": tables,
                    "page_number": getattr(chunk.metadata, "page_number", None),
                }
            )

            processed_docs.append(doc)

        return processed_docs

    

    def _generate_ai_summary(self, text, tables, images) -> str:        
        # Construct context for the LLM
        context_str = f"TEXT:\n{text}\n\n"
        for i, table in enumerate(tables):
            context_str += f"TABLE {i+1}:\n{table}\n\n"

        prompt = f"""You are a concise research summarizer. Create a brief, searchable summary integrating text and tables.

        SUMMARY RULES:
        - Extract core findings, methodology, and key results from text
        - Convert tables to brief data statements with exact numbers (e.g., "Accuracy increased from 50% to 85%")
        - Preserve domain terms and metric names for searchability
        - Never invent data—only use what's explicitly shown
        - Keep under 200 words; prioritize key insights over completeness

        CONTENT:
        {context_str}

        SUMMARY (direct, no formatting tags):"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"Summary failed: {e}")
            return text