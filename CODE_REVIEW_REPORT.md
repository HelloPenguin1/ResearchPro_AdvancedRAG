# Advanced RAG Project - Code Review Report
**Date:** November 21, 2025

---

## CRITICAL ERRORS (Will Cause Runtime Failures)

### 1. **SessionManager - Streamlit Dependency in FastAPI App** ⚠️ CRITICAL
**Location:** `session_manager.py` + `app.py:71`  
**Severity:** HIGH - Runtime Error

```python
# session_manager.py - Uses streamlit without checking context
if 'store' not in st.session_state:
    st.session_state.store = {}
```

**Problem:** 
- `session_manager.py` imports and uses `streamlit` (st)
- `app.py` uses `SessionManager()` in a **FastAPI** context, not Streamlit
- Streamlit is a frontend framework; FastAPI backend doesn't have `st.session_state`
- This will crash when `app.py` instantiates SessionManager on line 33

**Impact:** App startup fails immediately

**Fix Required:** Remove Streamlit dependency, use dict-based session storage

---

### 2. **DocumentProcessor - Missing extracted_tables Attribute** ⚠️ CRITICAL
**Location:** `document_process.py` + `app.py:85`  
**Severity:** HIGH - AttributeError

```python
# app.py line 85
session_manager.st.session_state.store = {} 
```

**Problem:**
- `document_process.py` constructor doesn't initialize `self.extracted_tables`
- But `app.py` line 85 tries to clear it: `document_processor.extracted_tables = []`
- The `MultimodalProcessor` extracts tables but doesn't return them separately
- `RAG_Pipeline.query()` tries to use `self.document_processor.get_table_context()` which doesn't exist

**Impact:** AttributeError when clearing vectorstore

---

### 3. **Incomplete Table Extraction** ⚠️ HIGH
**Location:** `mutimodal_processor.py:74-80`  
**Severity:** HIGH - Logic Error

```python
def _analyze_chunk_content(self, chunk):
    data = {'text': chunk.text, 'tables': [], 'images': []}
    
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            if element.category == 'Table':
                html = getattr(element.metadata, 'text_as_html', element.text)
                data['tables'].append(html)
    return data
```

**Problems:**
1. Only extracts tables but doesn't store them for later retrieval
2. Assumes `chunk.metadata.orig_elements` exists (not guaranteed)
3. No tables are actually stored/indexed for the retrieval pipeline
4. Tables are embedded in summaries but not separately searchable

**Impact:** Tables won't be properly indexed or retrievable

---

### 4. **RAG Pipeline Query Flow - Method Mismatch** ⚠️ HIGH
**Location:** `rag_pipeline.py:147-154` + referenced in `app.py:74`  
**Severity:** MEDIUM - Logic Error

```python
# rag_pipeline.py - Missing implementation
def set_document_processor(self, doc_processor):
    # Empty implementation - should store reference
```

**Problem:**
- `set_document_processor()` is defined but empty
- `query()` method assumes `self.document_processor` exists and has `get_table_context()`
- `get_table_context()` doesn't exist in `DocumentProcessor`
- Enhanced input never actually gets passed to the retriever

**Impact:** Table context isn't integrated into queries

---

## CODE INCONSISTENCIES

### 5. **Metadata Access Inconsistency** ⚠️ MEDIUM
**Locations:** 
- `mutimodal_processor.py:26` - `chunk.metadata.page_number`
- `mutimodal_processor.py:78` - `chunk.metadata.orig_elements`
- `mutimodal_processor.py:79` - `element.metadata.text_as_html`

**Problem:** Unstructured library's metadata access varies:
- Sometimes it's `element.metadata`, sometimes dictionary
- `chunk_by_title()` chunks may not have same metadata structure as `partition_pdf()` elements
- No validation before accessing nested attributes

**Risk:** Attribute errors with certain PDFs

---

### 6. **Hybrid Retriever + Compression Retriever Clash** ⚠️ MEDIUM
**Location:** `app.py:58-67`  
**Severity:** MEDIUM - Logic Flow Issue

```python
# Flow Problem:
semantic_retriever, syntactic_retriever = document_processor.create_retrievers(docs)  # Step 1
hybrid_retriever = rag_pipeline.create_hybrid_retriever(...)  # Step 2
compression_retriever = reranker.create_compression_retriever(hybrid_retriever)  # Step 3
rag_pipeline.create_rag_chain(compression_retriever)  # Step 4
```

**Problem:**
- `hybrid_retriever` is created but only used once to create `compression_retriever`
- The `compression_retriever` (reranker) uses the hybrid_retriever internally
- Then RAG chain uses compression_retriever
- Logic is correct but `hybrid_retriever` reference is stored in RAG pipeline but never used directly
- Potential issue: Compression retriever's `top_n=5` may be too low after hybrid retrieval

**Impact:** Reranker may work on limited document set

---

### 7. **Empty set_document_processor Implementation** ⚠️ MEDIUM
**Location:** `rag_pipeline.py:23-24`

```python
def set_document_processor(self, doc_processor):
    # Implementation missing - should store reference
```

**Should be:**
```python
def set_document_processor(self, doc_processor):
    self.document_processor = doc_processor
```

---

## PROMPT ENGINEERING ISSUES

### 8. **Table-Related Instructions Missing from Prompts** ⚠️ MEDIUM
**Location:** `rag_pipeline.py:85-99` (Answer Prompt)  
**Severity:** MEDIUM - Quality Issue

**Issue:** 
- Answer prompt mentions table handling (good)
- But reformulation prompt doesn't explicitly ask to preserve table references
- When user asks vague questions about tables, reformulation might lose context
- No explicit instruction to prioritize table-based queries

**Suggested Fix:**
Add table-aware reformulation prompt:
```python
"- If the question involves tables, data, or statistics, emphasize that in the reformulation"
"- Include table numbers or page references if mentioned in history"
```

---

### 9. **Prompt Doesn't Account for Summarized Tables** ⚠️ MEDIUM
**Location:** `mutimodal_processor.py:104-108`  
**Severity:** MEDIUM - Quality Issue

**Issue:**
- Tables are converted to summaries via LLM
- Answer prompt doesn't know these are summaries (vs original tables)
- Might lose precision in data interpretation
- No indication tables have been processed/summarized

---

## ARCHITECTURAL ISSUES

### 10. **No Validation of Retriever Chain** ⚠️ MEDIUM
**Location:** `app.py:72-77`

```python
# Verify state - but compression_retriever is not checked
if not rag_pipeline.vectorstore or not rag_pipeline.hybrid_retriever:
    raise HTTPException(status_code=500, detail="Failed to initialize retriever")
```

**Should check:**
- `compression_retriever` is initialized
- LLM is accessible
- Embedding model is loaded

---

### 11. **Session Storage Issues** ⚠️ MEDIUM
**Location:** `session_manager.py` + `app.py:87`

**Problem:**
- Deleting vectorstore tries to access `session_manager.st.session_state.store`
- This is Streamlit-specific code in FastAPI context
- Sessions don't properly clear on different instances

**Issue:** Stateful session management across multiple API calls

---

## MISSING IMPLEMENTATIONS

### 12. **No extracted_tables in DocumentProcessor** ⚠️ HIGH
**Status:** Not implemented but referenced

- `MultimodalProcessor` doesn't export tables separately
- `DocumentProcessor` doesn't have `extracted_tables` attribute
- `RAG_Pipeline.query()` tries to call non-existent `get_table_context()`

---

## SUMMARY TABLE

| Issue | Severity | Type | Fix Difficulty |
|-------|----------|------|-----------------|
| Streamlit in FastAPI | CRITICAL | Architecture | Medium |
| Missing extracted_tables | CRITICAL | Missing Code | High |
| Table extraction incomplete | HIGH | Logic | High |
| RAG query flow broken | HIGH | Missing Code | Medium |
| set_document_processor empty | MEDIUM | Missing Code | Low |
| Metadata inconsistency | MEDIUM | Logic | Medium |
| Retriever validation | MEDIUM | Error Handling | Low |
| Table-aware prompts | MEDIUM | Prompt | Low |
| Session management | MEDIUM | Architecture | High |
| **TOTAL ISSUES** | **9 CRITICAL/HIGH** | **Mixed** | **Varied** |

---

## RECOMMENDED FIX ORDER

1. **Fix SessionManager** - Remove Streamlit dependency (BLOCKS EXECUTION)
2. **Implement extracted_tables** - Add table tracking to DocumentProcessor
3. **Fix table extraction** - Make tables searchable separately
4. **Implement set_document_processor** - Store reference properly
5. **Fix RAG query flow** - Integrate table context properly
6. **Add validation checks** - Better error handling
7. **Enhance prompts** - Table-aware reformulation

---

## QUALITY NOTES FOR RAG ACCURACY

✓ **Good:**
- Hybrid retriever (semantic + syntactic) is excellent
- Reranker improves quality
- Conversational history maintained
- Table summarization helps with embeddings

⚠️ **Can Improve:**
- Table extraction is weak - not separately indexed
- Reranker top_n=5 might be limiting
- No boost for table-related queries
- Summaries may lose precision for numerical data

