import os, shutil, re
from typing import List, Tuple, Dict
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

def initialize_variables():
    DOCX_PATH = "tables_doc_LLM_friendly.docx"
    loader = Docx2txtLoader(DOCX_PATH)
    docs = loader.load()
    raw_text = "\n".join(d.page_content for d in docs)

    table_blocks, relationships_text = split_tables_and_relationships(raw_text)
    parent_docs = make_parent_documents(table_blocks)
    join_docs = parse_relationships_to_join_docs(relationships_text)
    
    OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    schema_persist_dir = "chroma_olist_schema_child_v2"
    joins_persist_dir  = "chroma_olist_joins_v2"

    for d in (schema_persist_dir, joins_persist_dir):
        if os.path.isdir(d):
            try:
                shutil.rmtree(d)
            except:
                print("Directory being used already. Code has rerun without clearing previous cache. Skipping the deletion step.")

    # --- 1) Embeddings ---
    # Set your key in env before running:  export OPENAI_API_KEY=sk-...
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    # --- 2) Vector store for SCHEMA CHILDREN (PDR will populate it) ---
    schema_child_vs = Chroma(
        collection_name = "olist_schema_child_v2",
        embedding_function = embeddings,
        persist_directory = schema_persist_dir,
    )

    # --- 3) Docstore to hold PARENT docs (PDR returns parents at query time) ---
    parent_docstore = InMemoryStore()

    # --- 4) Child splitter: EXACT same params you used before ---
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 40,
        separators = ["\nColumns:", "\nNotes:", "\nPurpose:", "\n\n"],
    )

    # --- 5) ParentDocumentRetriever (splits parents -> children, embeds children, stores mapping) ---
    schema_retriever = ParentDocumentRetriever(
        vectorstore = schema_child_vs,
        docstore = parent_docstore,
        child_splitter = child_splitter,
        search_kwargs = {"k": 6},  # tune later
    )

    # --- 6) Index TABLE PARENTS (PDR will create children internally with the SAME splitter) ---
    _ = schema_retriever.add_documents(parent_docs)

    # --- 7) JOINS: atomic one-edge-per-doc in their own collection ---
    joins_vs = Chroma.from_documents(
        documents = join_docs,   # from your relationships parser
        embedding = embeddings,
        collection_name = "olist_joins_v2",
        persist_directory = joins_persist_dir,
    )
    
    return schema_retriever, schema_child_vs, joins_vs

def split_tables_and_relationships(raw_text: str) -> Tuple[List[str], str]:
    """
    Returns:
      table_blocks: list of per-table blocks (each starts with 'Table:' and ends BEFORE 'Table Relationships:')
      relationships_text: text of 'Table Relationships:' section (or '' if absent)
    """
    # 1) Separate relationships section so it doesn’t mix into last table
    rel_match = re.search(r'^\s*Table Relationships\s*:\s*', raw_text, flags = re.IGNORECASE | re.MULTILINE)
    if rel_match:
        body_text = raw_text[:rel_match.start()]
        relationships_text = raw_text[rel_match.start():].strip()
    else:
        body_text = raw_text
        relationships_text = ""

    # 2) Ignore any intro; start splitting from first "Table:"
    first_table = re.search(r'^\s*Table:\s+', body_text, flags = re.MULTILINE)
    if first_table:
        tables_text = body_text[first_table.start():]
    else:
        tables_text = body_text

    # 3) Split into table blocks
    table_blocks = []
    if tables_text:
        blocks = re.split(r'(?=^\s*Table:\s+)', tables_text, flags = re.MULTILINE)
        for b in blocks:
            b = b.strip()
            if b:
                table_blocks.append(b)

    return table_blocks, relationships_text

def make_parent_documents(table_blocks: List[str]) -> List[Document]:
    """Create one parent Document per table, no 'source' metadata."""
    parents: List[Document] = []
    for block in table_blocks:
        m = re.match(r'^\s*Table:\s*([^\n\r]+)', block)
        if not m:
            continue
        table_name = m.group(1).strip()
        parents.append(
            Document(
                page_content=block,
                metadata={
                    "kind": "schema_parent",
                    "table": table_name
                }
            )
        )
    return parents

# ---------------- ONE JOIN PER CHUNK FROM relationships_text ----------------
def parse_relationships_to_join_docs(relationships_text: str) -> List[Document]:
    """
    Turn the 'Table Relationships:' section into a list of tiny Documents,
    one per relationship edge, each carrying precise join metadata.
    """
    if not relationships_text:
        return []

    join_docs: List[Document] = []
    for line in relationships_text.splitlines():
        line_stripped = line.strip()
        # skip header/empty lines
        if not line_stripped or not re.match(r'^\s*From Table\s*:', line_stripped, flags=re.IGNORECASE):
            continue

        # Split key/value pairs separated by "|"
        parts = [p.strip() for p in line_stripped.split("|")]
        fields: Dict[str, str] = {}
        for p in parts:
            m = re.match(r'\s*([A-Za-z ]+)\s*:\s*(.*)\s*$', p)
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                val = m.group(2).strip()
                fields[key] = val

        ft = fields.get("from_table", "")
        fc = fields.get("from_column", "")
        tt = fields.get("to_table", "")
        tc = fields.get("to_column", "")
        note = fields.get("relationship_note", "")

        join_docs.append(
            Document(
                page_content=line_stripped,
                metadata={
                    "kind": "join",
                    "from_table": ft,
                    "from_col": fc,
                    "to_table": tt,
                    "to_col": tc,
                    "note": note
                }
            )
        )
    return join_docs