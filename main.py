import os
import time
import chromadb

# Silence HF telemetry warning (cosmetic)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.voyageai import VoyageEmbedding

# ---------- CONFIG ----------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
RERANK_TOP = 3

EMBED_MODEL = "voyage-3.5-lite"
LLM_MODEL = "gemini-3.8-flash"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "docs"
DATA_DIR = "./data"

# ---------- 1. KEYS ----------
voyage_key = os.environ.get("VOYAGE_API_KEY")
google_key = os.environ.get("GOOGLE_API_KEY")

if not voyage_key:
    raise SystemExit(
        "VOYAGE_API_KEY not set. Get one at https://www.voyageai.com "
        "then run: export VOYAGE_API_KEY='your-key-here'"
    )
if not google_key:
    raise SystemExit(
        "GOOGLE_API_KEY not set. Get one at https://aistudio.google.com/apikey "
        "then run: export GOOGLE_API_KEY='your-key-here'"
    )

# ---------- 2. MODELS ----------
class ThrottledVoyage(VoyageEmbedding):
    """Enforce 3 RPM on Voyage's unverified tier."""
    def _get_text_embeddings(self, texts):
        time.sleep(20)  # 3 requests/min = 20s between calls
        return super()._get_text_embeddings(texts)

Settings.embed_model = ThrottledVoyage(
    model_name=EMBED_MODEL,
    api_key=voyage_key,
    embed_batch_size=8,
)

Settings.llm = GoogleGenAI(
    model=LLM_MODEL,
    api_key=google_key,
    temperature=0.2,
)

# ---------- 3. CHUNKING ----------
Settings.node_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# ---------- 4. VECTOR STORE ----------
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)

# ---------- 5. LOAD + INDEX ----------
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if collection.count() == 0:
    print("Loading documents...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    if not documents:
        raise SystemExit(f"No documents found in {DATA_DIR}. Add some files and rerun.")

    print("Building index (this runs once, ~8 min on Voyage free tier)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("Index ready!\n")
else:
    print(f"Reusing existing index ({collection.count()} chunks).\n")
    index = VectorStoreIndex.from_vector_store(vector_store)

# ---------- 6. QUERY ----------
query_engine = index.as_query_engine(similarity_top_k=TOP_K)

print("Ask questions (type 'quit' to exit):\n")

while True:
    try:
        question = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if question.lower() in ("quit", "exit", ""):
        break

    # Retry the query up to 3 times on transient server errors
    response = None
    for attempt in range(1, 4):
        try:
            response = query_engine.query(question)
            break
        except Exception as e:
            msg = str(e)
            transient = any(
                code in msg for code in ("503", "UNAVAILABLE", "500", "INTERNAL")
            )
            if not transient or attempt == 3:
                print(f"\n[!] Query failed: {msg.splitlines()[0]}")
                print("    Skipping this question.\n")
                break
            print(f"[retry {attempt}/3] Server busy, waiting {2 * attempt}s...")
            time.sleep(2 * attempt)

    if response is None:
        continue

    print(f"\nAnswer: {response}\n")
    print("Sources:")
    for i, node in enumerate(response.source_nodes[:RERANK_TOP], 1):
        score = node.score
        preview = node.text[:120].replace("\n", " ")
        print(f"  [{i}] score={score:.3f} | {preview}...")
    print("-" * 60)