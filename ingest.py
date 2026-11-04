
from pathlib import Path

import chromadb
import ollama

# STEP 1: Configure paths and models
DATA_DIR = Path("data")
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "embeddinggemma"

# STEP 2: Create a persistent vector database
db = chromadb.PersistentClient(path=DB_DIR)

collection = db.get_or_create_collection(
    name="clouddesk_docs"
)

# STEP 3: Split text into overlapping chunks
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# STEP 4: Read all company documents
documents = []
ids = []
metadata = []

for file_path in sorted(DATA_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")

    chunks = chunk_text(text)

    for index, chunk in enumerate(chunks):
        documents.append(chunk)

        ids.append(f"{file_path.stem}_{index}")

        metadata.append({
            "source": file_path.name
        })

# STEP 5: Generate embeddings
if not documents:
    raise RuntimeError("No documents found in data/")

print(f"Processing {len(documents)} document chunks...")

embeddings = []

for index, document in enumerate(documents):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=document
    )

    embeddings.append(response["embeddings"][0])

    print(f"Embedded {index + 1}/{len(documents)}")

# STEP 6: Store vectors and document text
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadata
)

print("Documents successfully stored!")
print("Total indexed chunks:", collection.count())
