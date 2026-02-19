import json
import numpy as np
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv
import os


load_dotenv()

# =============================
# CONFIG
# =============================

API_KEY = os.getenv("GOOGLE_API_KEY")
INPUT_FILE = "chunks.json"
EMBEDDING_OUTPUT = "embeddings.npy"
METADATA_OUTPUT = "metadata.json"
MODEL_NAME = "gemini-embedding-001"

# =============================
# Setup Client
# =============================

client = genai.Client(api_key=API_KEY)

# =============================
# Load Chunks
# =============================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# =============================
# Generate Embeddings
# =============================

embeddings = []
metadata = []

for chunk in tqdm(chunks):
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=chunk["text"],
        config={
            "task_type": "RETRIEVAL_DOCUMENT"
        }
    )

    vector = response.embeddings[0].values
    embeddings.append(vector)

    metadata.append({
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "source_file": chunk["metadata"]["source_file"],
        "page_start": chunk["metadata"]["page_start"],
        "page_end": chunk["metadata"]["page_end"],
        "language": chunk["metadata"]["language"]
    })

# =============================
# Save Files
# =============================

embeddings_array = np.array(embeddings)

np.save(EMBEDDING_OUTPUT, embeddings_array)

with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("===================================")
print("Embeddings shape:", embeddings_array.shape)
print(f"Saved to {EMBEDDING_OUTPUT}")
print(f"Saved to {METADATA_OUTPUT}")
print("===================================")
