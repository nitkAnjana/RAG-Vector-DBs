import os
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from datasets import load_dataset

# === Disable LLM usage ===
Settings.llm = None

# === Step 1: Stream news data from HuggingFace ===
print("Loading cc_news dataset...")
dataset = load_dataset("cc_news", split="train", streaming=True)
docs = []
for i, item in enumerate(dataset):
    docs.append(item["text"])
    if i >= 100:
        break

# Save docs to file
os.makedirs("data", exist_ok=True)
with open("data/news.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(docs))

# === Step 2: Load and parse documents ===
documents = SimpleDirectoryReader("data").load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# === Step 3: Use HuggingFace embedding ===
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# === Step 4: Set up Qdrant vector store (local memory mode) ===
qdrant_client = QdrantClient(path="qdrant_data")  # Uses local folder instead of Docker
collection_name = "news_collection"

# Create collection if not already present
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # 384 = embedding dim
    )

vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# === Step 5: Index + Query ===
print("Indexing...")
start = time.time()
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

print("Querying...")
query_engine = index.as_query_engine(retriever_mode="simple")
response = query_engine.query("What is the news about climate change?")
print("Response:", response)

print(f"\n Done in {time.time() - start:.2f} seconds")
