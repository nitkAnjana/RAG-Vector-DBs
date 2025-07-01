import os, time, shutil
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient

# Disable Chroma telemetry
os.environ["ALLOW_CHROMA_TELEMETRY"] = "FALSE"

# Clean existing DB
def force_delete(folder):
    for _ in range(5):
        try:
            shutil.rmtree(folder)
            break
        except PermissionError:
            time.sleep(1)
if os.path.exists("chroma_db"):
    force_delete("chroma_db")

# Use mock LLM
Settings.llm = None

# Step 1: Download articles
print("Downloading articles...")
dataset = load_dataset("cc_news", split="train", streaming=True)
docs = [item["text"] for _, item in zip(range(100), dataset)]

# Step 2: Save to file
os.makedirs("data", exist_ok=True)
with open("data/cc_news_sample.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(docs))

# Step 3: Embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Step 4: Read and parse docs
documents = SimpleDirectoryReader("data").load_data()
nodes = SimpleNodeParser().get_nodes_from_documents(documents)

# Step 5: Vector DB setup
chroma_client = PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="news_collection")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Step 6: Index and query
print("Indexing and querying...")
start = time.time()
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
query_engine = index.as_query_engine(retriever_mode="simple")
response = query_engine.query("What is the news about climate change?")
print("Response:", response)
print(f"Done in {time.time() - start:.2f} seconds")

#Done in 19.43 seconds