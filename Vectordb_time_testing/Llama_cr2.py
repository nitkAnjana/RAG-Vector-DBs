import os
import time

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

from datasets import load_dataset
from chromadb import PersistentClient
import shutil
os.environ["ALLOW_CHROMA_TELEMETRY"] = "FALSE"

def force_delete(folder, retries=5):
    for i in range(retries):
        try:
            shutil.rmtree(folder)
            break
        except PermissionError:
            print(f"Retrying delete {folder}... ({i+1}/{retries})")
            time.sleep(1)
            
if os.path.exists("./chroma_db"):
    force_delete("./chroma_db")

# Disable actual LLM usage (mocked)
Settings.llm = None

#Stream and save news articles from cc_news
print("Streaming 100 articles from cc_news...")
dataset = load_dataset("cc_news", split="train", streaming=True)
docs = []
for i, item in enumerate(dataset):
    docs.append(item["text"])
    if i >= 100:
        break

# Save to local file
os.makedirs("data", exist_ok=True)
with open("data/cc_news_sample.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(docs))

# Setup embedding model

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


#Load and parse documents

documents = SimpleDirectoryReader("data").load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

#Setup ChromaDB vector store

os.makedirs("./chroma_db", exist_ok=True)
chroma_client = PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="news_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Build index and query

print("Indexing...")

# start_time = time.time()
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

print("Querying...")
query_engine = index.as_query_engine(retriever_mode="simple")
start_time = time.time()

response = query_engine.query("What is the news about climate change?")
end_time = time.time()

print("Response:", response)

print(f"\nTime taken: {end_time - start_time:.2f} seconds")

#Time taken1: 14.48 seconds
#Time taken2: 12.10 seconds

#response Time taken: 0.50 seconds

#discarded coz 