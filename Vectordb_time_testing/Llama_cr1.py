import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from datasets import load_dataset
from llama_index.core import Settings
from chromadb import PersistentClient
import os
os.environ["CHROMA_TELEMETRY"] = "false"


Settings.llm = None  # Using MockLLM

# ✅ Step 1: Setup local Chroma client and collection
# Setup Chroma vector store properly
chroma_client = PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="news_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ✅ Step 2: Timer start
start_time = time.time()

# ✅ Step 3: Load dataset streamingly
dataset = load_dataset("cc_news", split="train", streaming=True)
docs = []
for i, item in enumerate(dataset):
    docs.append(item["text"])
    if i == 1000:
        break

# ✅ Step 4: Save docs locally
with open("data/cc_news_sample.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(docs))

# ✅ Step 5: Embed model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ✅ Step 6: Load and parse documents
documents = SimpleDirectoryReader("data").load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# ✅ Step 7: Build storage context and index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

# ✅ Step 8: Query
#query_engine = index.as_query_engine(llm=None, retriever_mode="simple")
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

# Query engine with safe retriever that avoids `filters={}`
query_engine = index.as_query_engine(
    llm=None,
    retriever_mode="simple",
    vector_store_query_mode="default",  # avoids 'hybrid' mode bugs
    similarity_top_k=5,
    filters=None  # ✅ explicitly set to None to avoid error
)
response = query_engine.query("What is the news about climate change?")
print(response)

# ✅ Step 9: Timer end
end_time = time.time()
print(f"\n Time taken: {end_time - start_time:.2f} seconds")
