import time
import faiss  # <-- Import FAISS core library
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from datasets import load_dataset
from llama_index.core import Settings

# Disable default LLM to avoid OpenAI key error
Settings.llm = None

# Start timer
#start_time = time.time()

# Load streaming dataset
dataset = load_dataset("cc_news", split="train", streaming=True)
docs = []
for i, item in enumerate(dataset):
    docs.append(item["text"])
    if i == 1000:
        break

# Save docs to 'data' folder
with open("data/cc_news_sample.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(docs))

# Set up embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Read and parse documents
documents = SimpleDirectoryReader("data").load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# ✅ Initialize empty FAISS index with dimension = 384 (bge-small-en-v1.5 output dim)
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build the index
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

# Query
query_engine = index.as_query_engine(llm=None)

start_time = time.time()
response = query_engine.query("What is the news about climate change?")
end_time = time.time()
print(response)

# End timer
#end_time = time.time()
print(f"\n Time taken: {end_time - start_time:.2f} seconds")


#Time taken: 134.86 seconds
#Time taken: 121.53 seconds

#after: 