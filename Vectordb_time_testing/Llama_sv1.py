import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import StorageContext
from llama_index.core.vector_stores.simple import SimpleVectorStore  # ✅ correct import
from datasets import load_dataset
import os
from llama_index.core import Settings

# Disable default LLM to avoid OpenAI key error
Settings.llm = None

# Timer start
#start_time = time.time()

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Load a streaming dataset (no full download)
dataset = load_dataset("cc_news", split="train", streaming=True)
docs = []
for i, item in enumerate(dataset):
    docs.append(item["text"])
    if i == 1000:  # Increase this to 10k+ if you want more test load
        break

# Save docs to file
with open("data/cc_news_sample.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(docs))

# Use a free Hugging Face embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load and parse documents
documents = SimpleDirectoryReader("data").load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# Create vector store and storage context
vector_store = SimpleVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index with storage and embedding model
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

# Run a sample query
query_engine = index.as_query_engine(llm=None)

start_time = time.time()
response = query_engine.query("What is the news about climate change?")
end_time = time.time()

print(response)

# Timer end
#end_time = time.time()
print(f"\n Time taken: {end_time - start_time:.2f} seconds")

#Time taken: 140.76 seconds
# Time taken: 126.69 seconds

#after:  Time taken: 0.05 seconds