import os
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
#from llama_index_vector_stores_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from datasets import load_dataset, DownloadMode

#Disable LLM usage
Settings.llm = None

#Load news data from HuggingFace
print("Loading 100 news articles from cc_news...")
try:
    dataset = load_dataset("cc_news", split="train[:100]", download_mode=DownloadMode.FORCE_REDOWNLOAD)
    docs = [item["text"] for item in dataset]
except Exception as e:
    print("Dataset load failed:", e)
    exit()

#Save docs to a text file
os.makedirs("data", exist_ok=True)
with open("data/news.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(docs))

#Load and parse documents
documents = SimpleDirectoryReader("data").load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

#Embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#Setup Qdrant vector store
qdrant_client = QdrantClient(path="qdrant_data")
collection_name = "news_collection"

if not qdrant_client.collection_exists(collection_name):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Build index and query
print("Indexing...")
# start = time.time()
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

print("Querying...")
query_engine = index.as_query_engine(retriever_mode="simple")
start = time.time()
response = query_engine.query("What is the news about climate change?")
print(f"\n Done in {time.time() - start:.2f} seconds")

print("Response:\n", response)


#Done in 0.07 seconds.