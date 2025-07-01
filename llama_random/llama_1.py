from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = None
Settings.embed_model = embed_model

docs = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
print(response)
