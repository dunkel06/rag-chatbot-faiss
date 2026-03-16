from embeddings import EmbeddingModel
from vector_store import VectorStore
import numpy as np

class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.store = VectorStore(384)

    def build_index(self, documents):
        embeddings = self.embedder.encode(documents)
        self.store.add(embeddings, documents)

    def retrieve(self, query):
        query_embedding = self.embedder.encode([query])
        return self.store.search(np.array(query_embedding))

    def generate_answer(self, query):
        context = self.retrieve(query)
        return f"Answer generated using context: {context}"
