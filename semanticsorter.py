import json
from sklearn.metrics.pairwise import cosine_similarity


class semanticsorter():
    def __init__(self, embedding):
        self.embedding = embedding

    def sort(self, base, documents):
        embedded_base = self.embedding.embed_documents([base])
        embedded_documents = self.embedding.embed_documents(documents)
        similarities = cosine_similarity(embedded_base, embedded_documents)
        return similarities[0].tolist()

