class EmbeddingModel:
    def embed(self, text):
        # Implement embedding logic
        pass

class Embedder:
    def __init__(self, model: EmbeddingModel):
        self.model = model

    def get_embedding(self, text):
        return self.model.embed(text)

class Preprocessor:
    def preprocess(self, text):
        # Implement preprocessing logic
        pass

class EmbeddingCache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        # Retrieve embedding from cache
        pass

    def set(self, key, value):
        # Set embedding in cache
        pass
