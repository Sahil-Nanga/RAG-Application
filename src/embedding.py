from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts):#returns encoded data in a numpy array with float32 datatype
        return self.model.encode(texts, convert_to_numpy=True)
    
    def store(self,data): #used to store the embedded document store in the Faiss index
        dimension = data.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(data)
        faiss.write_index(index, "embeddings\\faiss_index.bin")
        print("FAISS index saved.")

