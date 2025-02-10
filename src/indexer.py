#initially adds the document store to the faiss index
import os
import faiss
import numpy as np
from embedding import Embedder

class Indexer:
    def __init__(self,file_name):
        self.embedder = Embedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.data = self.get_data(file_name)
    def get_data(self,file_name):
        with open(f"data\\{file_name}","r",encoding="utf-8") as file:
            data = [line.strip().lower() for line in file]
        return data
    
    def add_documents(self):
       
        embeddings = self.embedder.encode(self.data)
        self.embedder.store(embeddings)
        print(f"Added {len(self.data)} documents to the vector database.")

if __name__ == "__main__":

    indexer = Indexer("externalData.txt")
    indexer.add_documents()
