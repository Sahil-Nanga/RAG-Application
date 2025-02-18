import faiss
import os
#index_path='faiss_index.bin'
#dim=384
class Retriever:
    def __init__(self, index_path,dim,doc_store_path):
        self.index_path = index_path
        self.doc_store_path =doc_store_path
        self.documents=self.load_document_store()
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)
    
    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(query_embedding, top_k)
        return indices
    def get_documents(self, indices):
        """
        Fetches documents corresponding to FAISS indices.
        """
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]
    def load_document_store(self):
        if os.path.exists(self.doc_store_path):
            with open(self.doc_store_path, "r", encoding="utf8") as f:
                documents = [line.strip() for line in f if line.strip()]
            return documents
        else:
            self.documents = []
        