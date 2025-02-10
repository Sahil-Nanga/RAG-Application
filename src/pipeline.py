from .embedding import Embedder
from .retriever import Retriever
from .generator import Generator
from .clean import Cleaner


class RAGPipeline:
    def __init__(self,embedder_model,index_path,dim,llm_model_name,doc_store_path):
        self.embedder = Embedder(embedder_model)
        self.retriever = Retriever(index_path,dim,doc_store_path)
        self.generator = Generator(llm_model_name)
        self.cleaner = Cleaner()
    
    def process(self, query):
        clean_query = self.cleaner.clean_text(query) #cleans user query
        query_embedding = self.embedder.encode([clean_query]) #embeds query
        retrieved_indices = self.retriever.search(query_embedding,top_k=2) #collects similar documents to the query from the faiss
        context = self.retrieve_documents(retrieved_indices) #Gathers the documents from the retrieved indexes
        response = self.generator.generate(context)
        return response
    def retrieve_documents(self, indices):
        docs = self.retriever.get_documents(indices)
        return "\n".join(docs)