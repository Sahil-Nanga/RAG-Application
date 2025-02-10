from src.pipeline import RAGPipeline
def main():
    # Initialize the pipeline with the required models, paths, and dimension.
    pipeline = RAGPipeline(
        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path="embeddings/faiss_index.bin",
        dim=384,
        llm_model_name="llama3.2:1b",
        doc_store_path="data\\externalData.txt" 
    )
    

    query = input("Enter your query: ")
    
    # Process the query through the pipeline.
    response = pipeline.process(query)

    print("Response:")
    print(response.response)

if __name__ == "__main__":
    main()