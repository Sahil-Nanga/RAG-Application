from src.pipeline import RAGPipeline
import psutil
import time
import threading
import json
def measure_cpu_usage(cpu_usage_samples, stop_event):
    """ Continuously measures CPU usage until stopped. """
    while not stop_event.is_set():
        usage = psutil.cpu_percent(interval=0.25)  # Measure CPU usage every 0.5 sec
        cpu_usage_samples.append(usage)

def main(model,query):
    # Initialize the pipeline
    start_init_time = time.time()
    pipeline = RAGPipeline(
        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path="embeddings/faiss_index.bin",
        dim=384,
        llm_model_name=model,
        doc_store_path="data\\externalData.txt"
    )
    end_init_time = time.time()
    init_time = end_init_time -start_init_time
   

    # Start CPU monitoring in a separate thread
    cpu_usage_samples = []
    stop_event = threading.Event()
    cpu_monitor_thread = threading.Thread(target=measure_cpu_usage, args=(cpu_usage_samples, stop_event))
    cpu_monitor_thread.start()

    # Start the timer
    start_time = time.time()

    # Process the query through the pipeline
    response,que = pipeline.process(query)
    print(que)
    # Stop the timer
    response_time = time.time() - start_time

    # Stop CPU monitoring
    stop_event.set()
    cpu_monitor_thread.join()

    # Compute average CPU usage
    avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples) if cpu_usage_samples else 0
    return {"query":query,
            "model":model,
            "response" :response.response,
            "initialization_time":init_time,
            "total_execution_time":response_time + init_time,
            "average_cpu_usage_during_processing":avg_cpu_usage,
            "peak_cpu_usage":max(cpu_usage_samples),}

if __name__ == "__main__":

    query = input("Enter query:")
    resp = main("deepseek-r1:1.5b",query)
    print(resp["response"])
        

        

