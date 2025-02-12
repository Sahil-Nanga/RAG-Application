from src.pipeline import RAGPipeline
import psutil
import time
import threading

def measure_cpu_usage(cpu_usage_samples, stop_event):
    """ Continuously measures CPU usage until stopped. """
    while not stop_event.is_set():
        usage = psutil.cpu_percent(interval=0.25)  # Measure CPU usage every 0.5 sec
        cpu_usage_samples.append(usage)

def main():
    # Initialize the pipeline
    start_init_time = time.time()
    pipeline = RAGPipeline(
        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
        index_path="embeddings/faiss_index.bin",
        dim=384,
        llm_model_name="llama3.2:1b",
        doc_store_path="data\\externalData.txt"
    )
    end_init_time = time.time()
    init_time = end_init_time -start_init_time
    query = input("Enter your query: ")

    # Start CPU monitoring in a separate thread
    cpu_usage_samples = []
    stop_event = threading.Event()
    cpu_monitor_thread = threading.Thread(target=measure_cpu_usage, args=(cpu_usage_samples, stop_event))
    cpu_monitor_thread.start()

    # Start the timer
    start_time = time.time()

    # Process the query through the pipeline
    response = pipeline.process(query)

    # Stop the timer
    response_time = time.time() - start_time

    # Stop CPU monitoring
    stop_event.set()
    cpu_monitor_thread.join()

    # Compute average CPU usage
    avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples) if cpu_usage_samples else 0

    print("\nResponse:")
    print(response.response)

    print("\nPerformance Metrics:")
    print(f"Initialization Time : {init_time:.2f} seconds")
    print(f"Total Execution Time: {(response_time + init_time):.2f} seconds")
    print(f"Average CPU Usage During Processing: {avg_cpu_usage:.2f}%")
    print(f"Peak Cpu usage: {max(cpu_usage_samples)}")

if __name__ == "__main__":
    main()
