import json
with open("responses\deepseek7b.json","r") as fp:
    d = json.load(fp)
total_execution_time=0
total_average_cpu_usage_during_processing=0
total_peak_cpu_usage=0
total_init_time = 0
for query in d:
    total_execution_time += query["total_execution_time"]
    total_average_cpu_usage_during_processing +=query["average_cpu_usage_during_processing"]
    total_peak_cpu_usage +=query["peak_cpu_usage"]
    total_init_time +=query["initialization_time"]
print(f"Average total execution time:{total_execution_time/8}\nAverage cpu usage:{total_average_cpu_usage_during_processing/8}\nAverage Peak CPU usage:{total_peak_cpu_usage/8}\n average init time: {total_init_time/8}")
   