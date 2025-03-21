import psutil
import time

def track_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # Convert to MB
    return memory_usage

def track_performance(func, *args):
    start_time = time.time()
    start_memory = track_memory_usage()
    
    result = func(*args)
    
    end_time = time.time()
    end_memory = track_memory_usage()
    
    time_taken = end_time - start_time
    memory_used = end_memory - start_memory

    print(f"Performance Metrics for {func.__name__}:")
    print(f"Time Taken: {time_taken:.2f} sec")
    print(f"Memory Used: {memory_used:.2f} MB")
    
    return time_taken, memory_used
