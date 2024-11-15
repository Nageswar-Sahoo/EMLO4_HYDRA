import os
import base64
import time
import numpy as np
import requests
import torch
import timm
import matplotlib.pyplot as plt
from PIL import Image
import psutil
import concurrent.futures
import io

try:
    import gpustat
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

SERVER_URL = "http://localhost:8000/predict"
DATASET_PATH = "data"  # Path to the dataset

def load_and_encode_image(image_path):
    """Load and encode an image in base64 format."""
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_img

def prepare_dataset_payloads():
    """Prepare a list of encoded images from Train, Test, and Validation directories."""
    image_paths = []
    for subfolder in ['test']:
        folder_path = os.path.join(DATASET_PATH, subfolder)
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, file))
    
    encoded_images = [load_and_encode_image(img_path) for img_path in image_paths]
    return encoded_images

def get_baseline_throughput(batch_size, num_iterations=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model('mambaout_base.in1k', pretrained=True).to(device)
    model.eval()
    
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    throughputs = []

    with torch.no_grad():
        model(x)
    
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        throughputs.append(batch_size / (t1 - t0))
    
    return np.mean(throughputs)

def send_request(payload):
    start_time = time.time()
    response = requests.post(SERVER_URL, json={"image": payload})
    end_time = time.time()
    return end_time - start_time, response.status_code

def get_system_metrics():
    metrics = {"cpu_usage": psutil.cpu_percent(0.1)}
    if GPU_AVAILABLE:
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            metrics["gpu_usage"] = sum([gpu.utilization for gpu in gpu_stats.gpus])
        except Exception:
            metrics["gpu_usage"] = -1
    else:
        metrics["gpu_usage"] = -1
    return metrics

import time
import concurrent.futures
import numpy as np
import psutil

# Assuming send_request, get_system_metrics, and GPU_AVAILABLE are already defined elsewhere

def benchmark_api(encoded_images, num_requests=100, concurrency_level=10):
    """Benchmark the API server with a restricted number of requests."""
    system_metrics = []
    response_times = []
    status_codes = []
    
    # Limiting the number of requests to num_requests
    #limited_encoded_images = encoded_images[:num_requests]

    start_benchmark_time = time.time()
    
    # Create a ThreadPoolExecutor to handle the concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(send_request, img) for img in encoded_images]
        
        # Collect system metrics during the benchmark
        while any(not f.done() for f in futures):
            system_metrics.append(get_system_metrics())
            time.sleep(0.1)
        
        for future in futures:
            response_time, status_code = future.result()
            response_times.append(response_time)
            status_codes.append(status_code)
    
    end_benchmark_time = time.time()
    total_benchmark_time = end_benchmark_time - start_benchmark_time
    
    # Calculate system metrics like CPU and GPU usage
    avg_cpu = np.mean([m["cpu_usage"] for m in system_metrics])
    avg_gpu = np.mean([m["gpu_usage"] for m in system_metrics]) if GPU_AVAILABLE else -1
    
    # Calculate success rate and average response time
    success_rate = (status_codes.count(200) / num_requests) * 100 if num_requests > 0 else 0
    avg_response_time = np.mean(response_times) * 1000  # Convert to milliseconds
    
    # Calculate requests per second
    requests_per_second = num_requests / total_benchmark_time if total_benchmark_time > 0 else 0
    
    return {
        "total_requests": num_requests,
        "concurrency_level": concurrency_level,
        "total_time": total_benchmark_time,
        "avg_response_time": avg_response_time,  # in ms
        "success_rate": success_rate,
        "requests_per_second": requests_per_second,
        "avg_cpu_usage": avg_cpu,
        "avg_gpu_usage": avg_gpu,
    }
def run_benchmarks():
    # Prepare payloads from the Dog Breed dataset
    encoded_images = prepare_dataset_payloads()
    
    # Baseline throughput tests with batch sizes
    batch_sizes = [1, 8, 32, 64]
    baseline_throughput = []
    print("\nRunning baseline throughput tests...")
    for batch_size in batch_sizes:
        reqs_per_sec = get_baseline_throughput(batch_size)
        baseline_throughput.append(reqs_per_sec)
        print(f"Batch size {batch_size}: {reqs_per_sec:.2f} reqs/sec")
    
    # API benchmark with concurrency levels
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128]
    api_throughput = []
    cpu_usage = []
    gpu_usage = []
    print("\nRunning API benchmarks...for Total Requests ")
    results_str = []
    headers = [
    "Concurrency Level",
    "Total Requests",
    "Total Time (s)",
    "Avg Response Time (ms)",
    "Success Rate (%)",
    "Requests per Second",
    "Avg CPU Usage (%)",
    "Avg GPU Usage (%)",
    ]
    header_row = " | ".join(f"{header:<22}" for header in headers)
    print(header_row)
    print("-" * len(header_row))
    for concurrency in concurrency_levels:
        metrics = benchmark_api(encoded_images, concurrency_level=concurrency)
        api_throughput.append(metrics["requests_per_second"])
        cpu_usage.append(metrics["avg_cpu_usage"])
        gpu_usage.append(metrics["avg_gpu_usage"])
        formatted_result = (f"Concurrency {concurrency}: {metrics['requests_per_second']:.2f} reqs/sec, "
                            f"CPU: {metrics['avg_cpu_usage']:.1f}%, GPU: {metrics['avg_gpu_usage']:.1f}%")
        results_str.append(formatted_result)
        row = (
        f"{metrics['concurrency_level']:<22} | "
        f"{metrics['total_requests']:<22} | "
        f"{metrics['total_time']:<22.2f} | "
        f"{metrics['avg_response_time']:<22.2f} | "
        f"{metrics['success_rate']:<22.2f} | "
        f"{metrics['requests_per_second']:<22.2f} | "
        f"{metrics['avg_cpu_usage']:<22.1f} | "
        f"{metrics['avg_gpu_usage']:<22.1f}"
        )
        print(row)
        #print(formatted_result)

    # Plotting
    plt.figure(figsize=(15, 5))

    # Throughput comparison
    plt.subplot(1, 3, 1)
    plt.plot(batch_sizes, baseline_throughput, 'b-', label='Baseline Model')
    plt.plot(concurrency_levels, api_throughput, 'r-', label='API Server')
    plt.xlabel('Batch Size / Concurrency Level')
    plt.ylabel('Throughput (requests/second)')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.grid(True)

    # CPU Usage
    plt.subplot(1, 3, 2)
    plt.plot(concurrency_levels, cpu_usage, 'g-')
    plt.xlabel('Concurrency Level')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage')
    plt.grid(True)

    # GPU Usage
    plt.subplot(1, 3, 3)
    plt.plot(concurrency_levels, gpu_usage, 'm-')
    plt.xlabel('Concurrency Level')
    plt.ylabel('GPU Usage (%)')
    plt.title('GPU Usage')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

    return results_str

if __name__ == "__main__":
    results = run_benchmarks()

