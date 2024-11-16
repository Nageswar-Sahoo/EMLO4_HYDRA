import time
import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import psutil
import concurrent.futures


try:
    import gpustat
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Constants
SERVER_URL = "http://localhost:8000"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TOKENIZE_TEXT = "The quick brown fox jumps over the lazy dog."

# Load Llama-based model with TorchAO quantization
def load_llama_model_with_quantization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Apply TorchAO dynamic quantization to the model for improved inference
    if device == "cpu":  # Quantization is typically CPU-based
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Applied TorchAO quantization.")

    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer, device

# Calculate theoretical max throughput
def calculate_theoretical_max_throughput(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Theoretical max throughput based on model parameters: {num_params / 1e6:.2f}M tokens/sec")
    return num_params

# Benchmark throughput for single token generation
def get_token_generation_throughput(model, tokenizer, device, num_iterations=10):
    input_ids = tokenizer(TOKENIZE_TEXT, return_tensors="pt").input_ids.to(device)
    throughputs = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            outputs = model.generate(input_ids, max_new_tokens=1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()

            tokens_per_sec = 1 / (end - start)  # Single token per generation
            throughputs.append(tokens_per_sec)

    return np.mean(throughputs)

# Prepare token payload
def prepare_test_payload(tokenizer):
    return tokenizer(TOKENIZE_TEXT, return_tensors="pt").input_ids

# Send request and measure response time
def send_request(payload):
    start_time = time.time()
    response = requests.post(SERVER_URL, json={"input_ids": payload.tolist()})
    end_time = time.time()
    return end_time - start_time, response.status_code

# Get CPU and GPU metrics
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

# API benchmark with concurrency
def benchmark_api(num_requests=100, concurrency_level=10):
    payload = prepare_test_payload(AutoTokenizer.from_pretrained(MODEL_NAME))
    system_metrics = []

    start_benchmark_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(num_requests)]
        response_times = []
        status_codes = []

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

    avg_cpu = np.mean([m["cpu_usage"] for m in system_metrics])
    avg_gpu = np.mean([m["gpu_usage"] for m in system_metrics]) if GPU_AVAILABLE else -1

    return {
        "total_requests": num_requests,
        "concurrency_level": concurrency_level,
        "total_time": total_benchmark_time,
        "avg_response_time": np.mean(response_times) * 1000,  # ms
        "success_rate": (status_codes.count(200) / num_requests) * 100,
        "requests_per_second": num_requests / total_benchmark_time,
        "avg_cpu_usage": avg_cpu,
        "avg_gpu_usage": avg_gpu
    }

# Run benchmarks and create plots
def run_benchmarks():
    model, tokenizer, device = load_llama_model_with_quantization()
    calculate_theoretical_max_throughput(model)

    # Measure baseline throughput
    print("Running token generation throughput test...")
    single_token_throughput = get_token_generation_throughput(model, tokenizer, device)
    print(f"Single token generation throughput: {single_token_throughput:.2f} tokens/sec")

    # API benchmarks
    concurrency_levels = [1, 8, 32, 64]
    api_throughput = []
    cpu_usage = []
    gpu_usage = []

    print("Running API benchmarks...")
    for concurrency in concurrency_levels:
        metrics = benchmark_api(num_requests=128, concurrency_level=concurrency)
        api_throughput.append(metrics["requests_per_second"])
        cpu_usage.append(metrics["avg_cpu_usage"])
        gpu_usage.append(metrics["avg_gpu_usage"])
        print(f"Concurrency {concurrency}: {metrics['requests_per_second']:.2f} reqs/sec, "
              f"CPU: {metrics['avg_cpu_usage']:.1f}%, GPU: {metrics['avg_gpu_usage']:.1f}%")

    # Create plots
    plt.figure(figsize=(15, 5))

    # Throughput plot
    plt.subplot(1, 3, 1)
    plt.plot(concurrency_levels, api_throughput, 'r-', label='API Server')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Throughput (requests/second)')
    plt.title('API Throughput')
    plt.legend()
    plt.grid(True)

    # CPU Usage plot
    plt.subplot(1, 3, 2)
    plt.plot(concurrency_levels, cpu_usage, 'g-')
    plt.xlabel('Concurrency Level')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage')
    plt.grid(True)

    # GPU Usage plot
    plt.subplot(1, 3, 3)
    plt.plot(concurrency_levels, gpu_usage, 'm-')
    plt.xlabel('Concurrency Level')
    plt.ylabel('GPU Usage (%)')
    plt.title('GPU Usage')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('llm_benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    print("Running benchmarks for Llama-based model with TorchAO quantization...")
    run_benchmarks()
