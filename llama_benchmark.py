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

print("*********************"+ str(GPU_AVAILABLE))
# Constants
SERVER_URL = "http://localhost:8000/v1"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TOKENIZE_TEXT = "Capital of India"

# Load Llama-based model with TorchAO quantization
def load_llama_model_with_quantization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Apply TorchAO dynamic quantization to the model for improved inference
    if device == "cpu":  # Quantization is typically CPU-based
        print("*********quantization******")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Applied TorchAO quantization.")
        print("*********quantization******")


    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer, device

# Calculate theoretical max throughput
def calculate_theoretical_max_throughput(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Theoretical max throughput based on model parameters: {num_params / 1e6:.2f}M tokens/sec")
    return num_params

# Benchmark throughput for single token generation
def get_token_generation_throughput11(model, tokenizer, device, num_iterations=10):
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

import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_token_generation_throughput_batch(batch_size, num_iterations=10):
    """
    Calculate the mean throughput for token generation with a given batch size.

    Args:
        batch_size (int): The size of the input batch.
        num_iterations (int): The number of iterations to measure throughput.

    Returns:
        float: The mean throughput in tokens per second.
    """

    # Initialize model and tokenizer
    # Initialize model and tokenizer
    model, tokenizer, device = load_llama_model_with_quantization()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token if available

    # Prepare input data
    input_text = TOKENIZE_TEXT * batch_size  # Replace with actual text
    tokenized_inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized_inputs.input_ids.to(device)

    total_tokens_generated = 0
    total_time = 0

    # Warm-up run
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=1)

    # Measure throughput
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=10)  # Generate multiple tokens
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Calculate tokens generated
        batch_tokens_generated = outputs.shape[0] * outputs.shape[1]  # Batch size * sequence length
        total_tokens_generated += batch_tokens_generated
        total_time += (t1 - t0)

    # Calculate throughput as tokens generated per second
    mean_throughput = total_tokens_generated / (total_time * num_iterations)
    return mean_throughput



# Prepare token payload
def prepare_test_payload(tokenizer):
    return tokenizer(TOKENIZE_TEXT, return_tensors="pt").input_ids

# Send request and measure response time
def send_request(payload):
    start_time = time.time()
    from openai import OpenAI

    # Initialize the OpenAI client
    client = OpenAI(
     base_url="http://localhost:8000/v1",
     api_key="dummy-key"
     )

     # Create a streaming chat completion
    stream = client.chat.completions.create(
     model=MODEL_NAME,  # Model name doesn't matter
     messages=[{"role": "user", "content": TOKENIZE_TEXT}],
     stream=True,
    )

    # Collect and store the response
    response_text = []
    token_count = 0


    for chunk in stream:
       if chunk.choices[0].delta.content is not None:
          response_text.append(chunk.choices[0].delta.content)
          token_count += len(chunk.choices[0].delta.content.split())  # Approximate tokens by splitting on spaces

    # Combine the collected chunks into a full response
    full_response = "".join(response_text)
    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_per_second = token_count
    # print("Full Response:", full_response)
    # print(f"Tokens generated: {token_count}")
    # print(f"Elapsed time: {elapsed_time:.2f} seconds")
    # print(f"Throughput: {tokens_per_second:.2f} tokens/second")

    #print("Collected Response:", full_response)
    return elapsed_time, tokens_per_second

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
def benchmark_api(num_requests=10, concurrency_level=10):
    payload = prepare_test_payload(AutoTokenizer.from_pretrained(MODEL_NAME))
    system_metrics = []

    start_benchmark_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(num_requests)]
        response_times = []
        totaltoken = []

        # Collect system metrics during the benchmark
        while any(not f.done() for f in futures):
            system_metrics.append(get_system_metrics())
            time.sleep(0.1)

        for future in futures:
            response_time, token = future.result()
            response_times.append(response_time)
            totaltoken.append(token)

    end_benchmark_time = time.time()
    total_benchmark_time = end_benchmark_time - start_benchmark_time

    avg_cpu = np.mean([m["cpu_usage"] for m in system_metrics])
    avg_gpu = np.mean([m["gpu_usage"] for m in system_metrics]) if GPU_AVAILABLE else -1

    return {
        "total_requests": num_requests,
        "concurrency_level": concurrency_level,
        "total_time": total_benchmark_time,
        "avg_response_time": np.mean(response_times) * 1000,  # ms
        "success_rate": 100,
        "requests_per_second": sum(totaltoken) / total_benchmark_time,
        "avg_cpu_usage": avg_cpu,
        "avg_gpu_usage": avg_gpu
    }

    
def run_benchmarks():
    model, tokenizer, device = load_llama_model_with_quantization()
    calculate_theoretical_max_throughput(model)
    batch_sizes = [1, 8, 32, 64]

    batch_sizes = [1, 8, 32, 64]
    baseline_throughput = []

    print("Running token generation throughput tests...")
    for batch_size in batch_sizes:
      throughput = get_token_generation_throughput_batch(batch_size)
      baseline_throughput.append(throughput)
      print(f"Batch size {batch_size}: {throughput:.2f} tokens/sec")
    
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
        metrics = benchmark_api(concurrency_level=concurrency)
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
    print("Running benchmarks for Llama-based model with TorchAO quantization...")
    run_benchmarks()