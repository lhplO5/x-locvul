import torch
import time
import numpy as np
import gc

def profile_latency(model, tokenizer, input_texts, device, num_warmup=30, num_runs=100):
    """
    Profiles inference latency using precise CUDA events.
    Returns median and P95 latency in milliseconds.
    """
    # Tokenize outside the timing loop unless end-to-end includes tokenization
    # We will include tokenization to reflect real-world latency
    
    latencies = []
    
    # Warmup
    for i in range(num_warmup):
        text = input_texts[i % len(input_texts)]
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            _ = model(**inputs) if not hasattr(model, 'generate') else model.generate(**inputs, max_new_tokens=10)
            
    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for i in range(num_runs):
        text = input_texts[i % len(input_texts)]
        
        torch.cuda.synchronize()
        start_event.record()
        
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            if hasattr(model, 'generate'):
                _ = model.generate(**inputs, max_new_tokens=100) # Stage 2/3 generation
            else:
                _ = model(**inputs) # Stage 1 classification
                
        end_event.record()
        torch.cuda.synchronize()
        
        latencies.append(start_event.elapsed_time(end_event))
        
    return np.median(latencies), np.percentile(latencies, 95)

def run_profiler():
    print("==================================================")
    print(" E4: Performance & Latency Profiler ")
    print("==================================================")
    
    if not torch.cuda.is_available():
        print("LỖI: Script này bắt buộc phải chạy trên GPU (CUDA) để dùng torch.cuda.Event.")
        return
        
    device = torch.device("cuda")
    dummy_text = ["int main() { char buf[10]; strcpy(buf, argv[1]); return 0; }"]
    
    print("\n[LƯU Ý] Chạy script này trên máy Local/Server có GPU ổn định, KHÔNG chạy trên Kaggle.")
    print("Đang khởi tạo Profiler... (Sẽ tốn thời gian tải model)\n")
    
    # Note: In a real environment, you would instantiate your actual models here.
    # To prevent this script from crashing if models aren't trained, we catch exceptions.
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, AutoModelForCausalLM
        
        # Stage 1
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print("Loading Stage 1 (CodeBERT)...")
        t0 = time.time()
        # Mocking with base model, replace with your checkpoint
        tokenizer_s1 = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model_s1 = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base").to(device)
        load_s1 = time.time() - t0
        
        med_s1, p95_s1 = profile_latency(model_s1, tokenizer_s1, dummy_text, device)
        peak_mem_s1 = torch.cuda.max_memory_allocated() / (1024**2)
        
        # Free memory
        del model_s1
        gc.collect()
        torch.cuda.empty_cache()
        
        # Stage 2
        torch.cuda.reset_peak_memory_stats()
        print("Loading Stage 2 (CodeT5)...")
        t0 = time.time()
        tokenizer_s2 = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        model_s2 = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").to(device)
        load_s2 = time.time() - t0
        
        med_s2, p95_s2 = profile_latency(model_s2, tokenizer_s2, dummy_text, device)
        peak_mem_s2 = torch.cuda.max_memory_allocated() / (1024**2)
        
        del model_s2
        gc.collect()
        torch.cuda.empty_cache()
        
        # Stage 3
        torch.cuda.reset_peak_memory_stats()
        print("Loading Stage 3 (Qwen 1.5B)...")
        t0 = time.time()
        tokenizer_s3 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
        model_s3 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", torch_dtype=torch.float16).to(device)
        load_s3 = time.time() - t0
        
        med_s3, p95_s3 = profile_latency(model_s3, tokenizer_s3, dummy_text, device)
        peak_mem_s3 = torch.cuda.max_memory_allocated() / (1024**2)
        
        # Print Report
        print("\n" + "="*50)
        print(" BÁO CÁO HIỆU NĂNG (PERFORMANCE REPORT)")
        print("="*50)
        print(f"{'Metric':<25} | {'Stage 1 (CodeBERT)':<18} | {'Stage 2 (CodeT5)':<18} | {'Stage 3 (Qwen)':<18}")
        print("-" * 85)
        print(f"{'Load Time (s)':<25} | {load_s1:<18.2f} | {load_s2:<18.2f} | {load_s3:<18.2f}")
        print(f"{'Median Latency (ms)':<25} | {med_s1:<18.2f} | {med_s2:<18.2f} | {med_s3:<18.2f}")
        print(f"{'P95 Latency (ms)':<25} | {p95_s1:<18.2f} | {p95_s2:<18.2f} | {p95_s3:<18.2f}")
        print(f"{'Peak VRAM (MB)':<25} | {peak_mem_s1:<18.2f} | {peak_mem_s2:<18.2f} | {peak_mem_s3:<18.2f}")
        print("=" * 85)
        print(f"End-to-End Pipeline Median Latency: {med_s1 + med_s2 + med_s3:.2f} ms")
        
    except Exception as e:
        print(f"Lỗi khi chạy profiler: {e}")
        print("Bạn cần tải đủ môi trường transformers, torch để chạy script này.")

if __name__ == "__main__":
    run_profiler()
