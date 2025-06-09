import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from multiprocessing import Process, Queue, Manager
import logging
import time
from pathlib import Path
import gc
import json
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s , %(levelname)s , %(message)s',
    handlers=[
        logging.FileHandler("model_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_TO_DIR = {
    #"mistralai/Mistral-7B-v0.1": "mistral-7b-v0.1",
    "HuggingFaceH4/zephyr-7b-alpha": "zephyr-7b-alpha",
    "meta-llama/Llama-2-7b-hf": "llama2-7b",
    "meta-llama/Llama-2-13b-hf": "llama2-13b"
    #"meta-llama/Llama-2-70b-hf": "llama2-70b",
    #"EleutherAI/pythia-1b": "pythia-1b",
    #"EleutherAI/pythia-2.8b": "pythia-2.8b",
    #"EleutherAI/pythia-6.9b": "pythia-6.9b",
    #"EleutherAI/pythia-12b": "pythia-12b",
    #"tiiuae/falcon-7b": "falcon-7b",
    #"tiiuae/falcon-40b": "falcon-40b",
    #"ContextualAI/archangel_dpo_pythia2-8b": "archangel-dpo-pythia2-8b",
    #"ContextualAI/archangel_ppo_pythia2-8b": "archangel-ppo-pythia2-8b"
}

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def get_checkpoint_path(self, csv_file: str) -> Path:
        name = Path(csv_file).stem
        return self.checkpoint_dir / f"{name}_checkpoint.json"

    def get_temp_path(self, csv_file: str) -> Path:
        p = Path(csv_file)
        return p.parent / f"{p.stem}_temp_progress.csv"

    def save_checkpoint(self, csv_file, completed, failed, df):
        data = {
            "csv_file": csv_file,
            "completed_models": completed,
            "failed_models": failed,
            "timestamp": time.time(),
            "total_rows": len(df)
        }
        cp = self.get_checkpoint_path(csv_file)
        tmp = self.get_temp_path(csv_file)
        with open(cp, "w") as f:
            json.dump(data, f, indent=2)
        df.to_csv(tmp, index=False)
        logger.info(f"{len(completed)} models done for {csv_file}")

    def load_checkpoint(self, csv_file):
        cp = self.get_checkpoint_path(csv_file)
        tmp = self.get_temp_path(csv_file)
        if not cp.exists() or not tmp.exists():
            return [], [], pd.read_csv(csv_file)
        try:
            with open(cp) as f:
                data = json.load(f)
            df = pd.read_csv(tmp)
            return data.get("completed_models", []), data.get("failed_models", []), df
        except Exception as e:
            logger.error(f"Error loading checkpoint , {e}")
            return [], [], pd.read_csv(csv_file)

    def cleanup(self, csv_file):
        for path in [self.get_checkpoint_path(csv_file), self.get_temp_path(csv_file)]:
            if path.exists():
                try:
                    path.unlink()
                    logger.info(f"Removed {path}")
                except Exception as e:
                    logger.warning(f"Could not remove {path} , {e}")


def generate_batch_responses(model, tokenizer, prompts: List[str], max_length: int, temperature: float, batch_size: int = 4):
    """Generate responses for a batch of prompts"""
    device = next(model.parameters()).device
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Filter out None/NaN prompts
        valid_prompts = [str(p) if pd.notna(p) else "" for p in batch_prompts]
        
        if not any(valid_prompts):  # Skip if all prompts are empty
            responses.extend([""] * len(batch_prompts))
            continue
            
        try:
            # Tokenize batch
            inputs = tokenizer(
                valid_prompts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length//2,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode responses
            batch_responses = []
            for j, output in enumerate(outputs):
                text = tokenizer.decode(output, skip_special_tokens=True)
                # Remove the original prompt from the response
                original_prompt = valid_prompts[j]
                if text.startswith(original_prompt):
                    response = text[len(original_prompt):].strip()
                else:
                    response = text.strip()
                batch_responses.append(response)
            
            responses.extend(batch_responses)
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            responses.extend(["ERROR"] * len(batch_prompts))
    
    return responses


def model_worker(gpu_id: int, model_path: str, model_name: str, work_queue: Queue, 
                result_queue: Queue, max_length: int, temperature: float, batch_size: int = 4):
    """Worker process that handles one model on one GPU"""
    
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = f"cuda:0"  # Since we set CUDA_VISIBLE_DEVICES, use cuda:0
    
    logger.info(f"Worker starting: GPU {gpu_id}, Model {model_name}")
    
    try:
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )
        except Exception as e:
            logger.warning(f"Fast tokenizer failed for {model_name}, using slow: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logger.info(f"Loading {model_name} on GPU {gpu_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"Model {model_name} loaded on GPU {gpu_id}")
        
        # Process work items
        while True:
            try:
                work_item = work_queue.get(timeout=1)
                if work_item is None:  # Poison pill
                    break
                    
                csv_file, prompts = work_item
                logger.info(f"GPU {gpu_id} processing {len(prompts)} prompts for {csv_file}")
                
                start_time = time.time()
                responses = generate_batch_responses(
                    model, tokenizer, prompts, max_length, temperature, batch_size
                )
                
                elapsed = time.time() - start_time
                rate = len(prompts) / elapsed if elapsed > 0 else 0
                logger.info(f"GPU {gpu_id} completed {len(prompts)} inferences in {elapsed:.2f}s ({rate:.2f} inf/s)")
                
                result_queue.put((csv_file, model_name, responses))
                
            except Exception as e:
                if "Empty" not in str(e):  # Ignore queue timeout
                    logger.error(f"Error in worker GPU {gpu_id}: {e}")
                    result_queue.put((csv_file, model_name, "ERROR"))
                break
                
    except Exception as e:
        logger.error(f"Failed to load model {model_name} on GPU {gpu_id}: {e}")
        result_queue.put((None, model_name, "LOAD_ERROR"))
    
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Worker GPU {gpu_id} finished")


def process_csv_files(csv_files: List[str], base_dir: str, selected_models: Dict[str, str], 
                     max_length: int, temperature: float, num_gpus: int = 4, batch_size: int = 4):
    """Process CSV files using multiple GPUs"""
    
    # Create queues for work distribution
    work_queue = Queue()
    result_queue = Queue()
    
    # Read all CSV files and prepare work items
    csv_data = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "prompt" not in df.columns:
            logger.error(f"No prompt column in {csv_file}")
            continue
        csv_data[csv_file] = df
        
        # Add work items for each model
        for model_name in selected_models.keys():
            prompts = df["prompt"].tolist()
            work_queue.put((csv_file, prompts))
    
    # Add poison pills (one per GPU)
    for _ in range(num_gpus):
        work_queue.put(None)
    
    # Start worker processes
    processes = []
    model_names = list(selected_models.keys())
    
    for gpu_id in range(min(num_gpus, len(model_names))):
        if gpu_id < len(model_names):
            model_name = model_names[gpu_id]
            model_subdir = selected_models[model_name]
            model_path = os.path.join(base_dir, model_subdir)
            
            p = Process(
                target=model_worker,
                args=(gpu_id, model_path, model_name, work_queue, result_queue, 
                      max_length, temperature, batch_size)
            )
            p.start()
            processes.append(p)
    
    # Collect results
    results = {}
    expected_results = len(csv_files) * len(selected_models)
    collected = 0
    
    while collected < expected_results:
        try:
            result = result_queue.get(timeout=10)
            csv_file, model_name, responses = result
            
            if csv_file is None or responses == "LOAD_ERROR":
                logger.error(f"Model {model_name} failed to load")
                collected += len(csv_files)  # Skip all CSV files for this model
                continue
                
            if responses == "ERROR":
                logger.error(f"Error processing {csv_file} with {model_name}")
                collected += 1
                continue
            
            # Store results
            if csv_file not in results:
                results[csv_file] = {}
            results[csv_file][model_name] = responses
            collected += 1
            
            logger.info(f"Collected results for {csv_file} + {model_name} ({collected}/{expected_results})")
            
        except Exception as e:
            logger.error(f"Error collecting results: {e}")
            break
    
    # Wait for all processes to finish
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    # Save results
    for csv_file, model_results in results.items():
        df = csv_data[csv_file].copy()
        
        for model_name, responses in model_results.items():
            col_name = f"{model_name.replace('/', '_').replace('-', '_')}_cont"
            df[col_name] = responses
        
        # Save updated CSV
        output_path = Path(csv_file).parent / f"{Path(csv_file).stem}_completed.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on CSVs with multi-GPU support")
    parser.add_argument("csv_files", nargs="+", help="CSV files to process")
    parser.add_argument("--model-dir", required=True, help="Directory containing model subdirectories")
    parser.add_argument("--models", nargs="+", default=["all"], help="Models to use")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()

    # Select models
    if "all" in args.models:
        selected_models = MODEL_TO_DIR
    else:
        selected_models = {k: v for k, v in MODEL_TO_DIR.items() if k in args.models}
    
    logger.info(f"Processing {len(args.csv_files)} CSV files with {len(selected_models)} models on {args.num_gpus} GPUs")
    logger.info(f"Selected models: {list(selected_models.keys())}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return
    
    available_gpus = torch.cuda.device_count()
    if available_gpus < args.num_gpus:
        logger.warning(f"Only {available_gpus} GPUs available, requested {args.num_gpus}")
        args.num_gpus = available_gpus
    
    # Process files
    process_csv_files(
        args.csv_files, 
        args.model_dir, 
        selected_models,
        args.max_length, 
        args.temperature, 
        args.num_gpus,
        args.batch_size
    )
    
    logger.info("All processing completed")


if __name__ == "__main__":
    main()