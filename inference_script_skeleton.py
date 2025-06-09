import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_resources():
    """Check and log GPU availability and resources"""
    logger.info("=== GPU Resource Check ===")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        logger.info(f"Number of GPUs: {gpu_count}")
        
        # Get current GPU
        current_device = torch.cuda.current_device()
        logger.info(f"Current GPU device: {current_device}")
        
        # Check each GPU
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Get memory usage
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            logger.info(f"  Memory allocated: {allocated:.2f} GB")
            logger.info(f"  Memory cached: {cached:.2f} GB")
            logger.info(f"  Memory free: {gpu_memory - cached:.2f} GB")
        
        # Reset to original device
        torch.cuda.set_device(current_device)
        
    else:
        logger.info("No CUDA GPUs available - will use CPU")
    
    logger.info("=== End GPU Check ===")
    return cuda_available

def generate_response(model, tokenizer, prompt: str, max_length: int = 512, 
                     temperature: float = 0.7) -> str:
    """Generate response from a model given a prompt"""
    try:
        # Get model device
        device = next(model.parameters()).device
        logger.info(f"Model is on device: {device}")
        
        # Check GPU memory before generation
        if torch.cuda.is_available() and device.type == 'cuda':
            gpu_id = device.index if device.index is not None else 0
            before_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            before_cached = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            logger.info(f"GPU memory before generation - Allocated: {before_allocated:.2f}GB, Cached: {before_cached:.2f}GB")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length//2)
        logger.info(f"Input tokens shape: {inputs['input_ids'].shape}")
        
        # Move to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logger.info(f"Inputs moved to device: {device}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                early_stopping=True
            )
        
        logger.info(f"Generated tokens shape: {outputs.shape}")
        
        # Check GPU memory after generation
        if torch.cuda.is_available() and device.type == 'cuda':
            after_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            after_cached = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            logger.info(f"GPU memory after generation - Allocated: {after_allocated:.2f}GB, Cached: {after_cached:.2f}GB")
            memory_diff = after_allocated - before_allocated
            logger.info(f"Memory difference: {memory_diff:.2f}GB")
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the original prompt)
        response = full_response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        # If it's a GPU memory error, provide specific info
        if "CUDA out of memory" in str(e):
            logger.error("GPU out of memory! Try reducing --max-length or using --device cpu")
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Simple model inference test with GPU monitoring")
    parser.add_argument("csv_file", help="CSV file to process")
    parser.add_argument("--model-dir", type=str, required=True, 
                        help="Path to the model directory")
    parser.add_argument("--output-file", type=str, 
                        help="Output CSV file (default: input_file_with_continuations.csv)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum length for generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--test-only", action="store_true",
                        help="Only test GPU resources and model loading, don't process CSV")
    
    args = parser.parse_args()
    
    # Check GPU resources first
    cuda_available = check_gpu_resources()
    
    # Determine actual device to use
    if args.device == "auto":
        if cuda_available:
            actual_device = "cuda"
            logger.info("Auto-selected device: CUDA")
        else:
            actual_device = "cpu"
            logger.info("Auto-selected device: CPU")
    else:
        actual_device = args.device
        logger.info(f"User-specified device: {actual_device}")
    
    # Test device compatibility
    try:
        if actual_device.startswith("cuda") and not cuda_available:
            logger.error(f"CUDA device requested but CUDA not available!")
            return
        
        # Test tensor creation on device
        test_tensor = torch.randn(10, 10).to(actual_device)
        logger.info(f"Successfully created test tensor on {test_tensor.device}")
        del test_tensor
        
    except Exception as e:
        logger.error(f"Error with device {actual_device}: {str(e)}")
        return
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        return
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        return
    
    # Load CSV
    logger.info(f"Loading CSV file: {args.csv_file}")
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return
    
    # Check if 'prompt' column exists
    if 'prompt' not in df.columns:
        logger.error("Column 'prompt' not found in CSV file")
        logger.info(f"Available columns: {list(df.columns)}")
        return
    
    logger.info(f"Found {len(df)} rows with prompts")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=args.device if args.device != "auto" else "auto",
            low_cpu_mem_usage=True
        )
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # Generate responses
    logger.info("Generating responses...")
    responses = []
    
    for idx, prompt in enumerate(df['prompt']):
        if pd.isna(prompt):
            responses.append("")
            continue
        
        logger.info(f"Processing prompt {idx + 1}/{len(df)}")
        response = generate_response(
            model, tokenizer, str(prompt), 
            max_length=args.max_length, 
            temperature=args.temperature
        )
        responses.append(response)
        
        # Print first few for debugging
        if idx < 3:
            logger.info(f"Prompt: {str(prompt)[:100]}...")
            logger.info(f"Response: {response[:100]}...")
    
    # Add responses to dataframe
    df['model_continuation'] = responses
    
    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        input_path = os.path.splitext(args.csv_file)
        output_file = f"{input_path[0]}_with_continuations{input_path[1]}"
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    try:
        df.to_csv(output_file, index=False)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return
    
    logger.info("Processing completed!")
    logger.info(f"Input: {args.csv_file} ({len(df)} rows)")
    logger.info(f"Output: {output_file}")

if __name__ == "__main__":
    main()