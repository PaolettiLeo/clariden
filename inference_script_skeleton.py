import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_response(model, tokenizer, prompt: str, max_length: int = 512, 
                     temperature: float = 0.7) -> str:
    """Generate response from a model given a prompt"""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length//2)
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the original prompt)
        response = full_response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Simple model inference test")
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
                        help="Device to use (auto, cpu, cuda, etc.)")
    
    args = parser.parse_args()
    
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