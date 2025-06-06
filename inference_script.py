import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import time
from pathlib import Path
import gc
from typing import List, Dict, Tuple
import threading
import json
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model directory mapping from your download script
MODEL_TO_DIR = {
    "mistralai/Mistral-7B-v0.1": "mistral-7b-v0.1",
    "HuggingFaceH4/zephyr-7b-alpha": "zephyr-7b-alpha",
    "meta-llama/Llama-2-7b-hf": "llama2-7b",
    "meta-llama/Llama-2-13b-hf": "llama2-13b",
    "meta-llama/Llama-2-70b-hf": "llama2-70b",
    "EleutherAI/pythia-1b": "pythia-1b",
    "EleutherAI/pythia-2.8b": "pythia-2.8b",
    "EleutherAI/pythia-6.9b": "pythia-6.9b",
    "EleutherAI/pythia-12b": "pythia-12b",
    "tiiuae/falcon-7b": "falcon-7b",
    "tiiuae/falcon-40b": "falcon-40b",
    "ContextualAI/archangel_dpo_pythia2-8b": "archangel-dpo-pythia2-8b",
    "ContextualAI/archangel_ppo_pythia2-8b": "archangel-ppo-pythia2-8b"
}

class CheckpointManager:
    """Manages saving and loading of progress checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def get_checkpoint_path(self, csv_file: str) -> Path:
        """Get checkpoint file path for a CSV file"""
        csv_name = Path(csv_file).stem
        return self.checkpoint_dir / f"{csv_name}_checkpoint.json"
    
    def get_temp_csv_path(self, csv_file: str) -> Path:
        """Get temporary CSV file path for incremental saves"""
        csv_path = Path(csv_file)
        return csv_path.parent / f"{csv_path.stem}_temp_progress.csv"
    
    def save_checkpoint(self, csv_file: str, completed_models: List[str], 
                       failed_models: List[str], current_df: pd.DataFrame):
        """Save checkpoint with completed models and current dataframe"""
        checkpoint_data = {
            "csv_file": csv_file,
            "completed_models": completed_models,
            "failed_models": failed_models,
            "timestamp": time.time(),
            "total_rows": len(current_df)
        }
        
        checkpoint_path = self.get_checkpoint_path(csv_file)
        temp_csv_path = self.get_temp_csv_path(csv_file)
        
        # Save checkpoint metadata
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save current dataframe
        current_df.to_csv(temp_csv_path, index=False)
        
        logger.info(f"Checkpoint saved: {len(completed_models)} models completed for {csv_file}")
    
    def load_checkpoint(self, csv_file: str) -> Tuple[List[str], List[str], pd.DataFrame]:
        """Load checkpoint if exists, return (completed_models, failed_models, df)"""
        checkpoint_path = self.get_checkpoint_path(csv_file)
        temp_csv_path = self.get_temp_csv_path(csv_file)
        
        if not checkpoint_path.exists() or not temp_csv_path.exists():
            return [], [], pd.read_csv(csv_file)
        
        try:
            # Load checkpoint metadata
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Load dataframe with progress
            df = pd.read_csv(temp_csv_path)
            
            completed_models = checkpoint_data.get("completed_models", [])
            failed_models = checkpoint_data.get("failed_models", [])
            
            logger.info(f"Checkpoint loaded: {len(completed_models)} models already completed for {csv_file}")
            if failed_models:
                logger.info(f"Previously failed models: {failed_models}")
            
            return completed_models, failed_models, df
            
        except Exception as e:
            logger.error(f"Error loading checkpoint for {csv_file}: {str(e)}")
            return [], [], pd.read_csv(csv_file)
    
    def cleanup_checkpoint(self, csv_file: str):
        """Clean up checkpoint files after successful completion"""
        checkpoint_path = self.get_checkpoint_path(csv_file)
        temp_csv_path = self.get_temp_csv_path(csv_file)
        
        for path in [checkpoint_path, temp_csv_path]:
            if path.exists():
                try:
                    path.unlink()
                    logger.info(f"Cleaned up checkpoint file: {path}")
                except Exception as e:
                    logger.warning(f"Could not clean up {path}: {str(e)}")
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints"""
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                checkpoints.append({
                    "csv_file": data["csv_file"],
                    "completed_models": len(data["completed_models"]),
                    "timestamp": data["timestamp"]
                })
        return checkpoints
    """Manages loading and unloading of models to optimize memory usage"""
    
    def __init__(self, model_base_dir: str, device: str = "auto"):
        self.model_base_dir = model_base_dir
        self.device = device
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.lock = threading.Lock()
        
    def load_model(self, model_dir_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a model and tokenizer, unloading previous model if necessary"""
        with self.lock:
            model_path = os.path.join(self.model_base_dir, model_dir_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found: {model_path}")
            
            # Unload current model if different
            if self.current_model_name != model_dir_name:
                self._unload_current_model()
                
                logger.info(f"Loading model from {model_path}")
                start_time = time.time()
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device if self.device != "auto" else "auto",
                    low_cpu_mem_usage=True
                )
                
                self.current_model = model
                self.current_tokenizer = tokenizer
                self.current_model_name = model_dir_name
                
                load_time = time.time() - start_time
                logger.info(f"Model {model_dir_name} loaded in {load_time:.2f} seconds")
            
            return self.current_model, self.current_tokenizer
    
    def _unload_current_model(self):
        """Unload current model to free memory"""
        if self.current_model is not None:
            logger.info(f"Unloading model {self.current_model_name}")
            del self.current_model
            del self.current_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None

def generate_response(model, tokenizer, prompt: str, max_length: int = 512, 
                     temperature: float = 0.7, do_sample: bool = True) -> str:
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
                do_sample=do_sample,
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

def process_csv_with_model(csv_file: str, model_base_dir: str, model_dir_name: str, 
                          model_name: str, max_length: int, temperature: float, 
                          device: str) -> pd.DataFrame:
    """Process a single CSV file with a single model"""
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        
        if 'prompt' not in df.columns:
            logger.error(f"Column 'prompt' not found in {csv_file}")
            return df
        
        # Initialize model manager for this process
        model_manager = ModelManager(model_base_dir, device)
        model, tokenizer = model_manager.load_model(model_dir_name)
        
        # Generate responses
        responses = []
        column_name = f"{model_name.replace('/', '_').replace('-', '_')}_continuation"
        
        logger.info(f"Processing {len(df)} prompts with {model_name}")
        
        for idx, prompt in enumerate(df['prompt']):
            if pd.isna(prompt):
                responses.append("")
                continue
                
            response = generate_response(
                model, tokenizer, str(prompt), 
                max_length=max_length, 
                temperature=temperature
            )
            responses.append(response)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} prompts for {model_name}")
        
        # Add responses to dataframe
        df[column_name] = responses
        
        logger.info(f"Completed processing {csv_file} with {model_name}")
        return df
        
    except Exception as e:
        logger.error(f"Error processing {csv_file} with {model_name}: {str(e)}")
        return pd.read_csv(csv_file)  # Return original dataframe

def process_single_csv(args_tuple):
    """Wrapper function for multiprocessing with checkpoint support"""
    csv_file, model_base_dir, available_models, max_length, temperature, device, max_workers, resume = args_tuple
    
    logger.info(f"Starting processing of {csv_file}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Load checkpoint if resuming
    if resume:
        completed_models, failed_models, result_df = checkpoint_manager.load_checkpoint(csv_file)
    else:
        completed_models, failed_models = [], []
        result_df = pd.read_csv(csv_file)
    
    # Determine which models still need processing
    models_to_process = {}
    for model_name, model_dir in available_models.items():
        column_name = f"{model_name.replace('/', '_').replace('-', '_')}_continuation"
        
        if resume and model_name in completed_models:
            logger.info(f"Skipping {model_name} for {csv_file} (already completed)")
            continue
        elif resume and model_name in failed_models:
            logger.info(f"Retrying {model_name} for {csv_file} (previously failed)")
        
        models_to_process[model_name] = model_dir
    
    if not models_to_process:
        logger.info(f"All models already completed for {csv_file}")
        return csv_file, result_df
    
    logger.info(f"Processing {len(models_to_process)} models for {csv_file}")
    
    # Process remaining models one by one
    for model_name, model_dir in models_to_process.items():
        logger.info(f"Processing {csv_file} with {model_name}")
        
        try:
            temp_df = process_csv_with_model(
                csv_file, model_base_dir, model_dir, model_name,
                max_length, temperature, device
            )
            
            # Merge the new column
            column_name = f"{model_name.replace('/', '_').replace('-', '_')}_continuation"
            if column_name in temp_df.columns:
                result_df[column_name] = temp_df[column_name]
                completed_models.append(model_name)
                
                # Remove from failed list if it was there
                if model_name in failed_models:
                    failed_models.remove(model_name)
                
                logger.info(f"Successfully completed {model_name} for {csv_file}")
            else:
                failed_models.append(model_name)
                logger.error(f"Failed to generate column for {model_name}")
        
        except Exception as e:
            logger.error(f"Error processing {csv_file} with {model_name}: {str(e)}")
            if model_name not in failed_models:
                failed_models.append(model_name)
        
        # Save checkpoint after each model
        checkpoint_manager.save_checkpoint(csv_file, completed_models, failed_models, result_df)
        
        # Force garbage collection after each model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Log final results for this CSV
    logger.info(f"Completed processing {csv_file}: {len(completed_models)} successful, {len(failed_models)} failed")
    if failed_models:
        logger.warning(f"Failed models for {csv_file}: {failed_models}")
    
    return csv_file, result_df

def main():
    parser = argparse.ArgumentParser(description="Run inference on multiple CSV files with multiple models")
    parser.add_argument("csv_files", nargs="+", help="CSV files to process")
    parser.add_argument("--model-dir", type=str, required=True, 
                        help="Base directory containing the downloaded models")
    parser.add_argument("--output-suffix", type=str, default="_with_continuations",
                        help="Suffix to add to output CSV files")
    parser.add_argument("--models", nargs="+", 
                        choices=list(MODEL_TO_DIR.keys()) + ["all"],
                        default=["all"], 
                        help="Specific models to use for inference (default: all)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum length for generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Maximum number of parallel workers for CSV processing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous checkpoints if available")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List available checkpoints and exit")
    parser.add_argument("--cleanup-checkpoints", action="store_true",
                        help="Clean up all checkpoint files and exit")
    
    args = parser.parse_args()
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Handle checkpoint listing
    if args.list_checkpoints:
        checkpoints = checkpoint_manager.list_checkpoints()
        if checkpoints:
            logger.info("Available checkpoints:")
            for cp in checkpoints:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cp['timestamp']))
                logger.info(f"  {cp['csv_file']}: {cp['completed_models']} models completed at {timestamp}")
        else:
            logger.info("No checkpoints found")
        return
    
    # Handle checkpoint cleanup
    if args.cleanup_checkpoints:
        checkpoint_files = list(checkpoint_manager.checkpoint_dir.glob("*"))
        for file in checkpoint_files:
            try:
                file.unlink()
                logger.info(f"Deleted {file}")
            except Exception as e:
                logger.error(f"Could not delete {file}: {str(e)}")
        logger.info("Checkpoint cleanup completed")
        return
    
    # Determine which models to use
    if "all" in args.models:
        models_to_use = MODEL_TO_DIR
    else:
        models_to_use = {k: v for k, v in MODEL_TO_DIR.items() if k in args.models}
    
    # Check which models are actually available
    available_models = {}
    for model_name, model_dir in models_to_use.items():
        model_path = os.path.join(args.model_dir, model_dir)
        if os.path.exists(model_path):
            available_models[model_name] = model_dir
            logger.info(f"Found model: {model_name} at {model_path}")
        else:
            logger.warning(f"Model directory not found: {model_path}")
    
    if not available_models:
        logger.error("No models found! Please check your model directory.")
        return
    
    # Validate CSV files
    valid_csv_files = []
    for csv_file in args.csv_files:
        if os.path.exists(csv_file):
            # Check if 'prompt' column exists
            try:
                df = pd.read_csv(csv_file)
                if 'prompt' in df.columns:
                    valid_csv_files.append(csv_file)
                    logger.info(f"Valid CSV file: {csv_file} ({len(df)} rows)")
                else:
                    logger.error(f"CSV file {csv_file} does not contain 'prompt' column")
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {str(e)}")
        else:
            logger.error(f"CSV file not found: {csv_file}")
    
    if not valid_csv_files:
        logger.error("No valid CSV files found!")
        return
    
    logger.info(f"Processing {len(valid_csv_files)} CSV files with {len(available_models)} models")
    if args.resume:
        logger.info("Resume mode enabled - will continue from previous checkpoints")
    
    # Prepare arguments for parallel processing
    process_args = [
        (csv_file, args.model_dir, available_models, args.max_length, 
         args.temperature, args.device, args.max_workers, args.resume)
        for csv_file in valid_csv_files
    ]
    
    # Process CSV files
    start_time = time.time()
    
    if args.max_workers == 1:
        # Sequential processing
        results = []
        for args_tuple in process_args:
            results.append(process_single_csv(args_tuple))
    else:
        # Parallel processing of CSV files
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(executor.map(process_single_csv, process_args))
    
    # Save final results and cleanup checkpoints
    for csv_file, result_df in results:
        # Create output filename
        csv_path = Path(csv_file)
        output_file = csv_path.parent / f"{csv_path.stem}{args.output_suffix}.csv"
        
        # Save the result
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved final results to {output_file}")
        
        # Clean up checkpoint files on successful completion
        checkpoint_manager.cleanup_checkpoint(csv_file)
    
    total_time = time.time() - start_time
    logger.info(f"All processing completed in {total_time:.2f} seconds")
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"Processed {len(valid_csv_files)} CSV files")
    logger.info(f"Used {len(available_models)} models: {', '.join(available_models.keys())}")
    logger.info("Output files:")
    for csv_file, _ in results:
        csv_path = Path(csv_file)
        output_file = csv_path.parent / f"{csv_path.stem}{args.output_suffix}.csv"
        logger.info(f"  {output_file}")

if __name__ == "__main__":
    main()