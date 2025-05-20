import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/capstor/scratch/cscs/leoplt/model_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "EleutherAI/pythia-6.9b",
    "tiiuae/falcon-7b",
    "bigscience/bloom-7b1",
    "mistralai/Mistral-7B-v0.1"
]

# clean directories
MODEL_TO_DIR = {
    "meta-llama/Llama-2-7b-hf": "llama2-7b",
    "EleutherAI/pythia-6.9b": "pythia-6.9b",
    "tiiuae/falcon-7b": "falcon-7b",
    "bigscience/bloom-7b1": "bloom-7b1",
    "mistralai/Mistral-7B-v0.1": "mistral-7b-v0.1"
}

def download_model(model_name, output_dir, use_auth=False, token=None):
    """
    Download a model and its tokenizer to a specific directory
    
    Args:
        model_name: Name of the model on Hugging Face Hub
        output_dir: Directory to save the model
        use_auth: Whether to use authentication for gated models
        token: Hugging Face token for authentication
    """
    model_dir = os.path.join(output_dir, MODEL_TO_DIR[model_name])
    
    try:
        #create the directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        #when downloading a model, two steps: tokenizer then model weights
        start_time = time.time()
        logger.info(f"Starting download of {model_name} to {model_dir}")
        
        logger.info(f"Downloading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            use_auth_token=token if use_auth else None
        )
        tokenizer.save_pretrained(model_dir)
        
        logger.info(f"Downloading model for {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=model_dir,
            use_auth_token=token if use_auth else None,
            local_files_only=False,
        )
        model.save_pretrained(model_dir)
        
        duration = time.time() - start_time
        logger.info(f"Successfully downloaded {model_name} in {duration:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download language models locally")
    parser.add_argument("--output-dir", type=str, default="./models", 
                        help="Base directory to save all models")
    parser.add_argument("--use-auth", action="store_true", 
                        help="Use authentication for gated models (like Llama-2)") # mistral and in the future llama
    parser.add_argument("--token", type=str, 
                        help="Hugging Face token for authentication")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_TO_DIR.keys()) + ["all"],
                        default=["all"], help="Specific models to download (default: all)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.use_auth and args.token:
        login(token=args.token)
        logger.info("Logged in to Hugging Face Hub")
    elif args.use_auth and not args.token:
        logger.warning("Authentication requested but no token provided. Models requiring authentication may fail.")
    
    models_to_download = MODELS if "all" in args.models else [m for m in MODELS if m in args.models]
    
    successful = 0
    failed = 0
    
    for model_name in models_to_download:
        logger.info(f"Processing model: {model_name}")
        
        success = download_model(
            model_name=model_name,
            output_dir=args.output_dir,
            use_auth=args.use_auth,
            token=args.token
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Download complete. Successfully downloaded {successful} models. Failed: {failed}")

    
    # Print the local paths
    logger.info("\nLocal paths for each model:")
    for model_name in models_to_download:
        dir_name = MODEL_TO_DIR[model_name]
        path = os.path.join(os.path.abspath(args.output_dir), dir_name)
        logger.info(f"{model_name} -> {path}")

if __name__ == "__main__":
    main()