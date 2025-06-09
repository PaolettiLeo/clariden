import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
from pathlib import Path
import gc
import json
from typing import List, Dict

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


def generate_batch_responses(model, tokenizer, prompts: List[str], max_length: int, temperature: float, batch_size: int = 4):
    device = next(model.parameters()).device
    responses = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        valid_prompts = [str(p) if pd.notna(p) else "" for p in batch_prompts]

        if not any(valid_prompts):
            responses.extend([""] * len(batch_prompts))
            continue

        try:
            inputs = tokenizer(
                valid_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length // 2,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            batch_responses = []
            for j, output in enumerate(outputs):
                text = tokenizer.decode(output, skip_special_tokens=True)
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


def process_csv_files(csv_files: List[str], base_dir: str, selected_models: Dict[str, str],
                      max_length: int, temperature: float, batch_size: int = 4):
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "prompt" not in df.columns:
            logger.error(f"No prompt column in {csv_file}")
            continue

        for model_name, model_subdir in selected_models.items():
            model_path = os.path.join(base_dir, model_subdir)
            logger.info(f"Loading model {model_name} from {model_path}")

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=True
                )
            except Exception as e:
                logger.warning(f"Fast tokenizer failed, using slow one: {e}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=False
                )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            model.eval()
            prompts = df["prompt"].tolist()

            logger.info(f"Generating responses for {len(prompts)} prompts with model {model_name}")
            start_time = time.time()

            responses = generate_batch_responses(
                model, tokenizer, prompts, max_length, temperature, batch_size
            )

            elapsed = time.time() - start_time
            logger.info(f"Generated {len(prompts)} responses in {elapsed:.2f} seconds")

            col_name = f"{model_name.replace('/', '_').replace('-', '_')}_cont"
            df[col_name] = responses

            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        output_path = Path(csv_file).parent / f"{Path(csv_file).stem}_completed.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on CSVs without parallelization")
    parser.add_argument("csv_files", nargs="+", help="CSV files to process")
    parser.add_argument("--model-dir", required=True, help="Directory containing model subdirectories")
    parser.add_argument("--models", nargs="+", default=["all"], help="Models to use")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()

    if "all" in args.models:
        selected_models = MODEL_TO_DIR
    else:
        selected_models = {k: v for k, v in MODEL_TO_DIR.items() if k in args.models}

    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return

    logger.info(f"Processing {len(args.csv_files)} CSV files with {len(selected_models)} models")
    process_csv_files(
        args.csv_files,
        args.model_dir,
        selected_models,
        args.max_length,
        args.temperature,
        args.batch_size
    )
    logger.info("All processing completed")


if __name__ == "__main__":
    main()
