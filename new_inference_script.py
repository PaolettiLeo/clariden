import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from pathlib import Path
import gc
import threading
import json

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

class ModelManager:
    def __init__(self, base_dir: str, device_map: str = "auto"):
        self.base_dir = base_dir
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self.name = None
        self.lock = threading.Lock()

    def load_model(self, subdir: str):
        with self.lock:
            path = os.path.join(self.base_dir, subdir)
            if self.name != subdir:
                if self.model:
                    self._unload()

                logger.info(f"Loading model from {path}")

                # try fast tokenizer first (needed for llama2-hf)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        path,
                        trust_remote_code=True,
                        use_fast=True
                    )
                except Exception as fast_err:
                    logger.warning(
                        f"fast tokenizer load failed for {subdir}, falling back to slow, "
                        f"reason: {fast_err}"
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        path,
                        trust_remote_code=True,
                        use_fast=False
                    )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.name = subdir

            return self.model, self.tokenizer

    def _unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.name = None

def generate_response(model, tokenizer, prompt, max_length, temperature):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length//2)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        early_stopping=True
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def process_csv_with_model(csv_file, base_dir, subdir, model_name, max_length, temperature, device):
    df = pd.read_csv(csv_file)
    if "prompt" not in df:
        logger.error(f"No prompt column in {csv_file}")
        return df
    mgr = ModelManager(base_dir, device)
    model, tokenizer = mgr.load_model(subdir)

    total = len(df)
    col = f"{model_name.replace('/', '_').replace('-', '_')}_cont"
    responses = [None] * total
    logger.info(f"Starting {total} inferences for {model_name}")
    start = time.time()

    for i, prompt in enumerate(df["prompt"]):
        if pd.isna(prompt):
            responses[i] = ""
        else:
            responses[i] = generate_response(model, tokenizer, str(prompt), max_length, temperature)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remain = (total - (i + 1)) / rate if rate > 0 else float("inf")
            e_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            r_str = time.strftime("%H:%M:%S", time.gmtime(remain))
            logger.info(
                f"{model_name} progress {i + 1}/{total} inferences , elapsed {e_str} , eta {r_str}"
            )

    df[col] = responses
    out_name = f"{Path(csv_file).stem}_{model_name.replace('/', '_')}.csv"
    df.to_csv(Path(csv_file).parent / out_name, index=False)
    logger.info(f"Wrote results to {out_name}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Run inference on CSVs")
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if "all" in args.models:
        sel = MODEL_TO_DIR
    else:
        sel = {k: v for k, v in MODEL_TO_DIR.items() if k in args.models}

    with ThreadPoolExecutor(max_workers=min(len(sel), args.max_workers)) as exe:
        futures = []
        for csv in args.csv_files:
            for model_name, subdir in sel.items():
                futures.append(
                    exe.submit(
                        process_csv_with_model,
                        csv, args.model_dir, subdir, model_name,
                        args.max_length, args.temperature, args.device
                    )
                )
        for f in futures:
            f.result()

if __name__ == "__main__":
    main()
