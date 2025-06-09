import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from pathlib import Path
import gc
from typing import List, Tuple
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
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def get_checkpoint_path(self, csv_file: str) -> Path:
        stem = Path(csv_file).stem
        return self.checkpoint_dir / f"{stem}_checkpoint.json"

    def get_temp_csv_path(self, csv_file: str) -> Path:
        p = Path(csv_file)
        return p.parent / f"{p.stem}_temp.csv"

    def save(self, csv_file: str, completed: List[str], failed: List[str], df: pd.DataFrame):
        data = {"csv": csv_file, "done": completed, "failed": failed, "time": time.time()}
        with open(self.get_checkpoint_path(csv_file), 'w') as f:
            json.dump(data, f, indent=2)
        df.to_csv(self.get_temp_csv_path(csv_file), index=False)
        logger.info(f"Checkpoint saved for {csv_file}: {len(completed)} done")

    def load(self, csv_file: str) -> Tuple[List[str], List[str], pd.DataFrame]:
        cp = self.get_checkpoint_path(csv_file)
        tmp = self.get_temp_csv_path(csv_file)
        if not cp.exists() or not tmp.exists():
            return [], [], pd.read_csv(csv_file)
        try:
            d = json.loads(cp.read_text())
            df = pd.read_csv(tmp)
            return d.get('done', []), d.get('failed', []), df
        except Exception as e:
            logger.error(f"Failed loading checkpoint: {e}")
            return [], [], pd.read_csv(csv_file)

class ModelManager:
    def __init__(self, base_dir: str, device: str):
        self.base_dir = base_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.name = None
        self.lock = threading.Lock()

    def load(self, model_dir: str):
        with self.lock:
            path = os.path.join(self.base_dir, model_dir)
            if self.name != model_dir:
                if self.model is not None:
                    del self.model
                    del self.tokenizer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                logger.info(f"Loading {path}")
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)
                self.name = model_dir
            return self.model, self.tokenizer

    def unload(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.name = None


def generate_response(model, tokenizer, prompt: str, max_len: int, temp: float) -> str:
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_len//2)
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k,v in inputs.items()}
    out = model.generate(**inputs, max_length=max_len, temperature=temp, do_sample=True,
                          pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return txt[len(prompt):].strip()


def process_model(csv_file, base_dir, model_dir, model_name, max_len, temp, device):
    mgr = ModelManager(base_dir, device)
    model, tokenizer = mgr.load(model_dir)
    df = pd.read_csv(csv_file)
    results = []
    for p in df['prompt']:
        if pd.isna(p):
            results.append('')
        else:
            results.append(generate_response(model, tokenizer, str(p), max_len, temp))
    out_df = pd.DataFrame({'prompt': df['prompt'], f'{model_name}_continuation': results})
    output_file = Path(csv_file).with_suffix('').stem + f'_{model_name.replace('/', '_')}.csv'
    out_df.to_csv(output_file, index=False)
    logger.info(f"Saved {output_file}")
    return model_name


def process_single_csv(args):
    csv_file, base_dir, models, max_len, temp, resume = args
    logger.info(f"Start {csv_file}")
    cp = CheckpointManager()
    done, failed, _ = cp.load(csv_file) if resume else ([], [], None)
    to_run = {k:v for k,v in models.items() if k not in done}
    if not to_run:
        logger.info("Nothing to do")
        return csv_file, done
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    futures = {}
    with ThreadPoolExecutor(max_workers=min(len(to_run), gpus)) as exe:
        for i,(name,md) in enumerate(to_run.items()):
            dev = f"cuda:{i % gpus}" if torch.cuda.is_available() else 'cpu'
            futures[exe.submit(process_model, csv_file, base_dir, md, name, max_len, temp, dev)] = name
        for fut in as_completed(futures):
            model_name = futures[fut]
            try:
                finished = fut.result()
                done.append(finished)
                if finished in failed:
                    failed.remove(finished)
                logger.info(f"Done {finished}")
            except Exception as e:
                failed.append(model_name)
                logger.error(f"Failed {model_name}: {e}")
            cp.save(csv_file, done, failed, pd.DataFrame())
    return csv_file, done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_files', nargs='+')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--models', nargs='+', default=['all'])
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    models = MODEL_TO_DIR if 'all' in args.models else {k:v for k,v in MODEL_TO_DIR.items() if k in args.models}
    available = {k:v for k,v in models.items() if os.path.exists(os.path.join(args.model_dir, v))}
    tasks = [(cf, args.model_dir, available, args.max_length, args.temperature, args.resume) for cf in args.csv_files]
    for t in tasks:
        process_single_csv(t)

if __name__ == '__main__':
    main()
