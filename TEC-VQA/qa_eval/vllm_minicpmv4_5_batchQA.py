import argparse
import json
import os
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


def find_image_path(lang_dir: Path, use_en: bool) -> str | None:
    if not lang_dir.exists():
        return None
    for fname in os.listdir(lang_dir):
        fpath = lang_dir / fname
        if use_en:
            if "translated_image_en" in fname or "英" in fname:
                return str(fpath)
            continue
        if "translated_image" in fname:
            continue
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")):
            return str(fpath)
    return None


class QADataset(Dataset):
    def __init__(self, qa_data, image_base_dir, tokenizer, system_prompt, use_en=False):
        self.qa_data = qa_data
        self.image_base_dir = Path(image_base_dir)
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.use_en = use_en

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, idx):
        item = self.qa_data[idx]
        if self.use_en:
            lang_dir = self.image_base_dir / item["语言"] / item["语言"] / item["图片编号"]
        else:
            lang_dir = self.image_base_dir / item["语言"] / item["图片编号"]

        image_path = find_image_path(lang_dir, self.use_en)
        if image_path is None:
            return None, {**item, "error": "image_not_found", "image_path": str(lang_dir)}

        question = item["最终问题"]
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"(<image>./</image>)\n{question}"},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        llm_inputs = {"prompt": prompt_text, "multi_modal_data": {"image": image}}
        metadata = item.copy()
        metadata["image_path"] = image_path
        return llm_inputs, metadata


def collate_fn(batch):
    llm_inputs_list = []
    valid_metadata_list = []
    error_metadata_list = []
    for item in batch:
        if item[0] is not None:
            llm_inputs_list.append(item[0])
            valid_metadata_list.append(item[1])
        else:
            error_metadata_list.append(item[1])
    return llm_inputs_list, valid_metadata_list, error_metadata_list


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Model path")
    parser.add_argument("--input_jsonl", type=str, default=str(data_dir / "all_qa_data.jsonl"))
    parser.add_argument("--image_base_dir", type=str, default=str(data_dir / "images"))
    parser.add_argument("--output_jsonl", type=str, default="", help="Default: data/qa_results/{model}.jsonl")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    parser.add_argument("--use_en", action="store_true", help="Use English-translated data layout")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)

    model_dir = args.model_dir
    model_name = os.path.basename(model_dir.rstrip("/")).lower()
    output_jsonl = args.output_jsonl or str(Path(__file__).resolve().parent.parent / "data" / "qa_results" / f"{model_name}.jsonl")
    os.makedirs(Path(output_jsonl).parent, exist_ok=True)

    print(f"input_jsonl={args.input_jsonl}")
    print(f"image_base_dir={args.image_base_dir}")
    print(f"output_jsonl={output_jsonl}")
    print(f"use_en={args.use_en}")

    llm = LLM(
        model=model_dir,
        limit_mm_per_prompt={"image": 1},
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    sampling_params = SamplingParams(
        stop_token_ids=stop_token_ids,
        temperature=0.0,
        top_p=0.95,
        repetition_penalty=1.05,
        max_tokens=256,
    )

    system_prompt = (
        "You are a helpful AI assistant that answers questions based on the given image. "
        "Answer directly and accurately in the same language as the question. "
        "If the answer is not visible in the image, say you cannot find the information."
    )

    qa_data = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_data.append(json.loads(line))
    print(f"Total QA pairs: {len(qa_data)}")

    if not qa_data:
        raise SystemExit("No data to process.")

    dataset = QADataset(
        qa_data=qa_data,
        image_base_dir=args.image_base_dir,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        use_en=args.use_en,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            {
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": True,
            }
        )
    dataloader = DataLoader(**loader_kwargs)

    total_processed = 0
    with open(output_jsonl, "w", encoding="utf-8") as output_file:
        for batch_idx, (batch_llm_inputs, batch_valid_metadata, batch_error_metadata) in enumerate(
            tqdm(dataloader, desc="Processing QA")
        ):
            for metadata in batch_error_metadata:
                metadata["predicted_answer"] = ""
                metadata["inference_error"] = metadata.get("error", "unknown_error")
                output_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            if not batch_llm_inputs:
                continue

            set_seed(42)
            outputs = llm.generate(batch_llm_inputs, sampling_params=sampling_params)
            for idx, output in enumerate(outputs):
                output_text = output.outputs[0].text.strip()
                metadata = batch_valid_metadata[idx]
                metadata["predicted_answer"] = output_text
                output_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            total_processed += len(batch_llm_inputs)
            if (batch_idx + 1) % 10 == 0:
                output_file.flush()

    print(f"All {total_processed} QA pairs processed.")
    print(f"Results saved to: {output_jsonl}")
