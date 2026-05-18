import argparse
import base64
import json
import mimetypes
import os
import time
from functools import wraps
from multiprocessing import Manager, Pool
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# Globals for multiprocessing workers.
client = None
write_lock = None
failed_lock = None
model_name = None


def init_worker(write_l, failed_l, api_key, gpt_model):
    global client, write_lock, failed_lock, model_name
    client = OpenAI(api_key=api_key)
    write_lock = write_l
    failed_lock = failed_l
    model_name = gpt_model


def retry_on_failure(max_retries=3, delay=1, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    wait_time = delay * (backoff ** (retries - 1))
                    print(f"Retry {retries}/{max_retries} after {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
            return None

        return wrapper

    return decorator


def find_image_path(image_base_dir: str, qa_item: dict, use_en: bool) -> tuple[str | None, str]:
    if use_en:
        lang_dir = os.path.join(image_base_dir, qa_item["语言"], qa_item["语言"], qa_item["图片编号"])
    else:
        lang_dir = os.path.join(image_base_dir, qa_item["语言"], qa_item["图片编号"])

    if os.path.exists(lang_dir):
        for fname in os.listdir(lang_dir):
            if use_en:
                if "translated_image_en" in fname or "英" in fname:
                    return os.path.join(lang_dir, fname), lang_dir
                continue
            if "translated_image" in fname:
                continue
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")):
                return os.path.join(lang_dir, fname), lang_dir
    return None, lang_dir


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"


def call_openai_vision(image_path, question, system_prompt):
    base64_image = encode_image(image_path)
    mime_type = get_mime_type(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                    {"type": "text", "text": question},
                ],
            },
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


@retry_on_failure(max_retries=3, delay=1, backoff=2)
def process_single_qa(args):
    qa_item, image_base_dir, system_prompt, output_file, use_en = args

    try:
        image_path, lang_dir = find_image_path(image_base_dir, qa_item, use_en)
        if image_path is None or not os.path.exists(image_path):
            result = qa_item.copy()
            result["predicted_answer"] = ""
            result["inference_error"] = "image_not_found"
            result["image_path"] = lang_dir
            with write_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
            return {"status": "failed", "error": "image_not_found"}

        question = qa_item["最终问题"]
        predicted_answer = call_openai_vision(image_path, question, system_prompt)

        result = qa_item.copy()
        result["image_path"] = image_path
        result["predicted_answer"] = predicted_answer.strip()
        with write_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
        return {"status": "success"}

    except Exception as e:
        error_result = qa_item.copy()
        error_result["predicted_answer"] = ""
        error_result["inference_error"] = str(e)
        with failed_lock:
            print(f"Failed: {qa_item.get('语言', '')}/{qa_item.get('图片编号', '')} - {str(e)}")
        with write_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                f.flush()
        return {"status": "failed", "error": str(e)}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--input_jsonl", type=str, default=str(data_dir / "all_qa_data.jsonl"))
    parser.add_argument("--image_base_dir", type=str, default=str(data_dir / "images"))
    parser.add_argument("--output_jsonl", type=str, default=str(data_dir / "qa_results" / "gpt-4o.jsonl"))
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--use_en", action="store_true", help="Use English-translated data layout")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API key is empty. Set --api_key or environment variable OPENAI_API_KEY.")

    os.makedirs(Path(args.output_jsonl).parent, exist_ok=True)
    if os.path.exists(args.output_jsonl):
        print(f"Warning: {args.output_jsonl} exists and will be overwritten.")
        os.remove(args.output_jsonl)

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

    task_args = [
        (qa_item, args.image_base_dir, system_prompt, args.output_jsonl, args.use_en)
        for qa_item in qa_data
    ]

    manager = Manager()
    write_l = manager.Lock()
    failed_l = manager.Lock()

    print(f"Starting {args.num_processes} worker processes with model={args.model} ...")
    with Pool(
        processes=args.num_processes,
        initializer=init_worker,
        initargs=(write_l, failed_l, api_key, args.model),
    ) as pool:
        results = list(
            tqdm(pool.imap(process_single_qa, task_args), total=len(task_args), desc="Processing QA")
        )

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    print("Done.")
    print(f"Success: {success_count}/{len(qa_data)}")
    print(f"Failed: {failed_count}/{len(qa_data)}")
    print(f"Saved: {args.output_jsonl}")
