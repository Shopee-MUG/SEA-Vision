"""
Dolphin image -> Markdown inference script for SEA-DocBench.

This script reads documents (.jpg/.png/.pdf) from --input-dir and writes
one Markdown file per sample into --save-dir, which is exactly the layout
expected by `configs/end2end_dolphin.yaml` (the `prediction.data_path`
field).

Model weights are pulled from HuggingFace
(https://huggingface.co/ByteDance/Dolphin-1.5) — pass `--model-id` to
point at a different checkpoint or local path. All layout/decoding
helpers from the upstream Dolphin project are vendored under
`tools/model_infer/dolphin_utils/`, so no extra `git clone` is needed.

Original code derived from the Dolphin project:
    Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
    SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

# Make the sibling `dolphin_utils` package importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dolphin_utils.utils import (  # noqa: E402
    convert_pdf_to_images,
    parse_layout_string,
    prepare_image,
    process_coordinates,
    save_combined_pdf_results,
    save_figure_to_local,
    save_outputs,
    setup_output_dirs,
)


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model

        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        self.tokenizer = self.processor.tokenizer

    def chat(self, prompt, image):
        """Process an image or batch of images with the given prompt(s)"""
        is_batch = isinstance(image, list)

        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)

        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values.half().to(self.device)

        prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
        )

        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)

        outputs = self.model.generate(
            pixel_values=batch_pixel_values,
            decoder_input_ids=batch_prompt_ids,
            decoder_attention_mask=batch_attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            temperature=1.0,
        )

        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)

        results = []
        for i, sequence in enumerate(sequences):
            cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
            results.append(cleaned)

        if not is_batch:
            return results[0]
        return results


def process_document(document_path, model, save_dir, max_batch_size=None):
    """Parse documents with two stages - handles both images and PDFs."""
    file_ext = os.path.splitext(document_path)[1].lower()

    if file_ext == ".pdf":
        images = convert_pdf_to_images(document_path)
        if not images:
            raise Exception(f"Failed to convert PDF {document_path} to images")

        all_results = []
        for page_idx, pil_image in enumerate(images):
            print(f"Processing page {page_idx + 1}/{len(images)}")
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"

            json_path, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, max_batch_size, save_individual=False
            )
            all_results.append({"page_number": page_idx + 1, "elements": recognition_results})

        combined_json_path = save_combined_pdf_results(all_results, document_path, save_dir)
        return combined_json_path, all_results

    pil_image = Image.open(document_path).convert("RGB")
    base_name = os.path.splitext(os.path.basename(document_path))[0]
    return process_single_image(pil_image, model, save_dir, base_name, max_batch_size)


def process_single_image(image, model, save_dir, image_name, max_batch_size=None, save_individual=True):
    """Process a single image (either from file or converted from PDF page)."""
    layout_output = model.chat("Parse the reading order of this document.", image)

    padded_image, dims = prepare_image(image)
    recognition_results = process_elements(
        layout_output, padded_image, dims, model, max_batch_size, save_dir, image_name
    )

    json_path = None
    if save_individual:
        dummy_image_path = f"{image_name}.jpg"
        json_path = save_outputs(recognition_results, dummy_image_path, save_dir)

    return json_path, recognition_results


def process_elements(layout_results, padded_image, dims, model, max_batch_size, save_dir=None, image_name=None):
    """Parse all document elements with parallel decoding."""
    layout_results = parse_layout_string(layout_results)

    text_elements = []
    table_elements = []
    figure_results = []
    previous_box = None
    reading_order = 0

    for bbox, label in layout_results:
        try:
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )

            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                if label == "fig":
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    figure_results.append({
                        "label": label,
                        "text": f"![Figure](figures/{figure_filename})",
                        "figure_path": f"figures/{figure_filename}",
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    })
                else:
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }
                    if label == "tab":
                        table_elements.append(element_info)
                    else:
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    recognition_results = figure_results.copy()

    if text_elements:
        text_results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(text_results)

    if table_elements:
        table_results = process_element_batch(table_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(table_results)

    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    """Process elements of the same type in batches."""
    results = []

    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)

    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i + batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        prompts_list = [prompt] * len(crops_list)

        batch_results = model.chat(prompts_list, crops_list)

        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
            })

    return results


def collect_documents(input_dir):
    """Gather all supported documents (jpg/jpeg/png/pdf) under input_dir."""
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.pdf", "*.PDF")
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pat)))
    return sorted(set(files))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Dolphin image->Markdown inference for SEA-DocBench."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input images (.jpg/.png) or PDFs (.pdf).",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Directory where Markdown predictions will be written. "
             "Set this to the same path as `prediction.data_path` in your evaluation yaml.",
    )
    parser.add_argument(
        "--model-id",
        default="ByteDance/Dolphin-1.5",
        help="HuggingFace model id or local path to a Dolphin checkpoint "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Maximum batch size for element-level decoding (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the document shuffling in distributed mode "
             "(default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = DOLPHIN(args.model_id)

    document_files = collect_documents(args.input_dir)
    if not document_files:
        raise SystemExit(f"No supported documents found under {args.input_dir}")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    random.seed(args.seed)
    random.shuffle(document_files)
    document_files = np.array_split(np.array(document_files), world_size)[rank].tolist()

    setup_output_dirs(args.save_dir)

    print(f"\nTotal files to process (rank {rank}/{world_size}): {len(document_files)}")
    for file_path in document_files:
        print(f"\nProcessing {file_path}")
        try:
            process_document(
                document_path=file_path,
                model=model,
                save_dir=args.save_dir,
                max_batch_size=args.max_batch_size,
            )
            print(f"Done. Results saved under {args.save_dir}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    log_path = os.path.join(args.save_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, f"success_{rank}.log"), "w") as f:
        f.write("1")

    if rank == 0:
        while len(glob.glob(os.path.join(log_path, "success*.log"))) != world_size:
            print("waiting!!!!!!!!!!!!!!!")
        print("finished!!!!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
