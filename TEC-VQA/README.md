# SEA-Vision TEC-VQA

Inference and evaluation pipeline for the **TEC-VQA** sub-task of:

> **SEA-Vision: A Multilingual Benchmark for Comprehensive Document and Scene Text Understanding in Southeast Asia** (CVPR 2026)
> [[Paper]](https://arxiv.org/abs/2603.15409) | [[Project Page]](https://swagger-coder.github.io/sea-vision-page/) | [[Dataset]](https://huggingface.co/datasets/xingranzhao/SEA-Vision)

> The large image archive `images_11langs.tar.gz` is hosted on Hugging Face and must be downloaded separately (it is not distributed with the code repository).
>
> All commands below assume you are inside the `TEC-VQA/` directory (`cd TEC-VQA` first).

---

## Overview

- **Task:** Text-Centric Visual Question Answering (TEC-VQA)
- **Languages:** 11 Southeast Asian languages — EN / ZH / VI / TH / FIL / MS / ID / LO / KM / MY / PT
- **QA pairs:** 7,496 pairs across 1,839 images, annotated with five capability labels: text recognition, numerical calculation, comparative analysis, logical reasoning, and spatial understanding
- **Inference backends:**
  - Local vLLM: Qwen / InternVL / Ovis / MiniCPM-V-4_5, etc.
  - API: OpenAI / Gemini, etc.
- **Evaluation:** `acc.py`

---

## Quick Start

### 1. Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install vllm openai google-generativeai tqdm
```

### 2. Data Setup

Ensure `data/` contains at least:

- `all_qa_data.jsonl`
- `images/` (the extracted 11-language image directory)

#### 2.1 Download from Hugging Face (recommended)

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p data

huggingface-cli download xingranzhao/SEA-Vision images_11langs.tar.gz \
  --repo-type dataset \
  --local-dir data

tar -xzf data/images_11langs.tar.gz -C data
```

> `all_qa_data.jsonl` ships with this Git repository at `TEC-VQA/data/all_qa_data.jsonl` — no separate download is needed.

#### 2.2 Verify Directory Structure

After extraction the layout should look like:

```text
data/
├── all_qa_data.jsonl
├── images/
│   ├── 中文
│   ├── 印尼语
│   ├── 泰语
│   ├── 缅甸语
│   ├── 老挝语
│   ├── 英语
│   ├── 菲律宾语
│   ├── 葡萄牙语
│   ├── 越南语
│   ├── 马来语
│   └── 高棉语
└── qa_results/
```

---

## Directory Layout

```text
TEC-VQA/
├── data/
│   ├── all_qa_data.jsonl
│   ├── images/
│   │   ├── 中文
│   │   ├── 印尼语
│   │   ├── 泰语
│   │   ├── 缅甸语
│   │   ├── 老挝语
│   │   ├── 英语
│   │   ├── 菲律宾语
│   │   ├── 葡萄牙语
│   │   ├── 越南语
│   │   ├── 马来语
│   │   └── 高棉语
│   ├── images_11langs.tar.gz
│   └── qa_results/
│       └── Qwen3-VL-32B-Instruct.jsonl
└── qa_eval/
    ├── vllm_qwen_intern_ovis_batchQA.py
    ├── vllm_minicpmv4_5_batchQA.py
    ├── api_gpt.py
    ├── api_gemini.py
    └── acc.py
```

---

## Running Inference

All commands below are executed from inside `TEC-VQA/qa_eval/`.

### 1. vLLM — Qwen / InternVL / Ovis (unified script)

```bash
cd qa_eval
python vllm_qwen_intern_ovis_batchQA.py \
  --model_dir /path/to/your/model \
  --input_jsonl ../data/all_qa_data.jsonl \
  --image_base_dir ../data/images \
  --output_jsonl ../data/qa_results/<model_name>.jsonl
```

For InternVL / Ovis, you can additionally specify:

```bash
  --processor_dir /path/to/processor
```

### 2. vLLM — MiniCPM-V-4_5

```bash
cd qa_eval
python vllm_minicpmv4_5_batchQA.py \
  --model_dir /path/to/your/model \
  --input_jsonl ../data/all_qa_data.jsonl \
  --image_base_dir ../data/images \
  --output_jsonl ../data/qa_results/<model_name>.jsonl
```

### 3. OpenAI API

```bash
export OPENAI_API_KEY=""
cd qa_eval
python api_gpt.py \
  --model gpt-4o \
  --input_jsonl ../data/all_qa_data.jsonl \
  --image_base_dir ../data/images \
  --output_jsonl ../data/qa_results/gpt-4o.jsonl
```

### 4. Gemini API

```bash
export GEMINI_API_KEY=""
cd qa_eval
python api_gemini.py \
  --model gemini-2.5-pro \
  --input_jsonl ../data/all_qa_data.jsonl \
  --image_base_dir ../data/images \
  --output_jsonl ../data/qa_results/gemini-2.5-pro.jsonl
```

### 5. Accuracy Evaluation

```bash
cd qa_eval
python acc.py ../data/qa_results --mode md
```

If a result JSONL does not contain a `最终答案` field, `acc.py` falls back to reading from:

```text
../data/all_qa_data.jsonl
```

To disable the fallback:

```bash
python acc.py ../data/qa_results --mode md --no_data_fallback
```

---

## Output Format

To ensure stable scoring by `acc.py`, each output JSONL should contain at minimum a sample identifier and an evaluable answer field. If the output includes a `最终答案` field, it is used preferentially for accuracy calculation.

Example (exact field names follow each script's parsing logic):

```json
{"id": "sample_0001", "question": "...", "model_output": "...", "最终答案": "..."}
```

---

## FAQ

**Q1: The repository doesn't contain the full image data.**  
The image archive is large and hosted externally. Download `images_11langs.tar.gz` from Hugging Face and extract it into `data/` as described above.

**Q2: Script reports missing image files.**  
Check that:
- `--image_base_dir` points to `../data/images`
- The relative image paths in `all_qa_data.jsonl` match the extraction directory
- Your system locale can read the Chinese directory names correctly

**Q3: API calls are slow or rate-limited.**  
Add retry logic, concurrency control, and checkpoint-resume support. Validate your script configuration on a small subset first.

---

## TODO

- [ ] Add a unified `requirements.txt`
- [ ] Add more baseline model results and reproduction configs
- [ ] Add an evaluation result visualization script

---

## Citation

If this repository is useful for your research, please cite:

```bibtex
@inproceedings{yue2026seavision,
  title={SEA-Vision: A Multilingual Benchmark for Comprehensive Document and Scene Text Understanding in Southeast Asia},
  author={Yue, Pengfei and Zhao, Xingran and Chen, Juntao and Hou, Peng and Longchao, Wang and Lin, Jianghang and Zhang, Shengchuan and Zeng, Anxiang and Cao, Liujuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## License

The license will be specified in the official open-source release. Until then, please use the code and data for academic research purposes only and comply with the terms of each upstream model / API / data source.

---

## Contact

Questions and issues are welcome — please file an issue on the repository or contact the maintainers directly.
