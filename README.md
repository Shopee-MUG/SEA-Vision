# SEA-Vision

**SEA-Vision: A Multilingual Benchmark for Comprehensive Document and Scene Text Understanding in Southeast Asia.**

SEA-Vision bundles two complementary benchmarks for evaluating multilingual visual document understanding across 11 Southeast-Asian languages (EN / ZH / VI / TH / FIL / MS / ID / LO / KM / MY / PT):

| Sub-benchmark   | Task                                                                                         | Where it lives                                |
| --------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------- |
| **SEA-DocBench** | End-to-end document parsing — text blocks, display formulas, tables, reading order.         | [`SEA-DocBench/`](./SEA-DocBench/README.md)   |
| **TEC-VQA**      | Text-centric visual question answering on natural-scene and document images.                | [`TEC-VQA/`](./TEC-VQA/README.md)             |

Each sub-benchmark has its own README with the full task description, data format, and command reference. This top-level README explains how the suite is laid out and gives a one-screen view of the end-to-end workflow.

---

## Repository layout

```
SeaVision/
├── README.md                          (this file)
├── SEA-DocBench/                      Document parsing benchmark + Dolphin reference inference
│   ├── README.md
│   ├── pdf_validation.py              Evaluation entrypoint
│   ├── configs/end2end_dolphin.yaml   Example end-to-end evaluation config
│   ├── tools/model_infer/             Reference inference scripts (Dolphin)
│   ├── dataset/ task/ metrics/ ...    Evaluation framework
│   └── data/                          (user-provided GT JSON + images, git-ignored)
│
├── TEC-VQA/                           Text-centric VQA benchmark
│   ├── README.md
│   ├── qa_eval/                       Inference + accuracy scripts (vLLM / API / acc.py)
│   └── data/                          (user-provided QA jsonl + images, git-ignored)
│
└── SEA-DocBench-images.tar.gz         Released image archive for SEA-DocBench (≈14 GB, distributed separately)
```

Both `SEA-DocBench/data/` and `TEC-VQA/data/` are kept as empty placeholders in version control. The actual datasets are distributed externally (Hugging Face) — see the per-task READMEs for download commands.

---

## Common installation

A single Python ≥ 3.10 environment can host both benchmarks. Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# SEA-DocBench evaluation framework
pip install -r SEA-DocBench/requirements.txt

# TEC-VQA inference (any subset you need)
pip install vllm openai google-generativeai tqdm
```

PyTorch / CUDA wheels are not pinned in either `requirements.txt` — install the build that matches your driver, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Quickstart

### 1. SEA-DocBench (document parsing)

```bash
cd SEA-DocBench

# (a) Place data — see SEA-DocBench/README.md for the GT JSON schema
#     data/ground_truth.json
#     data/images/*.jpg

# (b) Run reference inference (Dolphin); weights auto-download from HF on first use
python tools/model_infer/Dolphin_img2md.py \
    --input-dir   ./data/images \
    --save-dir    ./outputs/dolphin \
    --model-id    ByteDance/Dolphin-1.5 \
    --max-batch-size 16

# (c) Evaluate
python pdf_validation.py --config configs/end2end_dolphin.yaml
```

Per-element metrics (Edit_dist / TEDS / CDM_plain / ...) are written to `SEA-DocBench/metrics/result/`. Full instructions: [`SEA-DocBench/README.md`](./SEA-DocBench/README.md).

### 2. TEC-VQA (text-centric VQA)

```bash
cd TEC-VQA

# (a) Place data — see TEC-VQA/README.md for the QA jsonl schema
#     data/all_qa_data.jsonl
#     data/images/<lang>/...

# (b) Run inference (vLLM example with Qwen / InternVL / Ovis)
cd qa_eval
python vllm_qwen_intern_ovis_batchQA.py \
  --model_dir /path/to/your/model \
  --input_jsonl ../data/all_qa_data.jsonl \
  --image_base_dir ../data/images \
  --output_jsonl ../data/qa_results/<model_name>.jsonl

# (c) Evaluate accuracy
python acc.py ../data/qa_results --mode md
```

API-based inference (OpenAI, Gemini) and MiniCPM-V-4_5 each have their own scripts under `TEC-VQA/qa_eval/`. Full instructions: [`TEC-VQA/README.md`](./TEC-VQA/README.md).

---

## Data distribution

The actual benchmark data is hosted on Hugging Face at **[`xingranzhao/SEA-Vision`](https://huggingface.co/datasets/xingranzhao/SEA-Vision)**:

| Artefact                          | Sub-benchmark   | How to obtain                                                                                       |
| --------------------------------- | --------------- | --------------------------------------------------------------------------------------------------- |
| `SEA-DocBench-images.tar.gz` (≈14 GB, 15,234 images) | SEA-DocBench    | `huggingface-cli download xingranzhao/SEA-Vision SEA-DocBench-images.tar.gz --repo-type dataset --local-dir .` |
| Ground-truth JSON                 | SEA-DocBench    | Not yet released on HF; see [`SEA-DocBench/README.md`](./SEA-DocBench/README.md) for the schema and how to bring your own. |
| `all_qa_data.jsonl` (QA pairs)    | TEC-VQA         | Ships with this Git repo at `TEC-VQA/data/all_qa_data.jsonl` — no separate download needed.         |
| `images_11langs.tar.gz` (≈1.9 GB) | TEC-VQA         | `huggingface-cli download xingranzhao/SEA-Vision images_11langs.tar.gz --repo-type dataset --local-dir TEC-VQA/data` |

Concrete download commands and dataset schemas live in each sub-benchmark's README.

---

## Citation

```bibtex
@inproceedings{yue2026seavision,
  title={SEA-Vision: A Multilingual Benchmark for Comprehensive Document and Scene Text Understanding in Southeast Asia},
  author={Yue, Pengfei and Zhao, Xingran and Chen, Juntao and Hou, Peng and Longchao, Wang and Lin, Jianghang and Zhang, Shengchuan and Zeng, Anxiang and Cao, Liujuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## License & acknowledgements

- `SEA-DocBench/` is released under the Apache License 2.0 (see `SEA-DocBench/LICENSE`). It builds on the evaluation framework from [OmniDocBench](https://github.com/opendatalab/OmniDocBench), and the reference Dolphin inference helpers (vendored under `SEA-DocBench/tools/model_infer/dolphin_utils/`) are MIT-licensed by ByteDance ([github.com/bytedance/Dolphin](https://github.com/bytedance/Dolphin)). Model weights: [ByteDance/Dolphin-1.5](https://huggingface.co/ByteDance/Dolphin-1.5).
- `TEC-VQA/` license will be finalised at the official open-source release. Until then, please use the code and data for academic research only and comply with each upstream model / API / data-source's terms of use.

---

## Contact

Issues and pull requests are welcome on the repository. For questions about either sub-benchmark, please file an issue or contact the repository maintainers.
