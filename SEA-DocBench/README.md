# SEA-DocBench

End-to-end document parsing benchmark for South-East-Asian languages and beyond. The pipeline runs in two steps:

1. **Inference** — feed page images (or PDFs) into a document-parsing model and write a Markdown file per sample.
2. **Evaluation** — match those Markdown predictions against a ground-truth JSON and compute per-element metrics (text block, display formula, table, reading order).

This repository ships the evaluation framework plus one reference inference script (Dolphin). Plugging in a new model only requires a script that produces Markdown files in the agreed layout.

---

## Repository layout

```
SEA-DocBench/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── pdf_validation.py             # Evaluation entrypoint
├── configs/
│   └── end2end_dolphin.yaml      # Example end-to-end evaluation config
├── dataset/                      # Dataset loaders (registered via @DATASET_REGISTRY)
├── task/                         # Task runners (end2end / detection / recognition)
├── metrics/                      # Metric implementations: Edit_dist, BLEU, METEOR, TEDS, CDM
├── registry/                     # Decorator-based registry shared by dataset/task/metric
├── utils/                        # Matching, OCR helpers, table helpers
├── tools/
│   └── model_infer/
│       ├── Dolphin_img2md.py     # Reference inference script (Dolphin)
│       └── dolphin_utils/        # Vendored layout-parsing helpers (MIT, from upstream Dolphin)
├── data/                         # ← you provide: ground_truth.json + images/
└── outputs/                      # ← created by inference; one subdir per model
```

`data/` and `outputs/` are tracked as empty placeholders. Anything inside them is git-ignored.

---

## Installation

```bash
git clone <repo-url> SEA-DocBench
cd SEA-DocBench

# Python deps for evaluation framework
pip install -r requirements.txt
```

PyTorch and CUDA wheels are intentionally **not** pinned in `requirements.txt` — install the build that matches your driver, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

To run the reference Dolphin inference script you additionally need:

- `transformers`, `Pillow`, `opencv-python`, `pymupdf` (already in `requirements.txt`)
- The **Dolphin-1.5** model weights from HuggingFace. They are downloaded automatically on first run, or you can pre-fetch them:

  ```bash
  # Either let transformers cache them on first --model-id reference, or:
  pip install -U "huggingface_hub[cli]"
  huggingface-cli download ByteDance/Dolphin-1.5
  ```

  All layout-parsing / decoding helpers from the upstream Dolphin project are vendored under `tools/model_infer/dolphin_utils/`, so **you do not need to clone the Dolphin GitHub repo separately**.

---

## Data preparation

SEA-DocBench evaluation needs two artefacts:

| File / dir              | Where the example config expects it     | What it is                                                                                |
| ----------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------- |
| Image set               | `./data/images/`                        | One image (or PDF) per page. Filenames must match `page_info.image_path` in the GT JSON. |
| Ground-truth JSON       | `./data/ground_truth.json`              | Per-page layout / text / table / formula annotations (schema below).                      |

You can either grab the **official SEA-DocBench dataset** below, or bring your own data and point the yaml at it.

### Option A — Official SEA-DocBench dataset

The released dataset contains 15,234 document page images plus the matching ground-truth JSON.

```bash
# 1. Download from Hugging Face.
pip install -U "huggingface_hub[cli]"
huggingface-cli download xingranzhao/SEA-Vision SEA-DocBench-images.tar.gz \
    --repo-type dataset \
    --local-dir .

# 2. Extract into ./data/. The archive already contains a top-level `images/` directory,
#    so this places the images at ./data/images/<filename>.jpg.
tar xzf SEA-DocBench-images.tar.gz -C ./data/

# 3. Ground-truth JSON: not yet released on Hugging Face. Place it at
#    ./data/ground_truth.json once you obtain it (see the schema below).
```

After extraction the layout should be:

```
data/
├── ground_truth.json
└── images/
    ├── eng_Latn**34_38_363_2.jpg
    ├── ind_Latn**12_5_88_1.jpg
    └── ...   (15,234 files in total)
```

Sanity check (counts must agree):

```bash
python -c "import json; print(len(json.load(open('data/ground_truth.json'))))"
ls data/images/ | wc -l
```

### Option B — Bring your own data

Place images under `./data/images/` (or anywhere — just update `configs/end2end_dolphin.yaml`). Supported extensions: `.jpg .jpeg .png .pdf`. The image filenames must match the `page_info.image_path` field of your GT JSON.

### Ground-truth JSON schema

A list of page objects:

```json
[
  {
    "page_info": {
      "image_path": "page_001.jpg",
      "page_attribute": { "language": "english", ... }
    },
    "layout_dets": [
      {
        "category_type": "text_block",          // text_block | title | display_formula | table | figure | ...
        "poly": [x1,y1, x2,y1, x2,y2, x1,y2],    // 8-point polygon, in pixel coords
        "order": 0,                              // reading order index
        "anno_id": 12345,
        "text": "<ground-truth string>",         // markdown / LaTeX / HTML depending on category
        "ignore": false,
        "attribute": {
          "text_language": "text_english",
          "text_background": "white",
          "text_rotate": "normal"
        },
        "line_with_spans": [
          { "category_type": "text_span",
            "poly": [...],
            "text": "..." }
        ],
        "merge_list": []                         // optional, for merged blocks
      },
      ...
    ]
  }
]
```

Only the fields you actually evaluate need to be populated. The matchers (`utils/match_quick.py`) use `category_type`, `poly`, `text`, `order` and the `attribute` block.

---

## Step 1 — Inference (Dolphin example)

Generate one Markdown file per input image:

```bash
python tools/model_infer/Dolphin_img2md.py \
    --input-dir   ./data/images \
    --save-dir    ./outputs/dolphin \
    --model-id    ByteDance/Dolphin-1.5 \
    --max-batch-size 16
```

The first run will download the [Dolphin-1.5 weights](https://huggingface.co/ByteDance/Dolphin-1.5) from HuggingFace into your `~/.cache/huggingface/` directory.

| Flag                   | Description                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| `--input-dir`          | Directory of `.jpg/.png/.pdf` documents to parse.                                            |
| `--save-dir`           | Where Markdown predictions and figures are written. **Must equal `prediction.data_path` in the yaml.** |
| `--model-id`           | HF id or local path to a Dolphin checkpoint. Default: `ByteDance/Dolphin-1.5`.                |
| `--max-batch-size`     | Cap on per-batch element decoding. Default: `16`.                                            |
| `--seed`               | Seed for the document shuffling used in multi-rank inference. Default: `42`.                 |

Multi-process / multi-GPU sharding is honoured via standard `RANK` / `WORLD_SIZE` env vars:

```bash
WORLD_SIZE=8 RANK=0 python tools/model_infer/Dolphin_img2md.py ...
WORLD_SIZE=8 RANK=1 python tools/model_infer/Dolphin_img2md.py ...
# ...
```

After this step `./outputs/dolphin/` contains one `.md` per input plus a `figures/` subdir for cropped figures.

---

## Step 2 — Evaluation

Wire the GT and prediction paths into a yaml, then run:

```bash
python pdf_validation.py --config configs/end2end_dolphin.yaml
```

The shipped config (`configs/end2end_dolphin.yaml`) points at `./data/ground_truth.json` and `./outputs/dolphin/`. Edit those two fields if your layout differs:

```yaml
end2end_eval:
  dataset:
    ground_truth:
      data_path: ./data/ground_truth.json   # ← your GT
    prediction:
      data_path: ./outputs/dolphin          # ← --save-dir from Step 1
    match_method: quick_match               # quick_match | full_match
  metrics:
    text_block:       { metric: [Edit_dist] }
    display_formula:  { metric: [Edit_dist, CDM_plain] }
    table:            { metric: [TEDS, Edit_dist] }
    reading_order:    { metric: [Edit_dist] }
```

Per-sample and aggregate metric outputs are written under `metrics/result/` and printed to stdout.

### Metric reference

| Sub-task          | Supported metrics                                                |
| ----------------- | ---------------------------------------------------------------- |
| `text_block`      | `Edit_dist`, `BLEU`, `METEOR`                                    |
| `display_formula` | `Edit_dist`, `CDM_plain` (and `CDM` if a CDM environment is set up) |
| `table`           | `TEDS`, `Edit_dist`                                              |
| `reading_order`   | `Edit_dist`                                                      |

---

## Adding a new model

1. Drop a new script under `tools/model_infer/<YourModel>_img2md.py` that:
   - reads images from a `--input-dir`
   - writes one `<image_basename>.md` per input into a `--save-dir`
2. Copy `configs/end2end_dolphin.yaml` to `configs/end2end_<yourmodel>.yaml`.
3. Set `prediction.data_path` to your model's save dir.
4. Run `python pdf_validation.py --config configs/end2end_<yourmodel>.yaml`.

No changes to `dataset/`, `task/`, or `metrics/` are required — the matchers operate directly on the rendered Markdown.

---

## License & acknowledgements

Released under the Apache License 2.0 (see `LICENSE`). SEA-DocBench builds on the evaluation framework from [OmniDocBench](https://github.com/opendatalab/OmniDocBench). The reference inference script and the helpers vendored under `tools/model_infer/dolphin_utils/` are derived from [Dolphin](https://github.com/bytedance/Dolphin) (MIT-licensed by ByteDance); the model weights themselves come from [ByteDance/Dolphin-1.5](https://huggingface.co/ByteDance/Dolphin-1.5).
