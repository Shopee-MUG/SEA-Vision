# SEA-Vision TEC-VQA (QA) 评测

**SEA-Vision: A Multilingual Benchmark for Comprehensive Document and Scene Text Understanding in Southeast Asia** 中 TEC-VQA（QA）子任务的推理与评测流程。

> 大体量图片文件 `images_11langs.tar.gz` 将上传到 Hugging Face，需要额外下载（不会默认随代码仓库分发）。
>
> 下文所有命令默认在 `TEC-VQA/` 目录下执行（先 `cd TEC-VQA`）。

---

## 项目概览 (Overview)

- 任务：文本中心视觉问答（TEC-VQA / QA）
- 语言覆盖：11 种东南亚常见语言（EN / ZH / VI / TH / FIL / MS / ID / LO / KM / MY / PT）
- 推理方式：
  - 本地 vLLM：Qwen / InternVL / Ovis / MiniCPM-V-4_5 等
  - API：OpenAI / Gemini 等
- 评测： `acc.py`

---

## 快速开始 (Quick Start)

### 1) 环境准备

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

```bash
pip install vllm openai google-generativeai tqdm
```

### 2) 数据准备

请确保 `data/` 下至少包含：

- `all_qa_data.jsonl`
- `images/`（解压后的 11 语言图片目录）

### 2.1 从 Hugging Face 下载（推荐）

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p data

huggingface-cli download xingranzhao/SEA-Vision images_11langs.tar.gz \
  --repo-type dataset \
  --local-dir data

tar -xzf data/images_11langs.tar.gz -C data
```

> `all_qa_data.jsonl` 已随 GitHub 代码仓库一同分发，位于 `TEC-VQA/data/all_qa_data.jsonl`，无需单独下载。

### 2.2 目录校验

解压完成后，目录应类似：

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

## 目录结构 (Directory Layout)

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

## 评测脚本使用 (Evaluation Scripts)

所有命令默认从 `TEC-VQA/qa_eval` 执行。

### 1) vLLM（Qwen / InternVL / Ovis 统一脚本）

```bash
cd qa_eval
python vllm_qwen_intern_ovis_batchQA.py \
  --model_dir /path/to/your/model \
  --input_jsonl ../data/all_qa_data.jsonl \
  --image_base_dir ../data/images \
  --output_jsonl ../data/qa_results/<model_name>.jsonl
```

InternVL / Ovis 可额外指定：

```bash
--processor_dir /path/to/processor
```

### 2) vLLM（MiniCPM-V-4_5）

```bash
cd qa_eval
python vllm_minicpmv4_5_batchQA.py \
  --model_dir /path/to/your/model \
  --input_jsonl ../data/all_qa_data.jsonl \
  --image_base_dir ../data/images \
  --output_jsonl ../data/qa_results/<model_name>.jsonl
```

### 3) OpenAI API（官方接口）

```bash
export OPENAI_API_KEY=""
cd qa_eval
python api_gpt.py \
  --model gpt-4o \
  --input_jsonl ../data/all_qa_data.jsonl \
  --image_base_dir ../data/images \
  --output_jsonl ../data/qa_results/gpt-4o.jsonl
```

### 4) Gemini API（官方接口）

```bash
export GEMINI_API_KEY=""
cd qa_eval
python api_gemini.py \
  --model gemini-2.5-pro \
  --input_jsonl ../data/all_qa_data.jsonl \
  --image_base_dir ../data/images \
  --output_jsonl ../data/qa_results/gemini-2.5-pro.jsonl
```

### 5) 准确率统计

```bash
cd qa_eval
python acc.py ../data/qa_results --mode md
```

如果结果 jsonl 中没有 `最终答案` 字段，`acc.py` 会回退读取：

```text
../data/all_qa_data.jsonl
```

关闭回退模式：

```bash
python acc.py ../data/qa_results --mode md --no_data_fallback
```

---

## 输出格式建议 (Output Format)

为保证 `acc.py` 统计稳定，建议模型输出 jsonl 至少包含样本标识与可评测答案字段。
若脚本产物中包含 `最终答案` 字段，会被优先用于计算准确率。

示例（字段名以实际脚本解析逻辑为准）：

```json
{"id":"sample_0001","question":"...","model_output":"...","最终答案":"..."}
```

---

## 常见问题 (FAQ)

### Q1: 仓库里没有完整图片数据？

图片压缩包体积较大，采用外部托管。请从 Hugging Face 额外下载 `images_11langs.tar.gz` 并解压到 `data/`。

### Q2: 报错提示找不到图片文件？

请检查：

- `--image_base_dir` 是否指向 `../data/images`
- `all_qa_data.jsonl` 中图片相对路径是否与解压目录一致
- 中文路径在当前系统 locale 下是否可正常读取

### Q3: API 调用速度慢或限流怎么办？

建议增加请求重试、并发控制与断点续跑机制；必要时先在小子集上验证脚本配置。

---

## TODO / 待完善

- [ ] 补充 Hugging Face 数据集的正式下载链接
- [ ] 补充统一 `requirements.txt`
- [ ] 补充更多基线模型结果与复现实验配置
- [ ] 增加评测结果可视化脚本

---

## Citation

如果本仓库对你的研究有帮助，请引用：

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

许可证将在正式开源版本中补充。
在此之前，请仅将代码与数据用于学术研究用途，并遵守各模型/API/数据源的使用条款。

---

## Contact

如有问题，欢迎提交 Issue 或联系仓库维护者。
