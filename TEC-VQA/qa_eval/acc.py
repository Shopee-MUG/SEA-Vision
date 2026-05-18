import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def normalize(text, mode=""):
    text = str(text) if not isinstance(text, (str, list)) else text
    if isinstance(text, list):
        text = ", ".join(text)
    text = text.strip().lower().replace(".", "")
    if "md" in mode:
        for md_symbol in ["**", "*", "__", "_", "`", "```"]:
            text = text.replace(md_symbol, "")
    return text


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def evaluate_exact_match(pred, gt, mode=""):
    return 1.0 if normalize(gt, mode) in normalize(pred, mode) else 0.0


def evaluate_edit_distance(pred, gt, mode=""):
    p = normalize(pred, mode)
    g = normalize(gt, mode)
    if not p and not g:
        return 1.0
    max_len = max(len(p), len(g))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(p, g)
    return 1.0 - dist / max_len


def build_key(item):
    lang = str(item.get("语言", ""))
    img_id = str(item.get("图片编号", ""))
    idx = str(item.get("数据索引", ""))
    return lang, img_id, idx


def load_gt_map(data_jsonl_path):
    gt_map = {}
    with open(data_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = build_key(item)
            gt_map[key] = item.get("最终答案", "")
    return gt_map


def is_valid_jsonl(jsonl_path):
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if "predicted_answer" in item:
                    return True
        return False
    except Exception:
        return False


def evaluate_file(jsonl_path, mode="", gt_map=None):
    use_ed = "ed" in mode
    score_fn = evaluate_edit_distance if use_ed else evaluate_exact_match
    metric_name = "平均编辑距离相似度" if use_ed else "准确率"

    model_name = os.path.basename(jsonl_path).split(".")[0]
    lang_stats = defaultdict(lambda: {"score": 0.0, "total": 0})
    total_score = 0.0
    total_count = 0
    missing_gt = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lang = item.get("语言", "未知")
            pred = item.get("predicted_answer", "")
            gt = item.get("最终答案", "")

            if (not gt) and gt_map is not None:
                gt = gt_map.get(build_key(item), "")

            if gt == "":
                missing_gt += 1
                continue

            score = score_fn(pred, gt, mode)
            lang_stats[lang]["score"] += score
            lang_stats[lang]["total"] += 1
            total_score += score
            total_count += 1

    print(f"\nEvaluating: {model_name}  [{metric_name}]")
    for lang, stats in lang_stats.items():
        avg = stats["score"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  {lang}: {avg:.4f} ({stats['total']} 条)")
    overall = total_score / total_count if total_count > 0 else 0.0
    print(f"  整体{metric_name}: {overall:.4f} ({total_count} 条)")
    if missing_gt > 0:
        print(f"  跳过: {missing_gt} 条（缺少最终答案）")


def evaluate_path(input_path, mode="", gt_map=None):
    if os.path.isdir(input_path):
        jsonl_files = sorted(
            [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.endswith(".jsonl")
            ]
        )
        if not jsonl_files:
            print(f"文件夹 {input_path} 下未找到任何 .jsonl 文件")
            return
        valid_files = [p for p in jsonl_files if is_valid_jsonl(p)]
        if not valid_files:
            print(f"文件夹 {input_path} 下未找到有效结果文件（含 predicted_answer）")
            return
        print(f"在 {input_path} 下找到 {len(valid_files)} 个有效 jsonl 文件：")
        for p in valid_files:
            evaluate_file(p, mode=mode, gt_map=gt_map)
    else:
        evaluate_file(input_path, mode=mode, gt_map=gt_map)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default=str(data_dir / "qa_results"),
        help="Result jsonl file or directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="md",
        help="Mode flags, e.g. md, ed, md+ed",
    )
    parser.add_argument(
        "--data_jsonl",
        type=str,
        default=str(data_dir / "all_qa_data.jsonl"),
        help="Fallback GT data file",
    )
    parser.add_argument(
        "--no_data_fallback",
        action="store_true",
        help="Disable GT fallback from data_jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    gt_map = None
    if not args.no_data_fallback and os.path.exists(args.data_jsonl):
        gt_map = load_gt_map(args.data_jsonl)
        print(f"Loaded GT map from {args.data_jsonl}: {len(gt_map)} entries")
    elif not args.no_data_fallback:
        print(f"Warning: data_jsonl not found: {args.data_jsonl}, fallback disabled.")

    evaluate_path(args.input_path, mode=args.mode, gt_map=gt_map)
