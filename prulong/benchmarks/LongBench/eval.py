import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def compute_dataset_scores(dataset, baseline_preds, sparse_preds, sparsities, all_classes_list):
    if len(baseline_preds) == 0:
        return {"score": 0.0, "pass_rate": 0.0, "avg_f1": 0.0}

    metric_fn = dataset2metric.get(dataset)
    if metric_fn is None:
        return {"score": 0.0, "pass_rate": 0.0, "avg_f1": 0.0}

    total_score = 0.0
    pass_cases = 0
    f1_scores = []
    for idx, (baseline_pred, sparse_pred, sparsity) in enumerate(zip(baseline_preds, sparse_preds, sparsities)):
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            baseline_pred = baseline_pred.lstrip('\n').split('\n')[0]
            sparse_pred = sparse_pred.lstrip('\n').split('\n')[0]
        all_classes = all_classes_list[idx] if idx < len(all_classes_list) else None
        try:
            f1_value = metric_fn(sparse_pred, baseline_pred, all_classes=all_classes)
        except TypeError:
            f1_value = metric_fn(sparse_pred, baseline_pred)
        f1_scores.append(float(f1_value))
        if f1_value >= 0.99:
            total_score += float(sparsity)
            pass_cases += 1
    count = len(baseline_preds)
    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {
        "score": round(total_score / count, 4),
        "pass_rate": round(pass_cases / count, 4),
        "avg_f1": round(avg_f1, 4),
    }

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    if args.e:
        path = f"pred_e/{args.model}/"
    else:
        path = f"pred/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        baseline_preds, sparse_preds, sparsities, all_classes_list = [], [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                baseline = data.get("pred_full", data.get("pred", ""))
                sparse = data.get("pred_sparse", baseline)
                try:
                    sparsity = float(data.get("sparsity", 0.0))
                except (TypeError, ValueError):
                    sparsity = 0.0
                baseline_preds.append(baseline)
                sparse_preds.append(sparse)
                sparsities.append(sparsity)
                all_classes_list.append(data.get("all_classes"))
        score = compute_dataset_scores(dataset, baseline_preds, sparse_preds, sparsities, all_classes_list)
        scores[dataset] = score
    if args.e:
        out_path = f"pred_e/{args.model}/result.json"
    else:
        out_path = f"pred/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
