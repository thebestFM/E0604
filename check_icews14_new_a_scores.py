import argparse
import json
import os
import os.path as osp

import numpy as np
from scipy.sparse import load_npz


RUN = {
    "dataset": "ICEWS14",
    "name": "ICEWS14_new_a_from_logs_structure_0608",
    "source_log": "logs-structure-0608/ICEWS14_new_a.log",
    "logged": {
        "val_mrr_loose": 0.97954,
        "val_mrr_strict": 0.48202,
        "val_mrr_avg": 0.48219,
        "test_mrr_loose": 0.98505,
        "test_mrr_strict": 0.49351,
        "test_mrr_avg": 0.49362,
        "test_hr1_strict": 0.49274,
        "test_hr10_strict": 0.49418,
    },
    "path": "results_new_structure/ICEWS14/seed42/decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=1_gamma=0_impl=new_structure_v1_hb13931ad557f",
}


HIT_KS = (1, 3, 10)
CATEGORIES = ("nan", "pos_inf", "neg_inf", "gt_high", "lt_low", "abs_gt_high", "nonfinite")


def resolve_path(root, path):
    return path if osp.isabs(path) else osp.join(root, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def category_masks(values, high, low):
    finite = np.isfinite(values)
    return {
        "nan": np.isnan(values),
        "pos_inf": np.isposinf(values),
        "neg_inf": np.isneginf(values),
        "gt_high": finite & (values > high),
        "lt_low": finite & (values < low),
        "abs_gt_high": finite & (np.abs(values) > high),
        "nonfinite": ~finite,
    }


def empty_counts():
    return {k: {"queries": 0, "samples": 0, "examples": []} for k in CATEGORIES}


def add_examples(dst, examples, limit):
    if len(dst) < limit:
        dst.extend(examples[: limit - len(dst)])


def update_pos_counts(counts, pos, high, low, example_limit):
    flat = pos.reshape(pos.shape[0], -1)
    for cat, mask in category_masks(flat, high, low).items():
        row_has = np.any(mask, axis=1)
        counts[cat]["queries"] += int(np.sum(row_has))
        counts[cat]["samples"] += int(np.sum(mask))
        rows, cols = np.where(mask)
        examples = [
            {"row": int(r), "col": int(c), "value": float(flat[r, c])}
            for r, c in zip(rows[:example_limit], cols[:example_limit])
        ]
        add_examples(counts[cat]["examples"], examples, example_limit)


def metric_sums():
    sums = {"count": 0}
    for kind in ("loose", "strict", "avg"):
        sums[f"mrr_{kind}"] = 0.0
        for k in HIT_KS:
            sums[f"hit@{k}_{kind}"] = 0.0
    return sums


def add_rank(sums, loose, strict):
    avg = (loose + strict) * 0.5
    sums["count"] += 1
    for kind, rank in (("loose", loose), ("strict", strict), ("avg", avg)):
        sums[f"mrr_{kind}"] += 1.0 / float(rank)
        for k in HIT_KS:
            if rank <= k:
                sums[f"hit@{k}_{kind}"] += 1.0


def finalize(sums):
    count = max(int(sums["count"]), 1)
    out = {"count": int(sums["count"])}
    for kind in ("loose", "strict", "avg"):
        out[f"mrr_{kind}"] = float(sums[f"mrr_{kind}"] / count)
        for k in HIT_KS:
            out[f"hit@{k}_{kind}"] = float(sums[f"hit@{k}_{kind}"] / count)
    return out


def clip_values(values, high, low):
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=high, neginf=low), low, high)


def sample_quantiles(values, sample, limit):
    if len(sample) >= limit:
        return
    finite = values[np.isfinite(values)]
    if finite.size:
        take = min(limit - len(sample), int(finite.size))
        sample.extend(float(x) for x in finite[:take])


def summarize_quantiles(sample):
    if not sample:
        return {}
    arr = np.asarray(sample, dtype=np.float64)
    qs = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]
    vals = np.quantile(arr, qs)
    return {f"q{q:g}": float(v) for q, v in zip(qs, vals)}


def analyze_split(run_dir, split, high, low, example_limit, sample_limit):
    pos_path = osp.join(run_dir, f"{split}_pos.npy")
    neg_path = osp.join(run_dir, f"{split}_neg.npz")
    lens_path = osp.join(run_dir, f"{split}_valid_lens.npy")
    meta_path = osp.join(run_dir, f"{split}_meta.json")
    missing = [p for p in (pos_path, neg_path, lens_path, meta_path) if not osp.isfile(p)]
    if missing:
        return {"split": split, "missing": missing}

    pos = np.load(pos_path)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 1)
    neg = load_npz(neg_path).tocsr()
    valid_lens = np.load(lens_path).astype(np.int64)
    meta = load_json(meta_path)
    n_rows = int(pos.shape[0])
    max_negs = int(neg.shape[1])

    pos_counts = empty_counts()
    neg_counts = empty_counts()
    update_pos_counts(pos_counts, pos, high, low, example_limit)

    raw_sums = metric_sums()
    clip_sums = metric_sums()
    stats = {
        "finite_min": None,
        "finite_max": None,
        "stored_valid_neg_nnz": 0,
        "implicit_zero_neg_count": 0,
        "stored_invalid_neg_nnz": 0,
        "queries_all_candidates_zero": 0,
        "queries_pos_zero": 0,
        "queries_pos_tied_with_at_least_one_neg": 0,
        "queries_pos_tied_with_best": 0,
        "queries_loose_rank_1": 0,
        "queries_strict_rank_1": 0,
        "sum_ties_equal_pos": 0,
        "max_ties_equal_pos": 0,
    }
    quant_sample = []

    def update_range(values, include_zero=False):
        finite = values[np.isfinite(values)]
        if finite.size:
            mn = float(np.min(finite))
            mx = float(np.max(finite))
            stats["finite_min"] = mn if stats["finite_min"] is None else min(stats["finite_min"], mn)
            stats["finite_max"] = mx if stats["finite_max"] is None else max(stats["finite_max"], mx)
        if include_zero:
            stats["finite_min"] = 0.0 if stats["finite_min"] is None else min(stats["finite_min"], 0.0)
            stats["finite_max"] = 0.0 if stats["finite_max"] is None else max(stats["finite_max"], 0.0)

    update_range(pos.reshape(-1))
    sample_quantiles(pos.reshape(-1), quant_sample, sample_limit)

    indptr, indices, data = neg.indptr, neg.indices, neg.data
    for row in range(n_rows):
        lens = int(valid_lens[row])
        start, end = int(indptr[row]), int(indptr[row + 1])
        row_idx_all = indices[start:end]
        row_vals_all = data[start:end]
        valid_mask = row_idx_all < lens
        invalid_mask = ~valid_mask
        row_idx = row_idx_all[valid_mask]
        row_vals = row_vals_all[valid_mask]
        implicit_zero = max(lens - int(row_vals.size), 0)

        stats["stored_valid_neg_nnz"] += int(row_vals.size)
        stats["implicit_zero_neg_count"] += int(implicit_zero)
        stats["stored_invalid_neg_nnz"] += int(np.sum(invalid_mask))
        update_range(row_vals, include_zero=implicit_zero > 0)
        sample_quantiles(row_vals, quant_sample, sample_limit)

        for cat, mask in category_masks(row_vals, high, low).items():
            cnt = int(np.sum(mask))
            if cnt:
                neg_counts[cat]["queries"] += 1
                neg_counts[cat]["samples"] += cnt
                bad = np.flatnonzero(mask)
                examples = [
                    {"row": int(row), "col": int(row_idx[p]), "value": float(row_vals[p])}
                    for p in bad[:example_limit]
                ]
                add_examples(neg_counts[cat]["examples"], examples, example_limit)

        p = float(pos[row, 0])
        if p == 0.0:
            stats["queries_pos_zero"] += 1
        gt = int(np.sum(row_vals > p))
        ge = int(np.sum(row_vals >= p))
        eq = int(np.sum(row_vals == p))
        if 0.0 > p:
            gt += implicit_zero
        if 0.0 >= p:
            ge += implicit_zero
        if 0.0 == p:
            eq += implicit_zero
        loose = 1 + gt
        strict = 1 + ge
        add_rank(raw_sums, loose, strict)
        if loose == 1:
            stats["queries_loose_rank_1"] += 1
        if strict == 1:
            stats["queries_strict_rank_1"] += 1
        if eq > 0:
            stats["queries_pos_tied_with_at_least_one_neg"] += 1
        if loose == 1 and strict > 1:
            stats["queries_pos_tied_with_best"] += 1
        stats["sum_ties_equal_pos"] += int(eq)
        stats["max_ties_equal_pos"] = max(stats["max_ties_equal_pos"], int(eq))
        if p == 0.0 and row_vals.size == 0 and implicit_zero == lens:
            stats["queries_all_candidates_zero"] += 1

        p_clip = float(clip_values(np.asarray([p]), high, low)[0])
        row_clip = clip_values(row_vals.astype(np.float64, copy=False), high, low)
        gt_clip = int(np.sum(row_clip > p_clip))
        ge_clip = int(np.sum(row_clip >= p_clip))
        if 0.0 > p_clip:
            gt_clip += implicit_zero
        if 0.0 >= p_clip:
            ge_clip += implicit_zero
        add_rank(clip_sums, 1 + gt_clip, 1 + ge_clip)

    shape_issues = []
    if int(meta.get("num_rows", -1)) != n_rows:
        shape_issues.append(f"meta num_rows={meta.get('num_rows')} pos_rows={n_rows}")
    if neg.shape[0] != n_rows:
        shape_issues.append(f"neg_rows={neg.shape[0]} pos_rows={n_rows}")
    if valid_lens.shape[0] != n_rows:
        shape_issues.append(f"valid_lens_rows={valid_lens.shape[0]} pos_rows={n_rows}")
    if np.any(valid_lens < 0) or np.any(valid_lens > max_negs):
        shape_issues.append("valid_lens outside [0, max_negs]")

    return {
        "split": split,
        "missing": [],
        "meta": meta,
        "shape": {
            "pos_shape": list(pos.shape),
            "neg_shape": list(neg.shape),
            "valid_lens_min": int(np.min(valid_lens)) if valid_lens.size else None,
            "valid_lens_max": int(np.max(valid_lens)) if valid_lens.size else None,
            "shape_issues": shape_issues,
        },
        "score_range": {
            "finite_min": stats["finite_min"],
            "finite_max": stats["finite_max"],
            "sampled_finite_quantiles": summarize_quantiles(quant_sample),
            "sample_size": len(quant_sample),
        },
        "storage": {
            "total_valid_neg": int(np.sum(valid_lens)),
            "stored_valid_neg_nnz": int(stats["stored_valid_neg_nnz"]),
            "implicit_zero_neg_count": int(stats["implicit_zero_neg_count"]),
            "stored_invalid_neg_nnz": int(stats["stored_invalid_neg_nnz"]),
        },
        "pos_anomalies": pos_counts,
        "neg_anomalies": neg_counts,
        "tie_checks": {
            "queries_pos_zero": int(stats["queries_pos_zero"]),
            "queries_all_candidates_zero": int(stats["queries_all_candidates_zero"]),
            "queries_loose_rank_1": int(stats["queries_loose_rank_1"]),
            "queries_strict_rank_1": int(stats["queries_strict_rank_1"]),
            "queries_pos_tied_with_at_least_one_neg": int(stats["queries_pos_tied_with_at_least_one_neg"]),
            "queries_pos_tied_with_best": int(stats["queries_pos_tied_with_best"]),
            "avg_ties_equal_pos_per_query": float(stats["sum_ties_equal_pos"] / max(n_rows, 1)),
            "max_ties_equal_pos": int(stats["max_ties_equal_pos"]),
        },
        "metrics_raw": finalize(raw_sums),
        "metrics_clipped": finalize(clip_sums),
    }


def print_split(result, high, low):
    split = result["split"]
    if result.get("missing"):
        print(f"[{split}] MISSING")
        for p in result["missing"]:
            print(f"  {p}")
        return
    raw = result["metrics_raw"]
    clipped = result["metrics_clipped"]
    shape = result["shape"]
    print(
        f"[{split}] rows={shape['pos_shape'][0]} neg_cols={shape['neg_shape'][1]} "
        f"valid_lens={shape['valid_lens_min']}..{shape['valid_lens_max']}"
    )
    print(
        f"  raw strict:     mrr={raw['mrr_strict']:.5f} hr1={raw['hit@1_strict']:.5f} "
        f"hr10={raw['hit@10_strict']:.5f}"
    )
    print(
        f"  raw loose:      mrr={raw['mrr_loose']:.5f} hr1={raw['hit@1_loose']:.5f} "
        f"hr10={raw['hit@10_loose']:.5f}"
    )
    print(
        f"  clipped strict: mrr={clipped['mrr_strict']:.5f} hr1={clipped['hit@1_strict']:.5f} "
        f"hr10={clipped['hit@10_strict']:.5f} clip=[{low:g},{high:g}]"
    )
    print(f"  finite_range={result['score_range']['finite_min']}..{result['score_range']['finite_max']}")
    print(f"  sample_quantiles={result['score_range']['sampled_finite_quantiles']}")
    storage = result["storage"]
    print(
        f"  storage: stored_valid={storage['stored_valid_neg_nnz']} "
        f"implicit_zero={storage['implicit_zero_neg_count']} invalid_stored={storage['stored_invalid_neg_nnz']}"
    )
    for side in ("pos_anomalies", "neg_anomalies"):
        pieces = []
        for cat in CATEGORIES:
            item = result[side][cat]
            if item["samples"]:
                pieces.append(f"{cat}:q={item['queries']},n={item['samples']}")
        print(f"  {side}: " + (", ".join(pieces) if pieces else "none"))
    ties = result["tie_checks"]
    print(
        "  ties: "
        f"pos_zero_q={ties['queries_pos_zero']} all_zero_q={ties['queries_all_candidates_zero']} "
        f"loose_rank1_q={ties['queries_loose_rank_1']} strict_rank1_q={ties['queries_strict_rank_1']} "
        f"pos_tied_any_neg_q={ties['queries_pos_tied_with_at_least_one_neg']} "
        f"pos_tied_best_q={ties['queries_pos_tied_with_best']} "
        f"avg_equal_pos={ties['avg_ties_equal_pos_per_query']:.2f} max_equal_pos={ties['max_ties_equal_pos']}"
    )
    if shape["shape_issues"]:
        print(f"  shape_issues={shape['shape_issues']}")


def parse_args():
    parser = argparse.ArgumentParser("Check saved ICEWS14 A-only score store from logs-structure-0608/ICEWS14_new_a.log.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--high", type=float, default=10000.0)
    parser.add_argument("--low", type=float, default=-10000.0)
    parser.add_argument("--example_limit", type=int, default=5)
    parser.add_argument("--sample_limit", type=int, default=200000)
    parser.add_argument("--json_out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = resolve_path(args.root, RUN["path"])
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    print("=" * 100)
    print(f"{RUN['dataset']} {RUN['name']}")
    print(f"source_log={RUN['source_log']}")
    print(f"path={run_dir}")
    print(f"logged={RUN['logged']}")
    report = {"run": RUN, "resolved_path": run_dir, "thresholds": {"high": args.high, "low": args.low}, "splits": {}}
    for split in splits:
        result = analyze_split(run_dir, split, args.high, args.low, args.example_limit, args.sample_limit)
        report["splits"][split] = result
        print_split(result, args.high, args.low)
    if args.json_out:
        out_path = resolve_path(args.root, args.json_out)
        os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"saved JSON report -> {out_path}")


if __name__ == "__main__":
    main()
