import argparse
import json
import os
import os.path as osp

import numpy as np
from scipy.sparse import load_npz


RUNS = [
    {
        "dataset": "ICEWS14",
        "name": "icews14_c01_dsh0.95_tag_sum_a0.012_b0.92",
        "source_log": "logs-structure-0609/icews14_add_dmh_c01_dsh0.95_tag_sum_a0.012_b0.92.log",
        "logged_test_mrr": 0.58000,
        "logged_test_hr1": 0.53116,
        "logged_test_hr10": 0.67713,
        "path": "results_new_structure/ICEWS14/seed42/decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.95_gamma=0_impl=new_structure_h776ac7bb370c",
    },
    {
        "dataset": "ICEWS14",
        "name": "icews14_c01_dsh0.90_tag_sum_a0.012_b0.92",
        "source_log": "logs-structure-0609/icews14_add_dmh_c01_dsh0.90_tag_sum_a0.012_b0.92.log",
        "logged_test_mrr": 0.58000,
        "logged_test_hr1": 0.53116,
        "logged_test_hr10": 0.67709,
        "path": "results_new_structure/ICEWS14/seed42/decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.9_gamma=0_impl=new_structure__hc73e01e914fc",
    },
    {
        "dataset": "ICEWS14",
        "name": "icews14_c01_dsh0.80_tag_sum_a0.012_b0.92",
        "source_log": "logs-structure-0609/icews14_add_dmh_c01_dsh0.80_tag_sum_a0.012_b0.92.log",
        "logged_test_mrr": 0.57999,
        "logged_test_hr1": 0.53116,
        "logged_test_hr10": 0.67709,
        "path": "results_new_structure/ICEWS14/seed42/decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.8_gamma=0_impl=new_structure__hc8e543b6bbd8",
    },
    {
        "dataset": "GDELT",
        "name": "gdelt_c02_dsh0.95_tag_sum_a0.012_b0.95",
        "source_log": "logs-structure-0609/gdelt_add_dmh_c02_dsh0.95_tag_sum_a0.012_b0.95.log",
        "logged_test_mrr": 0.64161,
        "logged_test_hr1": 0.61265,
        "logged_test_hr10": 0.69555,
        "path": "results_new_structure/GDELT/seed42/decay_direct=0.1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.95_gamma=0_impl=new_structu_h1eb30931e794",
    },
    {
        "dataset": "GDELT",
        "name": "gdelt_c02_dsh0.90_tag_sum_a0.012_b0.95",
        "source_log": "logs-structure-0609/gdelt_add_dmh_c02_dsh0.90_tag_sum_a0.012_b0.95.log",
        "logged_test_mrr": 0.64160,
        "logged_test_hr1": 0.61264,
        "logged_test_hr10": 0.69556,
        "path": "results_new_structure/GDELT/seed42/decay_direct=0.1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.9_gamma=0_impl=new_structur_hb66aa724158b",
    },
    {
        "dataset": "GDELT",
        "name": "gdelt_c02_dsh0.80_tag_sum_a0.012_b0.95",
        "source_log": "logs-structure-0609/gdelt_add_dmh_c02_dsh0.80_tag_sum_a0.012_b0.95.log",
        "logged_test_mrr": 0.64160,
        "logged_test_hr1": 0.61263,
        "logged_test_hr10": 0.69556,
        "path": "results_new_structure/GDELT/seed42/decay_direct=0.1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.8_gamma=0_impl=new_structur_h1f5cf6230a55",
    },
    {
        "dataset": "tkgl-polecat",
        "name": "polecat_a04_dsh0.05_decay0.10",
        "source_log": "logs-structure-0609/polecat_add_dsh_a04_dsh0.05_decay0.10.log",
        "logged_test_mrr": 0.58845,
        "logged_test_hr1": 0.53911,
        "logged_test_hr10": 0.67842,
        "path": "results_new_structure/tkgl-polecat/seed42/close_update_backward=0_decay_direct=0.1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.05__h21804db1a695",
    },
    {
        "dataset": "tkgl-polecat",
        "name": "polecat_a04_dsh0.10_decay0.10",
        "source_log": "logs-structure-0609/polecat_add_dsh_a04_dsh0.10_decay0.10.log",
        "logged_test_mrr": 0.58811,
        "logged_test_hr1": 0.53892,
        "logged_test_hr10": 0.67769,
        "path": "results_new_structure/tkgl-polecat/seed42/close_update_backward=0_decay_direct=0.1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.1_g_h4a62665bd1dc",
    },
    {
        "dataset": "tkgl-polecat",
        "name": "polecat_a04_dsh0.20_decay0.10",
        "source_log": "logs-structure-0609/polecat_add_dsh_a04_dsh0.20_decay0.10.log",
        "logged_test_mrr": 0.58736,
        "logged_test_hr1": 0.53858,
        "logged_test_hr10": 0.67607,
        "path": "results_new_structure/tkgl-polecat/seed42/close_update_backward=0_decay_direct=0.1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.2_g_h672f788a3866",
    },
]


HIT_KS = (1, 3, 10)
CATEGORIES = ("nan", "pos_inf", "neg_inf", "gt_high", "lt_low", "abs_gt_high", "nonfinite")


def resolve_path(root, path):
    return path if osp.isabs(path) else osp.join(root, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def empty_category_counts():
    return {key: {"queries": 0, "samples": 0, "examples": []} for key in CATEGORIES}


def add_examples(bucket, examples, limit):
    if len(bucket) >= limit:
        return
    need = limit - len(bucket)
    bucket.extend(examples[:need])


def category_masks(values, high, low):
    values = np.asarray(values)
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


def update_pos_counts(counts, pos, high, low, example_limit):
    masks = category_masks(pos.reshape(pos.shape[0], -1), high, low)
    for cat, mask in masks.items():
        if mask.size == 0:
            continue
        row_has = np.any(mask, axis=1)
        counts[cat]["queries"] += int(np.sum(row_has))
        counts[cat]["samples"] += int(np.sum(mask))
        rows, cols = np.where(mask)
        examples = [
            {"row": int(r), "col": int(c), "value": float(pos.reshape(pos.shape[0], -1)[r, c])}
            for r, c in zip(rows[:example_limit], cols[:example_limit])
        ]
        add_examples(counts[cat]["examples"], examples, example_limit)


def metric_sums_init(prefix=""):
    sums = {"count": 0}
    for kind in ("loose", "strict", "avg"):
        sums[f"{prefix}mrr_{kind}"] = 0.0
        for k in HIT_KS:
            sums[f"{prefix}hit@{k}_{kind}"] = 0.0
    return sums


def add_rank(sums, loose, strict):
    avg = (loose + strict) * 0.5
    sums["count"] += 1
    for kind, rank in (("loose", loose), ("strict", strict), ("avg", avg)):
        sums[f"mrr_{kind}"] += 1.0 / float(rank)
        for k in HIT_KS:
            if rank <= k:
                sums[f"hit@{k}_{kind}"] += 1.0


def finalize_metrics(sums):
    count = max(int(sums.get("count", 0)), 1)
    out = {"count": int(sums.get("count", 0))}
    for kind in ("loose", "strict", "avg"):
        out[f"mrr_{kind}"] = float(sums[f"mrr_{kind}"] / count)
        for k in HIT_KS:
            out[f"hit@{k}_{kind}"] = float(sums[f"hit@{k}_{kind}"] / count)
    return out


def clipped_values(values, high, low):
    values = np.nan_to_num(values, nan=0.0, posinf=high, neginf=low)
    return np.clip(values, low, high)


def finite_min_max_update(stats, values, include_implicit_zero=False):
    finite = values[np.isfinite(values)]
    if finite.size:
        mn = float(np.min(finite))
        mx = float(np.max(finite))
        stats["finite_min"] = mn if stats["finite_min"] is None else min(stats["finite_min"], mn)
        stats["finite_max"] = mx if stats["finite_max"] is None else max(stats["finite_max"], mx)
    if include_implicit_zero:
        stats["finite_min"] = 0.0 if stats["finite_min"] is None else min(stats["finite_min"], 0.0)
        stats["finite_max"] = 0.0 if stats["finite_max"] is None else max(stats["finite_max"], 0.0)


def append_sample(sample, values, limit):
    if len(sample) >= limit:
        return
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    take = min(limit - len(sample), int(finite.size))
    sample.extend(float(x) for x in finite[:take])


def summarize_sample(sample):
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
    shape_issues = []
    if int(meta.get("num_rows", -1)) != n_rows:
        shape_issues.append(f"meta num_rows={meta.get('num_rows')} pos_rows={n_rows}")
    if int(meta.get("max_negs", max_negs)) != max_negs:
        shape_issues.append(f"meta max_negs={meta.get('max_negs')} neg_cols={max_negs}")
    if neg.shape[0] != n_rows:
        shape_issues.append(f"neg_rows={neg.shape[0]} pos_rows={n_rows}")
    if valid_lens.shape[0] != n_rows:
        shape_issues.append(f"valid_lens_rows={valid_lens.shape[0]} pos_rows={n_rows}")
    if np.any(valid_lens < 0) or np.any(valid_lens > max_negs):
        shape_issues.append("valid_lens outside [0, max_negs]")

    pos_counts = empty_category_counts()
    neg_counts = empty_category_counts()
    update_pos_counts(pos_counts, pos, high, low, example_limit)

    raw_sums = metric_sums_init()
    clipped_sums = metric_sums_init()
    stats = {
        "finite_min": None,
        "finite_max": None,
        "stored_valid_neg_nnz": 0,
        "stored_invalid_neg_nnz": 0,
        "implicit_valid_zero_neg_count": 0,
        "queries_with_no_stored_valid_neg": 0,
        "queries_all_candidates_zero": 0,
        "queries_pos_gt_all_neg_raw": 0,
        "queries_pos_tied_with_best_raw": 0,
    }
    sample = []
    finite_min_max_update(stats, pos.reshape(-1), include_implicit_zero=False)
    append_sample(sample, pos.reshape(-1), sample_limit)

    indptr = neg.indptr
    indices = neg.indices
    data = neg.data

    for row in range(n_rows):
        lens = int(valid_lens[row]) if row < valid_lens.shape[0] else 0
        start = int(indptr[row])
        end = int(indptr[row + 1])
        row_idx = indices[start:end]
        row_vals_all = data[start:end]
        valid_mask = row_idx < lens
        invalid_mask = ~valid_mask
        row_vals = row_vals_all[valid_mask]
        row_cols = row_idx[valid_mask]
        stored_valid = int(row_vals.size)
        implicit_zero = max(lens - stored_valid, 0)
        stats["stored_valid_neg_nnz"] += stored_valid
        stats["stored_invalid_neg_nnz"] += int(np.sum(invalid_mask))
        stats["implicit_valid_zero_neg_count"] += int(implicit_zero)
        if stored_valid == 0:
            stats["queries_with_no_stored_valid_neg"] += 1

        finite_min_max_update(stats, row_vals, include_implicit_zero=implicit_zero > 0)
        append_sample(sample, row_vals, sample_limit)

        masks = category_masks(row_vals, high, low)
        for cat, mask in masks.items():
            cnt = int(np.sum(mask))
            if cnt:
                neg_counts[cat]["queries"] += 1
                neg_counts[cat]["samples"] += cnt
                bad_pos = np.flatnonzero(mask)
                examples = [
                    {"row": int(row), "col": int(row_cols[p]), "value": float(row_vals[p])}
                    for p in bad_pos[:example_limit]
                ]
                add_examples(neg_counts[cat]["examples"], examples, example_limit)

        p_raw = float(pos[row, 0])
        gt = int(np.sum(row_vals > p_raw))
        ge = int(np.sum(row_vals >= p_raw))
        if 0.0 > p_raw:
            gt += implicit_zero
        if 0.0 >= p_raw:
            ge += implicit_zero
        add_rank(raw_sums, 1 + gt, 1 + ge)
        if gt == 0:
            stats["queries_pos_gt_all_neg_raw"] += 1
        if gt == 0 and ge > 0:
            stats["queries_pos_tied_with_best_raw"] += 1

        p_clip = float(clipped_values(np.asarray([p_raw], dtype=np.float64), high, low)[0])
        row_clip = clipped_values(row_vals.astype(np.float64, copy=False), high, low)
        gt_clip = int(np.sum(row_clip > p_clip))
        ge_clip = int(np.sum(row_clip >= p_clip))
        if 0.0 > p_clip:
            gt_clip += implicit_zero
        if 0.0 >= p_clip:
            ge_clip += implicit_zero
        add_rank(clipped_sums, 1 + gt_clip, 1 + ge_clip)

        if p_raw == 0.0 and stored_valid == 0 and implicit_zero == lens:
            stats["queries_all_candidates_zero"] += 1

    total_valid_neg = int(np.sum(valid_lens)) if valid_lens.size else 0
    density = float(stats["stored_valid_neg_nnz"] / total_valid_neg) if total_valid_neg else 0.0
    return {
        "split": split,
        "missing": [],
        "meta": meta,
        "shape": {
            "pos_shape": list(pos.shape),
            "neg_shape": list(neg.shape),
            "valid_lens_shape": list(valid_lens.shape),
            "shape_issues": shape_issues,
            "valid_lens_min": int(np.min(valid_lens)) if valid_lens.size else None,
            "valid_lens_max": int(np.max(valid_lens)) if valid_lens.size else None,
            "valid_lens_unique_first10": [int(x) for x in np.unique(valid_lens)[:10]],
        },
        "score_range": {
            "finite_min": stats["finite_min"],
            "finite_max": stats["finite_max"],
            "sampled_finite_quantiles": summarize_sample(sample),
            "sample_size": len(sample),
        },
        "storage": {
            "total_valid_neg": total_valid_neg,
            "stored_valid_neg_nnz": int(stats["stored_valid_neg_nnz"]),
            "stored_invalid_neg_nnz": int(stats["stored_invalid_neg_nnz"]),
            "implicit_valid_zero_neg_count": int(stats["implicit_valid_zero_neg_count"]),
            "stored_valid_neg_density": density,
        },
        "pos_anomalies": pos_counts,
        "neg_anomalies": neg_counts,
        "other_checks": {
            "queries_with_no_stored_valid_neg": int(stats["queries_with_no_stored_valid_neg"]),
            "queries_all_candidates_zero": int(stats["queries_all_candidates_zero"]),
            "queries_pos_gt_all_neg_raw": int(stats["queries_pos_gt_all_neg_raw"]),
            "queries_pos_tied_with_best_raw": int(stats["queries_pos_tied_with_best_raw"]),
        },
        "metrics_raw": finalize_metrics(raw_sums),
        "metrics_clipped": finalize_metrics(clipped_sums),
    }


def print_split_summary(result, high, low):
    split = result["split"]
    if result.get("missing"):
        print(f"  [{split}] MISSING")
        for p in result["missing"]:
            print(f"    {p}")
        return
    shape = result["shape"]
    metrics = result["metrics_raw"]
    metrics_clip = result["metrics_clipped"]
    print(
        f"  [{split}] rows={shape['pos_shape'][0]} neg_cols={shape['neg_shape'][1]} "
        f"valid_lens={shape['valid_lens_min']}..{shape['valid_lens_max']} "
        f"raw_mrr={metrics['mrr_strict']:.5f} raw_hr1={metrics['hit@1_strict']:.5f} "
        f"raw_hr10={metrics['hit@10_strict']:.5f}"
    )
    print(
        f"        clipped[{low:g},{high:g}] mrr={metrics_clip['mrr_strict']:.5f} "
        f"hr1={metrics_clip['hit@1_strict']:.5f} hr10={metrics_clip['hit@10_strict']:.5f}"
    )
    rng = result["score_range"]
    print(
        f"        finite_range=[{rng['finite_min']}, {rng['finite_max']}] "
        f"sample_quantiles={rng['sampled_finite_quantiles']}"
    )
    storage = result["storage"]
    print(
        f"        neg_storage density={storage['stored_valid_neg_density']:.6g} "
        f"stored_valid={storage['stored_valid_neg_nnz']} implicit_zero={storage['implicit_valid_zero_neg_count']} "
        f"stored_invalid={storage['stored_invalid_neg_nnz']}"
    )
    for side in ("pos_anomalies", "neg_anomalies"):
        counts = result[side]
        pieces = []
        for cat in CATEGORIES:
            item = counts[cat]
            if item["samples"]:
                pieces.append(f"{cat}:q={item['queries']},n={item['samples']}")
        print(f"        {side}: " + (", ".join(pieces) if pieces else "none"))
    other = result["other_checks"]
    print(
        "        other: "
        f"no_stored_valid_neg_q={other['queries_with_no_stored_valid_neg']} "
        f"all_candidates_zero_q={other['queries_all_candidates_zero']} "
        f"pos_gt_all_neg_q={other['queries_pos_gt_all_neg_raw']} "
        f"pos_tied_best_q={other['queries_pos_tied_with_best_raw']}"
    )
    issues = shape.get("shape_issues") or []
    if issues:
        print(f"        shape_issues: {issues}")


def parse_args():
    parser = argparse.ArgumentParser("Check saved structure score stores for anomalous values.")
    parser.add_argument("--root", default=".", help="Repository/server run root that contains results_new_structure.")
    parser.add_argument("--dataset", default="all", choices=("all", "ICEWS14", "GDELT", "tkgl-polecat"))
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to check.")
    parser.add_argument("--high", type=float, default=10000.0)
    parser.add_argument("--low", type=float, default=-10000.0)
    parser.add_argument("--example_limit", type=int, default=5)
    parser.add_argument("--sample_limit", type=int, default=200000)
    parser.add_argument("--json_out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    selected = [r for r in RUNS if args.dataset == "all" or r["dataset"] == args.dataset]
    report = {
        "thresholds": {"high": float(args.high), "low": float(args.low)},
        "splits": splits,
        "runs": [],
    }
    for idx, run in enumerate(selected, start=1):
        run_dir = resolve_path(args.root, run["path"])
        print("=" * 100)
        print(f"[{idx}/{len(selected)}] {run['dataset']} {run['name']}")
        print(f"  logged_test: mrr={run['logged_test_mrr']:.5f} hr1={run['logged_test_hr1']:.5f} hr10={run['logged_test_hr10']:.5f}")
        print(f"  path: {run_dir}")
        run_result = dict(run)
        run_result["resolved_path"] = run_dir
        run_result["splits"] = {}
        if not osp.isdir(run_dir):
            print("  MISSING RUN DIR")
            run_result["missing_run_dir"] = True
            report["runs"].append(run_result)
            continue
        run_result["missing_run_dir"] = False
        for split in splits:
            result = analyze_split(
                run_dir,
                split,
                high=float(args.high),
                low=float(args.low),
                example_limit=int(args.example_limit),
                sample_limit=int(args.sample_limit),
            )
            run_result["splits"][split] = result
            print_split_summary(result, high=float(args.high), low=float(args.low))
        report["runs"].append(run_result)

    if args.json_out:
        out_path = resolve_path(args.root, args.json_out)
        os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("=" * 100)
        print(f"saved JSON report -> {out_path}")


if __name__ == "__main__":
    main()
