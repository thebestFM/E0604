import argparse
import json
import os
import os.path as osp

import numpy as np
from scipy.sparse import load_npz


HYBRID_OUTPUT_DIR = "results_new_hybrid/ICEWS14/seed42/new_hybrid_f43c3bb8adc2"


RUNS = [
    {
        "kind": "structure",
        "id": "s01_0609_rank1_v2",
        "logged_test_mrr": 0.42196,
        "logged_test_hr1": 0.32317,
        "logged_test_hr10": 0.61693,
        "path": "results_new_structure/ICEWS14/seed42/decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.95_gamma=0_impl=new_structure_hb8a1ef33f08b",
    },
    {
        "kind": "structure",
        "id": "s02_0609_rank2_v2",
        "logged_test_mrr": 0.42162,
        "logged_test_hr1": 0.32268,
        "logged_test_hr10": 0.61704,
        "path": "results_new_structure/ICEWS14/seed42/decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.9_gamma=0_impl=new_structure__h0cd799355526",
    },
    {
        "kind": "structure",
        "id": "s03_0609_rank3_v2",
        "logged_test_mrr": 0.42117,
        "logged_test_hr1": 0.32245,
        "logged_test_hr10": 0.61685,
        "path": "results_new_structure/ICEWS14/seed42/decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.8_gamma=0_impl=new_structure__h13a42747ada6",
    },
    {
        "kind": "time",
        "id": "time_cfg2_mrr",
        "logged_test_mrr": 0.34213,
        "logged_test_hr1": 0.24070,
        "logged_test_hr10": 0.53687,
        "path": "results_time_tkg_single/ICEWS14/seed42/r9eb5b85515d8_topk30_mw5-15-30_ed96_hd192_bs4096_ebs384_neg6_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin",
    },
    {
        "kind": "time",
        "id": "time_cfg1_mrr",
        "logged_test_mrr": 0.34106,
        "logged_test_hr1": 0.23998,
        "logged_test_hr10": 0.54054,
        "path": "results_time_tkg_single/ICEWS14/seed42/r210529791eed_topk40_mw5-15-30-60_ed96_hd192_bs4096_ebs384_neg8_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin",
    },
    {
        "kind": "time",
        "id": "time_cfg3_hr10",
        "logged_test_mrr": 0.32153,
        "logged_test_hr1": 0.20746,
        "logged_test_hr10": 0.54330,
        "path": "results_time_tkg_single/ICEWS14/seed42/r041812cea350_topk70_mw5-15-30-60-120_ed96_hd192_bs4096_ebs384_neg4_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin",
    },
]


HIT_KS = (1, 3, 10)
CATEGORIES = ("nan", "pos_inf", "neg_inf", "gt_high", "lt_low", "abs_gt_high", "nonfinite")


def resolve_path(root, path):
    return path if osp.isabs(path) else osp.join(root, path)


def read_json(path):
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
    return {key: {"queries": 0, "samples": 0, "examples": []} for key in CATEGORIES}


def add_examples(dst, examples, limit):
    if len(dst) < limit:
        dst.extend(examples[: limit - len(dst)])


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


def summarize_sample(sample):
    if not sample:
        return {}
    arr = np.asarray(sample, dtype=np.float64)
    qs = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]
    vals = np.quantile(arr, qs)
    return {f"q{q:g}": float(v) for q, v in zip(qs, vals)}


def append_sample(sample, values, limit):
    if len(sample) >= limit:
        return
    finite = values[np.isfinite(values)]
    if finite.size:
        take = min(limit - len(sample), int(finite.size))
        sample.extend(float(x) for x in finite[:take])


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
    meta = read_json(meta_path)
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
        "queries_pos_tied_with_best": 0,
        "queries_loose_rank1": 0,
        "queries_strict_rank1": 0,
        "sum_equal_pos": 0,
        "max_equal_pos": 0,
    }
    sample = []

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
    append_sample(sample, pos.reshape(-1), sample_limit)

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
        append_sample(sample, row_vals, sample_limit)

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
            stats["queries_loose_rank1"] += 1
        if strict == 1:
            stats["queries_strict_rank1"] += 1
        if loose == 1 and strict > 1:
            stats["queries_pos_tied_with_best"] += 1
        stats["sum_equal_pos"] += int(eq)
        stats["max_equal_pos"] = max(stats["max_equal_pos"], int(eq))
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
            "sampled_finite_quantiles": summarize_sample(sample),
            "sample_size": len(sample),
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
            "queries_loose_rank1": int(stats["queries_loose_rank1"]),
            "queries_strict_rank1": int(stats["queries_strict_rank1"]),
            "queries_pos_tied_with_best": int(stats["queries_pos_tied_with_best"]),
            "avg_equal_pos_per_query": float(stats["sum_equal_pos"] / max(n_rows, 1)),
            "max_equal_pos": int(stats["max_equal_pos"]),
        },
        "metrics_raw": finalize(raw_sums),
        "metrics_clipped": finalize(clip_sums),
    }


def print_split(result, high, low):
    split = result["split"]
    if result.get("missing"):
        print(f"  [{split}] MISSING")
        for p in result["missing"]:
            print(f"    {p}")
        return
    raw = result["metrics_raw"]
    clipped = result["metrics_clipped"]
    shape = result["shape"]
    print(
        f"  [{split}] rows={shape['pos_shape'][0]} neg_cols={shape['neg_shape'][1]} "
        f"valid_lens={shape['valid_lens_min']}..{shape['valid_lens_max']} "
        f"raw_mrr={raw['mrr_strict']:.5f} raw_hr1={raw['hit@1_strict']:.5f} raw_hr10={raw['hit@10_strict']:.5f}"
    )
    print(
        f"        clipped[{low:g},{high:g}] mrr={clipped['mrr_strict']:.5f} "
        f"hr1={clipped['hit@1_strict']:.5f} hr10={clipped['hit@10_strict']:.5f}"
    )
    print(f"        finite_range={result['score_range']['finite_min']}..{result['score_range']['finite_max']}")
    print(f"        sample_quantiles={result['score_range']['sampled_finite_quantiles']}")
    for side in ("pos_anomalies", "neg_anomalies"):
        pieces = []
        for cat in CATEGORIES:
            item = result[side][cat]
            if item["samples"]:
                pieces.append(f"{cat}:q={item['queries']},n={item['samples']}")
        print(f"        {side}: " + (", ".join(pieces) if pieces else "none"))
    ties = result["tie_checks"]
    print(
        f"        ties: pos_zero_q={ties['queries_pos_zero']} all_zero_q={ties['queries_all_candidates_zero']} "
        f"loose_rank1_q={ties['queries_loose_rank1']} strict_rank1_q={ties['queries_strict_rank1']} "
        f"pos_tied_best_q={ties['queries_pos_tied_with_best']} "
        f"avg_equal_pos={ties['avg_equal_pos_per_query']:.2f} max_equal_pos={ties['max_equal_pos']}"
    )
    storage = result["storage"]
    print(
        f"        storage: stored_valid={storage['stored_valid_neg_nnz']} "
        f"implicit_zero={storage['implicit_zero_neg_count']} invalid_stored={storage['stored_invalid_neg_nnz']}"
    )
    if shape["shape_issues"]:
        print(f"        shape_issues={shape['shape_issues']}")


def inspect_hybrid_summary(root):
    out_dir = resolve_path(root, HYBRID_OUTPUT_DIR)
    summary_path = osp.join(out_dir, "summary.json")
    metrics_path = osp.join(out_dir, "metrics.json")
    print("=" * 100)
    print(f"[hybrid output] {out_dir}")
    if not osp.isdir(out_dir):
        print("  missing output dir")
        return {"missing_output_dir": True}
    payload = {"missing_output_dir": False, "output_dir": out_dir}
    for path in (summary_path, metrics_path):
        if osp.isfile(path):
            data = read_json(path)
            payload[osp.basename(path)] = data
            best = data.get("best") if isinstance(data, dict) else None
            if best:
                tm = best.get("test_metrics", {})
                print(
                    f"  {osp.basename(path)} best: struct={best.get('struct_id')} time={best.get('time_id')} "
                    f"param={best.get('best_param_name')} mrr={tm.get('mrr_strict')} "
                    f"hr1={tm.get('hit@1_strict')} hr10={tm.get('hit@10_strict')}"
                )
        else:
            print(f"  missing {path}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser("Check score stores used by new_hybrid_ICEWS14_latest.log.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--kind", default="all", choices=("all", "structure", "time"))
    parser.add_argument("--splits", default="test")
    parser.add_argument("--high", type=float, default=10000.0)
    parser.add_argument("--low", type=float, default=-10000.0)
    parser.add_argument("--example_limit", type=int, default=5)
    parser.add_argument("--sample_limit", type=int, default=200000)
    parser.add_argument("--json_out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    selected = [r for r in RUNS if args.kind == "all" or r["kind"] == args.kind]
    report = {
        "thresholds": {"high": float(args.high), "low": float(args.low)},
        "splits": splits,
        "hybrid_summary": inspect_hybrid_summary(args.root),
        "runs": [],
    }
    for idx, run in enumerate(selected, start=1):
        run_dir = resolve_path(args.root, run["path"])
        print("=" * 100)
        print(f"[{idx}/{len(selected)}] {run['kind']} {run['id']}")
        print(
            f"  logged_test: mrr={run['logged_test_mrr']:.5f} "
            f"hr1={run['logged_test_hr1']:.5f} hr10={run['logged_test_hr10']:.5f}"
        )
        print(f"  path: {run_dir}")
        run_report = dict(run)
        run_report["resolved_path"] = run_dir
        run_report["splits"] = {}
        if not osp.isdir(run_dir):
            print("  MISSING RUN DIR")
            run_report["missing_run_dir"] = True
            report["runs"].append(run_report)
            continue
        run_report["missing_run_dir"] = False
        for split in splits:
            result = analyze_split(
                run_dir,
                split,
                float(args.high),
                float(args.low),
                int(args.example_limit),
                int(args.sample_limit),
            )
            run_report["splits"][split] = result
            print_split(result, float(args.high), float(args.low))
        report["runs"].append(run_report)
    if args.json_out:
        out_path = resolve_path(args.root, args.json_out)
        os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("=" * 100)
        print(f"saved JSON report -> {out_path}")


if __name__ == "__main__":
    main()
