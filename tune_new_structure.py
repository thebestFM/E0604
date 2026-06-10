import argparse
import copy
import json
import os
import os.path as osp
from types import SimpleNamespace

import numpy as np

import train_new_structure
from utils import (
    ScoreStore,
    add_metric_sums,
    compute_ranking_metric_sums,
    finalize_metric_sums,
    load_metrics,
    save_config,
)


DATASET_DEFAULTS = {
    "ICEWS14": {
        "ns_q": 6000,
        "batch_size": 8192,
        "max_events_in_single_batch": 60000,
        "dsh_decay": 1.0,
        "combine_weights": [0.60, 0.75, 0.85, 0.90, 0.95],
        "shared_gammas": [0.001, 0.003, 0.01, 0.03, 0.10],
        "top_dmh_for_shared": 3,
        "top_dsh_for_combo": 3,
        "top_c_for_combo": 3,
    },
    "GDELT": {
        "ns_q": 5000,
        "batch_size": 8192,
        "max_events_in_single_batch": 60000,
        "dsh_decay": 0.1,
        "combine_weights": [0.60, 0.75, 0.85, 0.90, 0.95],
        "shared_gammas": [0.0003, 0.001, 0.003, 0.01, 0.03],
        "top_dmh_for_shared": 3,
        "top_dsh_for_combo": 3,
        "top_c_for_combo": 3,
    },
    "tkgl-polecat": {
        "ns_q": 5000,
        "batch_size": 4096,
        "max_events_in_single_batch": 60000,
        "dsh_decay": 0.01,
        "combine_weights": [0.02, 0.05, 0.10, 0.15, 0.25],
        "shared_gammas": [0.003, 0.006, 0.01, 0.02, 0.04],
        "top_dmh_for_shared": 3,
        "top_dsh_for_combo": 3,
        "top_c_for_combo": 3,
    },
    "tkgl-icews": {
        "ns_q": 5000,
        "batch_size": 4096,
        "max_events_in_single_batch": 60000,
        "dsh_decay": 1.0,
        "combine_weights": [0.60, 0.75, 0.88, 0.95],
        "shared_gammas": [0.003, 0.006, 0.01, 0.02],
        "top_dmh_for_shared": 2,
        "top_dsh_for_combo": 3,
        "top_c_for_combo": 2,
    },
}


METRIC_KEYS = {
    "MRR": "test_mrr_strict",
    "H1": "test_hit@1_strict",
    "H10": "test_hit@10_strict",
}


def metric_value(record, focus):
    return float(record["metrics"][METRIC_KEYS[focus]])


def structure_arg_dict(args, overrides):
    defaults = DATASET_DEFAULTS[args.dataset]
    cfg = {
        "dataset": args.dataset,
        "seed": args.seed,
        "gpu": args.gpu,
        "ns_q": defaults["ns_q"],
        "ns_seed": args.ns_seed,
        "train_predict_ratio": args.train_predict_ratio,
        "batch_size": defaults["batch_size"],
        "max_events_in_single_batch": defaults["max_events_in_single_batch"],
        "source_join_threads": args.source_join_threads,
        "source_join_log_batches": args.source_join_log_batches,
        "close_update_backward": False,
        "dict_mode": "tag_sum",
        "shared_w": "dual_msim",
        "per_rel_use_mtrans": False,
        "ppr_k": 1000,
        "top_k_relation": 0,
        "ppr_alpha": 0.012,
        "ppr_beta": 0.93,
        "gamma": 0.0,
        "direct_single_hop": 1.0,
        "decay_direct": defaults["dsh_decay"],
        "top_share": 100,
        "top_direct": -1,
        "decay_rel_trans": 0.05,
        "window_semantic_sim": 5.0,
        "window_trans": 5.0,
        "skip_val_eval": True,
    }
    cfg.update(overrides)
    return cfg


def dsh_grid(dataset):
    common = {"direct_single_hop": 1.0, "gamma": 0.0, "dict_mode": "tag_sum", "ppr_alpha": 0.012, "ppr_beta": 0.93}
    if dataset == "ICEWS14":
        decays = [0.35, 0.50, 0.70, 0.85, 1.00, 1.15, 1.35, 1.60, 2.00, 2.50]
    elif dataset == "GDELT":
        decays = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.80]
    elif dataset == "tkgl-polecat":
        decays = [0.001, 0.002, 0.005, 0.008, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500]
    elif dataset == "tkgl-icews":
        decays = [0.50, 0.80, 1.00, 1.30, 1.80]
    else:
        raise ValueError(dataset)
    return [dict(common, decay_direct=d) for d in decays]


def dmh_grid(dataset):
    base = {
        "direct_single_hop": 0.0,
        "gamma": 0.0,
        "shared_w": "dual_msim",
        "ppr_k": 1000,
        "top_share": 100,
        "top_direct": -1,
        "decay_rel_trans": 0.05,
        "window_semantic_sim": 5.0,
        "window_trans": 5.0,
    }
    if dataset == "ICEWS14":
        rows = [
            ("tag_sum", 0.008, 0.90),
            ("tag_sum", 0.010, 0.91),
            ("tag_sum", 0.012, 0.92),
            ("tag_sum", 0.014, 0.93),
            ("tag_sum", 0.016, 0.94),
            ("tag_sum", 0.020, 0.95),
            ("tag_sum", 0.030, 0.96),
            ("tag_max", 0.010, 0.92),
            ("tag_max", 0.015, 0.93),
            ("tag_max", 0.025, 0.95),
        ]
        decay = 1.0
    elif dataset == "GDELT":
        rows = [
            ("tag_sum", 0.006, 0.93),
            ("tag_sum", 0.008, 0.94),
            ("tag_sum", 0.010, 0.94),
            ("tag_sum", 0.012, 0.95),
            ("tag_sum", 0.015, 0.95),
            ("tag_sum", 0.020, 0.96),
            ("tag_sum", 0.025, 0.90),
            ("tag_max", 0.006, 0.97),
            ("tag_max", 0.010, 0.94),
            ("tag_max", 0.015, 0.95),
        ]
        decay = 0.1
    elif dataset == "tkgl-polecat":
        rows = [
            ("tag_sum", 0.0100, 0.920),
            ("tag_sum", 0.0125, 0.925),
            ("tag_sum", 0.0150, 0.930),
            ("tag_sum", 0.01579502319249557, 0.9343207039457382),
            ("tag_sum", 0.0175, 0.938),
            ("tag_sum", 0.0200, 0.940),
            ("tag_sum", 0.0250, 0.945),
            ("tag_max", 0.0125, 0.930),
            ("tag_max", 0.01579502319249557, 0.9343207039457382),
            ("tag_max", 0.0200, 0.940),
        ]
        decay = 0.01
    elif dataset == "tkgl-icews":
        rows = [
            ("tag_sum", 0.0125, 0.925),
            ("tag_sum", 0.01579502319249557, 0.9343207039457382),
            ("tag_sum", 0.0180, 0.940),
            ("tag_sum", 0.0220, 0.948),
            ("tag_max", 0.01579502319249557, 0.9343207039457382),
        ]
        decay = 1.0
    else:
        raise ValueError(dataset)
    return [dict(base, dict_mode=m, ppr_alpha=a, ppr_beta=b, decay_direct=decay) for m, a, b in rows]


def shared_variants(dataset, base_cfg):
    gammas = DATASET_DEFAULTS[dataset]["shared_gammas"]
    variants = []
    for gamma in gammas:
        cfg = copy.deepcopy(base_cfg)
        cfg["gamma"] = gamma
        cfg["top_share"] = 100
        cfg["top_direct"] = 500 if dataset == "tkgl-polecat" else -1
        variants.append(cfg)
    return variants


def stage2_grid(dataset):
    if dataset == "ICEWS14":
        base = {
            "dict_mode": "tag_sum",
            "ppr_alpha": 0.030,
            "ppr_beta": 0.960,
            "gamma": 0.003,
            "direct_single_hop": 0.85,
            "decay_direct": 0.35,
            "top_direct": -1,
        }
        rows = [
            ("icews14_s2_best_replay", {}),
            ("icews14_s2_decay0.30", {"decay_direct": 0.30}),
            ("icews14_s2_decay0.25", {"decay_direct": 0.25}),
            ("icews14_s2_decay0.20", {"decay_direct": 0.20}),
            ("icews14_s2_decay0.40", {"decay_direct": 0.40}),
            ("icews14_s2_w0.80", {"direct_single_hop": 0.80}),
            ("icews14_s2_w0.88", {"direct_single_hop": 0.88}),
            ("icews14_s2_w0.92", {"direct_single_hop": 0.92}),
            ("icews14_s2_decay0.30_w0.80", {"decay_direct": 0.30, "direct_single_hop": 0.80}),
            ("icews14_s2_decay0.30_w0.90", {"decay_direct": 0.30, "direct_single_hop": 0.90}),
            ("icews14_s2_a0.035_b0.965", {"ppr_alpha": 0.035, "ppr_beta": 0.965}),
            ("icews14_s2_a0.040_b0.970", {"ppr_alpha": 0.040, "ppr_beta": 0.970}),
            ("icews14_s2_a0.025_b0.955", {"ppr_alpha": 0.025, "ppr_beta": 0.955}),
            ("icews14_s2_a0.030_b0.965", {"ppr_beta": 0.965}),
            ("icews14_s2_a0.035_b0.960", {"ppr_alpha": 0.035}),
            ("icews14_s2_gamma0", {"gamma": 0.0}),
            ("icews14_s2_gamma0.001", {"gamma": 0.001}),
            ("icews14_s2_gamma0.006", {"gamma": 0.006}),
            ("icews14_s2_gamma0.010", {"gamma": 0.010}),
            ("icews14_s2_w0.65", {"direct_single_hop": 0.65}),
            ("icews14_s2_w0.75", {"direct_single_hop": 0.75}),
        ]
    elif dataset == "tkgl-polecat":
        base = {
            "dict_mode": "tag_sum",
            "ppr_alpha": 0.025,
            "ppr_beta": 0.945,
            "gamma": 0.0,
            "direct_single_hop": 0.15,
            "decay_direct": 0.002,
            "top_direct": -1,
        }
        rows = [
            ("polecat_s2_best_replay", {}),
            ("polecat_s2_decay0.0015", {"decay_direct": 0.0015}),
            ("polecat_s2_decay0.003", {"decay_direct": 0.003}),
            ("polecat_s2_decay0.004", {"decay_direct": 0.004}),
            ("polecat_s2_decay0.005", {"decay_direct": 0.005}),
            ("polecat_s2_w0.10", {"direct_single_hop": 0.10}),
            ("polecat_s2_w0.12", {"direct_single_hop": 0.12}),
            ("polecat_s2_w0.18", {"direct_single_hop": 0.18}),
            ("polecat_s2_w0.20", {"direct_single_hop": 0.20}),
            ("polecat_s2_a0.0225_b0.942", {"ppr_alpha": 0.0225, "ppr_beta": 0.942}),
            ("polecat_s2_a0.0275_b0.947", {"ppr_alpha": 0.0275, "ppr_beta": 0.947}),
            ("polecat_s2_a0.030_b0.950", {"ppr_alpha": 0.030, "ppr_beta": 0.950}),
            ("polecat_s2_a0.035_b0.955", {"ppr_alpha": 0.035, "ppr_beta": 0.955}),
            ("polecat_s2_b0.950", {"ppr_beta": 0.950}),
            ("polecat_s2_gamma0.0005_top500", {"gamma": 0.0005, "top_direct": 500}),
            ("polecat_s2_gamma0.001_top500", {"gamma": 0.001, "top_direct": 500}),
            ("polecat_s2_gamma0.002_top500", {"gamma": 0.002, "top_direct": 500}),
            ("polecat_s2_gamma0.003_top500", {"gamma": 0.003, "top_direct": 500}),
            ("polecat_s2_gamma0.001_full_direct", {"gamma": 0.001, "top_direct": -1}),
            ("polecat_s2_w0.12_decay0.003_a0.0275", {
                "direct_single_hop": 0.12,
                "decay_direct": 0.003,
                "ppr_alpha": 0.0275,
                "ppr_beta": 0.947,
            }),
        ]
    else:
        raise ValueError("--stage stage2 currently supports only ICEWS14 and tkgl-polecat")

    configs = []
    for label, overrides in rows:
        cfg = copy.deepcopy(base)
        cfg.update(overrides)
        cfg["stage2_id"] = label
        configs.append(cfg)
    return configs


def run_structure(args, stage, idx, overrides):
    cfg = structure_arg_dict(args, overrides)
    sargs = SimpleNamespace(**cfg)
    out_dir = train_new_structure.make_new_result_dir(sargs)
    print(f"[TuneStructure] {stage}[{idx}] -> {out_dir}", flush=True)
    metrics = train_new_structure.main(sargs)
    return {
        "kind": "structure",
        "stage": stage,
        "index": int(idx),
        "out_dir": out_dir,
        "config": cfg,
        "metrics": metrics,
    }


def evaluate_linear_combo(dsh_dir, c_dir, weight_dsh, ns_q, block_size=1024):
    dsh_store = ScoreStore(dsh_dir, "test")
    c_store = ScoreStore(c_dir, "test")
    if dsh_store.num_rows != c_store.num_rows:
        raise RuntimeError(f"row mismatch: dsh={dsh_store.num_rows}, c={c_store.num_rows}")
    width = min(int(ns_q), dsh_store.max_negs, c_store.max_negs)
    sums = {}
    for start in range(0, dsh_store.num_rows, int(block_size)):
        end = min(start + int(block_size), dsh_store.num_rows)
        dsh_pos, dsh_neg, dsh_mask = dsh_store.get_block(start, end, width)
        c_pos, c_neg, c_mask = c_store.get_block(start, end, width)
        mask = dsh_mask & c_mask
        if int(ns_q) > 0 and not np.all(mask):
            bad = int(mask.size - np.sum(mask))
            raise RuntimeError(f"invalid combo mask in rows {start}:{end}, bad={bad}")
        pos = float(weight_dsh) * dsh_pos + (1.0 - float(weight_dsh)) * c_pos
        neg = float(weight_dsh) * dsh_neg + (1.0 - float(weight_dsh)) * c_neg
        pos_bad = int(np.size(pos) - np.sum(np.isfinite(pos)))
        neg_bad = int(np.size(neg[mask]) - np.sum(np.isfinite(neg[mask])))
        if pos_bad or neg_bad:
            raise RuntimeError(
                f"non-finite combo scores rows {start}:{end}: pos_bad={pos_bad} neg_bad={neg_bad}"
            )
        add_metric_sums(sums, compute_ranking_metric_sums(pos, neg, mask))
    metrics = finalize_metric_sums(sums)
    return {f"test_{k}": v for k, v in metrics.items()}


def sort_top(records, focus, k):
    return sorted(records, key=lambda x: metric_value(x, focus), reverse=True)[: int(k)]


def record_line(record, focus):
    m = record["metrics"]
    return (
        f"{record['stage']}[{record['index']}] "
        f"{METRIC_KEYS[focus]}={metric_value(record, focus):.5f} "
        f"MRR={m['test_mrr_strict']:.5f} "
        f"H1={m['test_hit@1_strict']:.5f} "
        f"H10={m['test_hit@10_strict']:.5f}"
    )


def output_dir(args):
    suffix = f"metric={args.focus_test_metric}_nsq={DATASET_DEFAULTS[args.dataset]['ns_q']}_nsseed={args.ns_seed}"
    if getattr(args, "stage", "stage1") != "stage1":
        suffix = f"{suffix}_{args.stage}"
    return osp.join(
        args.output_root,
        args.dataset,
        f"seed{args.seed}",
        suffix,
    )


def save_summary(out_dir, payload):
    os.makedirs(out_dir, exist_ok=True)
    save_config(out_dir, payload["args"])
    with open(osp.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(args):
    if args.dataset not in DATASET_DEFAULTS:
        raise ValueError(f"--dataset must be one of {sorted(DATASET_DEFAULTS)}")
    if getattr(args, "stage", "stage1") == "stage2":
        return main_stage2(args)
    focus = args.focus_test_metric
    out_dir = output_dir(args)
    summary = {
        "format": "new_structure_tuning_v1",
        "args": vars(args).copy(),
        "dataset_defaults": DATASET_DEFAULTS[args.dataset],
        "dsh": [],
        "dmh": [],
        "dmh_shared": [],
        "offline_combos": [],
    }
    save_summary(out_dir, summary)

    print(f"[TuneStructure] output -> {out_dir}", flush=True)
    print("[TuneStructure] stage 1: pure DSH", flush=True)
    for idx, cfg in enumerate(dsh_grid(args.dataset), start=1):
        rec = run_structure(args, "pure_dsh", idx, cfg)
        summary["dsh"].append(rec)
        print("[TuneStructure] " + record_line(rec, focus), flush=True)
        save_summary(out_dir, summary)

    print("[TuneStructure] stage 2a: pure DMH", flush=True)
    for idx, cfg in enumerate(dmh_grid(args.dataset), start=1):
        rec = run_structure(args, "pure_dmh", idx, cfg)
        summary["dmh"].append(rec)
        print("[TuneStructure] " + record_line(rec, focus), flush=True)
        save_summary(out_dir, summary)

    top_dmh = sort_top(summary["dmh"], focus, DATASET_DEFAULTS[args.dataset]["top_dmh_for_shared"])
    print("[TuneStructure] top pure DMH:", flush=True)
    for rec in top_dmh:
        print("[TuneStructure]   " + record_line(rec, focus), flush=True)

    print("[TuneStructure] stage 2b: DMH + Shared from top pure DMH", flush=True)
    idx = 1
    for base in top_dmh:
        for cfg in shared_variants(args.dataset, base["config"]):
            rec = run_structure(args, "dmh_shared", idx, cfg)
            rec["parent_stage"] = base["stage"]
            rec["parent_index"] = base["index"]
            rec["parent_out_dir"] = base["out_dir"]
            summary["dmh_shared"].append(rec)
            print("[TuneStructure] " + record_line(rec, focus), flush=True)
            save_summary(out_dir, summary)
            idx += 1

    top_dsh = sort_top(summary["dsh"], focus, DATASET_DEFAULTS[args.dataset]["top_dsh_for_combo"])
    c_pool = summary["dmh"] + summary["dmh_shared"]
    top_c = sort_top(c_pool, focus, DATASET_DEFAULTS[args.dataset]["top_c_for_combo"])
    print("[TuneStructure] top pure DSH:", flush=True)
    for rec in top_dsh:
        print("[TuneStructure]   " + record_line(rec, focus), flush=True)
    print("[TuneStructure] top C-side runs:", flush=True)
    for rec in top_c:
        print("[TuneStructure]   " + record_line(rec, focus), flush=True)

    print("[TuneStructure] stage 3: offline DSH x C linear combinations", flush=True)
    combo_idx = 1
    for dsh_rec in top_dsh:
        for c_rec in top_c:
            for w in DATASET_DEFAULTS[args.dataset]["combine_weights"]:
                metrics = evaluate_linear_combo(
                    dsh_rec["out_dir"],
                    c_rec["out_dir"],
                    weight_dsh=w,
                    ns_q=DATASET_DEFAULTS[args.dataset]["ns_q"],
                    block_size=args.combine_query_batch,
                )
                rec = {
                    "kind": "offline_combo",
                    "stage": "offline_dsh_x_c",
                    "index": combo_idx,
                    "weight_dsh": float(w),
                    "weight_c": float(1.0 - w),
                    "dsh_stage": dsh_rec["stage"],
                    "dsh_index": dsh_rec["index"],
                    "dsh_out_dir": dsh_rec["out_dir"],
                    "c_stage": c_rec["stage"],
                    "c_index": c_rec["index"],
                    "c_out_dir": c_rec["out_dir"],
                    "metrics": metrics,
                }
                summary["offline_combos"].append(rec)
                print("[TuneStructure] " + record_line(rec, focus), flush=True)
                save_summary(out_dir, summary)
                combo_idx += 1

    all_records = summary["dsh"] + summary["dmh"] + summary["dmh_shared"] + summary["offline_combos"]
    summary["best"] = sort_top(all_records, focus, 1)[0]
    save_summary(out_dir, summary)
    best = summary["best"]
    print("[TuneStructure] best overall: " + record_line(best, focus), flush=True)
    print(f"[TuneStructure] summary saved: {osp.join(out_dir, 'summary.json')}", flush=True)
    return summary


def main_stage2(args):
    focus = args.focus_test_metric
    out_dir = output_dir(args)
    summary = {
        "format": "new_structure_tuning_stage2_v1",
        "args": vars(args).copy(),
        "dataset_defaults": DATASET_DEFAULTS[args.dataset],
        "stage2": [],
    }
    save_summary(out_dir, summary)

    print(f"[TuneStructure] output -> {out_dir}", flush=True)
    print("[TuneStructure] stage 2: focused full-structure combinations", flush=True)
    for idx, cfg in enumerate(stage2_grid(args.dataset), start=1):
        stage2_id = cfg.pop("stage2_id")
        rec = run_structure(args, "stage2_full", idx, cfg)
        rec["stage2_id"] = stage2_id
        summary["stage2"].append(rec)
        print(f"[TuneStructure] {stage2_id} " + record_line(rec, focus), flush=True)
        save_summary(out_dir, summary)

    summary["best"] = sort_top(summary["stage2"], focus, 1)[0]
    save_summary(out_dir, summary)
    best = summary["best"]
    print("[TuneStructure] best stage2: " + record_line(best, focus), flush=True)
    print(f"[TuneStructure] summary saved: {osp.join(out_dir, 'summary.json')}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser("Tune train_new_structure.py with hard-coded promising grids.")
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="ICEWS14")
    parser.add_argument("--focus_test_metric", choices=sorted(METRIC_KEYS), default="MRR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--source_join_threads", type=int, default=60)
    parser.add_argument("--source_join_log_batches", type=int, default=0)
    parser.add_argument("--combine_query_batch", type=int, default=1024)
    parser.add_argument("--output_root", default="tuning_records_new_structure")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
