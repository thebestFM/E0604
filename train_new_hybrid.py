import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import numpy as np

import train_new_structure
from utils import (
    ScoreStore,
    add_rank_sums,
    collect_eval_batch,
    dense_rank,
    describe_loaded_data,
    finalize_metric_sums,
    is_run_complete,
    load_datasets,
    ranking_metric_key,
    save_config,
    save_metrics,
    set_random_seed,
)


EPS = 1e-12
FIT_PROTOCOL = "new_hybrid_optuna_v1"
EXPECTED_STRUCTURE_IMPL = "new_structure_v3"
DEFAULT_HYBRID_CONFIG = osp.join("configs", "new_hybrid_inputs.json")


STRUCTURE_CONFIGS = {
    "ICEWS14": [
        {
            "id": "s01_0609_rank1",
            "source_log": "icews14_add_dmh_c01_dsh0.95_tag_sum_a0.012_b0.92.log",
            "batch_size": 8192,
            "max_events_in_single_batch": 60000,
            "dict_mode": "tag_sum",
            "shared_w": "dual_msim",
            "per_rel_use_mtrans": False,
            "ppr_k": 1000,
            "top_k_relation": 0,
            "ppr_alpha": 0.012,
            "ppr_beta": 0.92,
            "gamma": 0.0,
            "direct_single_hop": 0.95,
            "decay_direct": 1.0,
            "top_share": 100,
            "top_direct": -1,
            "decay_rel_trans": 0.05,
            "window_semantic_sim": 5.0,
            "window_trans": 5.0,
            "close_update_backward": False,
        },
        {
            "id": "s02_0609_rank2",
            "source_log": "icews14_add_dmh_c01_dsh0.90_tag_sum_a0.012_b0.92.log",
            "batch_size": 8192,
            "max_events_in_single_batch": 60000,
            "dict_mode": "tag_sum",
            "shared_w": "dual_msim",
            "per_rel_use_mtrans": False,
            "ppr_k": 1000,
            "top_k_relation": 0,
            "ppr_alpha": 0.012,
            "ppr_beta": 0.92,
            "gamma": 0.0,
            "direct_single_hop": 0.90,
            "decay_direct": 1.0,
            "top_share": 100,
            "top_direct": -1,
            "decay_rel_trans": 0.05,
            "window_semantic_sim": 5.0,
            "window_trans": 5.0,
            "close_update_backward": False,
        },
        {
            "id": "s03_0609_rank3",
            "source_log": "icews14_add_dmh_c01_dsh0.80_tag_sum_a0.012_b0.92.log",
            "batch_size": 8192,
            "max_events_in_single_batch": 60000,
            "dict_mode": "tag_sum",
            "shared_w": "dual_msim",
            "per_rel_use_mtrans": False,
            "ppr_k": 1000,
            "top_k_relation": 0,
            "ppr_alpha": 0.012,
            "ppr_beta": 0.92,
            "gamma": 0.0,
            "direct_single_hop": 0.80,
            "decay_direct": 1.0,
            "top_share": 100,
            "top_direct": -1,
            "decay_rel_trans": 0.05,
            "window_semantic_sim": 5.0,
            "window_trans": 5.0,
            "close_update_backward": False,
        },
    ],
    "GDELT": [
        {
            "id": "s01_0609_rank1",
            "source_log": "gdelt_add_dmh_c02_dsh0.95_tag_sum_a0.012_b0.95.log",
            "batch_size": 8192,
            "max_events_in_single_batch": 60000,
            "dict_mode": "tag_sum",
            "shared_w": "dual_msim",
            "per_rel_use_mtrans": False,
            "ppr_k": 1000,
            "top_k_relation": 0,
            "ppr_alpha": 0.012,
            "ppr_beta": 0.95,
            "gamma": 0.0,
            "direct_single_hop": 0.95,
            "decay_direct": 0.1,
            "top_share": 100,
            "top_direct": -1,
            "decay_rel_trans": 0.05,
            "window_semantic_sim": 5.0,
            "window_trans": 5.0,
            "close_update_backward": False,
        },
        {
            "id": "s02_0609_rank2",
            "source_log": "gdelt_add_dmh_c01_dsh0.95_tag_sum_a0.006_b0.93.log",
            "batch_size": 8192,
            "max_events_in_single_batch": 60000,
            "dict_mode": "tag_sum",
            "shared_w": "dual_msim",
            "per_rel_use_mtrans": False,
            "ppr_k": 1000,
            "top_k_relation": 0,
            "ppr_alpha": 0.006,
            "ppr_beta": 0.93,
            "gamma": 0.0,
            "direct_single_hop": 0.95,
            "decay_direct": 0.1,
            "top_share": 100,
            "top_direct": -1,
            "decay_rel_trans": 0.05,
            "window_semantic_sim": 5.0,
            "window_trans": 5.0,
            "close_update_backward": False,
        },
        {
            "id": "s03_0609_rank3",
            "source_log": "gdelt_add_dmh_c04_dsh0.95_tag_sum_a0.025_b0.90.log",
            "batch_size": 8192,
            "max_events_in_single_batch": 60000,
            "dict_mode": "tag_sum",
            "shared_w": "dual_msim",
            "per_rel_use_mtrans": False,
            "ppr_k": 1000,
            "top_k_relation": 0,
            "ppr_alpha": 0.025,
            "ppr_beta": 0.90,
            "gamma": 0.0,
            "direct_single_hop": 0.95,
            "decay_direct": 0.1,
            "top_share": 100,
            "top_direct": -1,
            "decay_rel_trans": 0.05,
            "window_semantic_sim": 5.0,
            "window_trans": 5.0,
            "close_update_backward": False,
        },
    ],
    "tkgl-polecat": [
        {
            "id": "s01_0609_rank1",
            "source_log": "polecat_add_dsh_a01_dsh0.05_decay0.005.log",
            "batch_size": 4096,
            "max_events_in_single_batch": 60000,
            "dict_mode": "tag_sum",
            "shared_w": "dual_msim",
            "per_rel_use_mtrans": False,
            "ppr_k": 1000,
            "top_k_relation": 0,
            "ppr_alpha": 0.01579502319249557,
            "ppr_beta": 0.9343207039457382,
            "gamma": 0.01,
            "direct_single_hop": 0.05,
            "decay_direct": 0.005,
            "top_share": 100,
            "top_direct": -1,
            "decay_rel_trans": 0.05,
            "window_semantic_sim": 5.0,
            "window_trans": 5.0,
            "close_update_backward": False,
        },
        {
            "id": "s02_0609_rank2",
            "source_log": "polecat_add_dsh_a02_dsh0.05_decay0.01.log",
            "batch_size": 4096,
            "max_events_in_single_batch": 60000,
            "dict_mode": "tag_sum",
            "shared_w": "dual_msim",
            "per_rel_use_mtrans": False,
            "ppr_k": 1000,
            "top_k_relation": 0,
            "ppr_alpha": 0.01579502319249557,
            "ppr_beta": 0.9343207039457382,
            "gamma": 0.01,
            "direct_single_hop": 0.05,
            "decay_direct": 0.01,
            "top_share": 100,
            "top_direct": -1,
            "decay_rel_trans": 0.05,
            "window_semantic_sim": 5.0,
            "window_trans": 5.0,
            "close_update_backward": False,
        },
        {
            "id": "s03_0609_rank3",
            "source_log": "polecat_add_dsh_a03_dsh0.05_decay0.05.log",
            "batch_size": 4096,
            "max_events_in_single_batch": 60000,
            "dict_mode": "tag_sum",
            "shared_w": "dual_msim",
            "per_rel_use_mtrans": False,
            "ppr_k": 1000,
            "top_k_relation": 0,
            "ppr_alpha": 0.01579502319249557,
            "ppr_beta": 0.9343207039457382,
            "gamma": 0.01,
            "direct_single_hop": 0.05,
            "decay_direct": 0.05,
            "top_share": 100,
            "top_direct": -1,
            "decay_rel_trans": 0.05,
            "window_semantic_sim": 5.0,
            "window_trans": 5.0,
            "close_update_backward": False,
        },
    ],
}


TIME_RUNS = {
    "ICEWS14": [
        {
            "id": "time_cfg2_mrr",
            "dir": "results_time_tkg_single/ICEWS14/seed42/r9eb5b85515d8_topk30_mw5-15-30_ed96_hd192_bs4096_ebs384_neg6_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin",
            "test_mrr": 0.34213,
            "test_hit10": 0.53687,
        },
        {
            "id": "time_cfg1_mrr",
            "dir": "results_time_tkg_single/ICEWS14/seed42/r210529791eed_topk40_mw5-15-30-60_ed96_hd192_bs4096_ebs384_neg8_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin",
            "test_mrr": 0.34106,
            "test_hit10": 0.54054,
        },
        {
            "id": "time_cfg3_hr10",
            "dir": "results_time_tkg_single/ICEWS14/seed42/r041812cea350_topk70_mw5-15-30-60-120_ed96_hd192_bs4096_ebs384_neg4_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin",
            "test_mrr": 0.32153,
            "test_hit10": 0.54330,
        },
    ],
    "GDELT": [
        {
            "id": "time_mrr",
            "dir": "results_time_tkg_single/GDELT/seed42/rec9ebf5506ad_topk60_mw7-30_ed64_hd128_bs2048_ebs192_neg4_samgroup_nsq5000_nss42_tpr0.3_abs1r0_gateoff_rank1_lossmargin",
            "test_mrr": 0.23923,
            "test_hit10": 0.39306,
        },
    ],
    "tkgl-polecat": [
        {
            "id": "time_mrr",
            "dir": "results_time_tkg_single/tkgl-polecat/seed42/r59bae37154aa_topk80_mw30_ed64_hd128_bs2048_ebs128_neg2_samgroup_nsq5000_nss42_tpr0.3_abs1r0_gateoff_rank1_lossmargin",
            "test_mrr": 0.32686,
            "test_hit10": 0.52778,
        },
    ],
}


DATASET_COMMON = {
    "ICEWS14": {"ns_q": 6000, "batch_size": 8192},
    "GDELT": {"ns_q": 5000, "batch_size": 8192},
    "tkgl-polecat": {"ns_q": 5000, "batch_size": 4096},
}


def stable_hash(payload, length=12):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[: int(length)]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_hybrid_inputs(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "new_hybrid_inputs_v1":
        raise ValueError(f"unsupported hybrid input config format in {path}: {payload.get('format')!r}")
    for key in ("dataset_common", "structure_top_configs", "time_runs"):
        if key not in payload:
            raise ValueError(f"hybrid input config missing {key!r}: {path}")
    payload["_path"] = path
    payload["_hash"] = stable_hash(payload, length=12)
    return payload


def dataset_common_from_config(inputs, dataset):
    try:
        common = inputs["dataset_common"][dataset]
    except KeyError as exc:
        raise ValueError(f"dataset {dataset!r} is missing from hybrid input config") from exc
    return common


def structure_configs_from_config(inputs, dataset):
    configs = inputs["structure_top_configs"].get(dataset, [])
    if not configs:
        raise ValueError(f"no structure configs for dataset {dataset!r} in {inputs['_path']}")
    return configs


def time_runs_from_config(inputs, dataset):
    runs = inputs["time_runs"].get(dataset, [])
    if not runs:
        raise ValueError(f"no time runs for dataset {dataset!r} in {inputs['_path']}")
    return runs


def minmax_by_query(scores, valid):
    scores = sanitize_score_matrix(scores, valid)
    low = np.min(np.where(valid, scores, np.inf), axis=1, keepdims=True)
    high = np.max(np.where(valid, scores, -np.inf), axis=1, keepdims=True)
    denom = np.maximum(high - low, EPS)
    out = (scores - low) / denom
    return np.where(valid, out, 0.0).astype(np.float32, copy=False)


def zscore_by_query(scores, valid):
    scores = sanitize_score_matrix(scores, valid)
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.sum(np.where(valid, scores, 0.0), axis=1, keepdims=True) / count
    var = np.sum(np.where(valid, (scores - mean) ** 2, 0.0), axis=1, keepdims=True) / count
    return np.where(valid, (scores - mean) / np.sqrt(np.maximum(var, EPS)), 0.0).astype(np.float32, copy=False)


def sanitize_score_matrix(scores, valid):
    scores = np.asarray(scores, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if np.all(np.isfinite(np.where(valid, scores, 0.0))):
        return np.where(valid, scores, 0.0).astype(np.float32, copy=False)

    out = np.where(valid, scores, 0.0).astype(np.float32, copy=True)
    finite_valid = valid & np.isfinite(out)
    for row in range(out.shape[0]):
        row_valid = valid[row]
        if not np.any(row_valid):
            continue
        row_finite = finite_valid[row]
        if np.any(row_finite):
            vals = out[row, row_finite]
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            span = max(hi - lo, 1.0)
            pos_fill = hi + span
            neg_fill = lo - span
        else:
            pos_fill = 1.0
            neg_fill = -1.0
        pos_inf = row_valid & np.isposinf(out[row])
        neg_inf = row_valid & np.isneginf(out[row])
        nan_mask = row_valid & np.isnan(out[row])
        out[row, pos_inf] = pos_fill
        out[row, neg_inf] = neg_fill
        out[row, nan_mask] = neg_fill
    return np.where(valid, out, 0.0).astype(np.float32, copy=False)


def make_base_scores(struct_scores, time_scores, valid):
    return ((minmax_by_query(struct_scores, valid) + minmax_by_query(time_scores, valid)) * 0.5).astype(
        np.float32, copy=False
    )


def metric_value(metrics, metric):
    return float(metrics[ranking_metric_key(metric, strict=True)])


def format_metric_line(metrics):
    return (
        f"mrr={metrics['mrr_strict']:.5f} "
        f"hr1={metrics['hit@1_strict']:.5f} "
        f"hr10={metrics['hit@10_strict']:.5f}"
    )


def evaluate_score_store(out_dir, label, data, args, split="test"):
    store = ScoreStore(out_dir, split)
    expected_width = int(args.ns_q)
    sums = {}
    row_offset = 0
    nonfinite = 0
    for events, _, t_orig in split_snapshots(data, split):
        for batch_data, neg_arr, _ in collect_eval_batch(
            events,
            t_orig,
            data["negative_sampler"],
            split,
            int(args.query_batch_size),
        ):
            batch_size = int(len(batch_data))
            width = int(neg_arr.shape[1])
            if width != expected_width:
                raise RuntimeError(
                    f"{label} {split} negative width mismatch at row {row_offset}: "
                    f"got={width} expected={expected_width}"
                )
            pos, neg, neg_mask = store.get_block(row_offset, row_offset + batch_size, width)
            valid = np.concatenate(
                (
                    np.ones((batch_size, 1), dtype=bool),
                    neg_mask,
                ),
                axis=1,
            )
            raw_scores = np.concatenate((pos, neg), axis=1).astype(np.float32, copy=False)
            nonfinite += int(np.size(raw_scores[valid]) - np.sum(np.isfinite(raw_scores[valid])))
            scores = sanitize_score_matrix(raw_scores, valid)

            loose = []
            strict = []
            avg = []
            for row in range(batch_size):
                row_valid = valid[row]
                row_scores = scores[row, row_valid]
                pos_score = row_scores[0]
                neg_scores = row_scores[1:]
                l_rank = 1 + int(np.sum(neg_scores > pos_score))
                s_rank = 1 + int(np.sum(neg_scores >= pos_score))
                loose.append(l_rank)
                strict.append(s_rank)
                avg.append((l_rank + s_rank) * 0.5)
            add_rank_sums(
                sums,
                np.asarray(loose, dtype=np.int64),
                np.asarray(strict, dtype=np.int64),
                np.asarray(avg, dtype=np.float64),
            )
            row_offset += batch_size

    if row_offset != store.num_rows:
        raise RuntimeError(f"{label} {split} row count mismatch: streamed={row_offset} store={store.num_rows}")
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    metrics["nonfinite_scores"] = int(nonfinite)
    return metrics


def print_score_store_baseline(out_dir, label, data, args, split="test"):
    metrics = evaluate_score_store(out_dir, label, data, args, split=split)
    print(
        f"[NewHybrid][baseline] {label} {split}: {format_metric_line(metrics)} "
        f"queries={metrics['num_queries']} nonfinite_scores={metrics['nonfinite_scores']}",
        flush=True,
    )
    return metrics


class NewHybridFeatureBuilder:
    def __init__(self, num_rels):
        self.num_rels = int(num_rels)
        self.feature_names = []
        self._init_names()

    def _add(self, name):
        self.feature_names.append(name)

    def _init_names(self):
        for prefix in ("struct", "time", "base"):
            self._add(f"{prefix}_score")
            self._add(f"{prefix}_z")
            self._add(f"{prefix}_minmax")
            self._add(f"{prefix}_rank_log")
            self._add(f"{prefix}_rank_recip")
        for name in (
            "struct_minus_time",
            "abs_struct_minus_time",
            "struct_times_time",
            "score_mean",
            "score_max",
            "score_min",
            "rank_min",
            "rank_gap",
            "both_top10",
            "either_top10",
            "both_top50",
            "either_top50",
            "struct_top1",
            "time_top1",
            "base_top1",
            "relation_is_inverse",
            "candidate_is_source",
        ):
            self._add(name)

    def make(self, struct_scores, time_scores, base_scores, valid, batch_data, cand_ids):
        ranks = {
            "struct": dense_rank(struct_scores, valid),
            "time": dense_rank(time_scores, valid),
            "base": dense_rank(base_scores, valid),
        }
        score_map = {"struct": struct_scores, "time": time_scores, "base": base_scores}
        features = []
        for prefix in ("struct", "time", "base"):
            score = np.where(valid, score_map[prefix], 0.0).astype(np.float32, copy=False)
            rank = ranks[prefix].astype(np.float32, copy=False)
            features.extend(
                [
                    score,
                    zscore_by_query(score, valid),
                    minmax_by_query(score, valid),
                    np.log1p(rank).astype(np.float32, copy=False),
                    (1.0 / np.maximum(rank, 1.0)).astype(np.float32, copy=False),
                ]
            )

        struct_rank = ranks["struct"].astype(np.float32, copy=False)
        time_rank = ranks["time"].astype(np.float32, copy=False)
        features.extend(
            [
                (struct_scores - time_scores).astype(np.float32, copy=False),
                np.abs(struct_scores - time_scores).astype(np.float32, copy=False),
                (struct_scores * time_scores).astype(np.float32, copy=False),
                ((struct_scores + time_scores) * 0.5).astype(np.float32, copy=False),
                np.maximum(struct_scores, time_scores).astype(np.float32, copy=False),
                np.minimum(struct_scores, time_scores).astype(np.float32, copy=False),
                np.minimum(struct_rank, time_rank).astype(np.float32, copy=False),
                np.abs(struct_rank - time_rank).astype(np.float32, copy=False),
                ((struct_rank <= 10) & (time_rank <= 10)).astype(np.float32),
                ((struct_rank <= 10) | (time_rank <= 10)).astype(np.float32),
                ((struct_rank <= 50) & (time_rank <= 50)).astype(np.float32),
                ((struct_rank <= 50) | (time_rank <= 50)).astype(np.float32),
                (struct_rank <= 1).astype(np.float32),
                (time_rank <= 1).astype(np.float32),
                (ranks["base"].astype(np.float32, copy=False) <= 1).astype(np.float32),
            ]
        )
        rels = batch_data[:, 1].astype(np.int64, copy=False)
        sources = batch_data[:, 0].astype(np.int64, copy=False)
        features.append((rels.reshape(-1, 1) >= self.num_rels // 2).repeat(valid.shape[1], axis=1).astype(np.float32))
        features.append((cand_ids == sources.reshape(-1, 1)).astype(np.float32, copy=False))
        return np.stack(features, axis=2).astype(np.float32, copy=False)


def candidate_scores(pos, neg, valid):
    scores = np.concatenate((pos, neg), axis=1).astype(np.float32, copy=False)
    return sanitize_score_matrix(scores, valid)


def selected_for_training(struct_scores, time_scores, base_scores, valid, topk):
    if int(topk) < 0:
        return valid.copy()
    struct_rank = dense_rank(struct_scores, valid)
    time_rank = dense_rank(time_scores, valid)
    base_rank = dense_rank(base_scores, valid)
    selected = (
        ((struct_rank <= int(topk) + 1) & valid)
        | ((time_rank <= int(topk) + 1) & valid)
        | ((base_rank <= int(topk) + 1) & valid)
    )
    selected[:, 0] = True
    return selected & valid


def split_snapshots(data, split):
    if split == "train":
        return data["train_list"][data["train_predict_start_idx"] :]
    if split == "val":
        return data["val_list"]
    if split == "test":
        return data["test_list"]
    raise ValueError(split)


def split_query_count(data, split):
    return int(sum(len(events) for events, _, _ in split_snapshots(data, split)))


def iter_hybrid_blocks(context, split, args):
    struct_store = ScoreStore(context.struct_dir, split)
    time_store = ScoreStore(context.time_dir, split)
    snapshots = split_snapshots(context.data, split)
    neg_sampler = context.data["negative_sampler"]
    row_offset = 0

    for events, _, t_orig in snapshots:
        for batch_data, neg_arr, neg_mask in collect_eval_batch(events, t_orig, neg_sampler, split, args.query_batch_size):
            width = int(neg_arr.shape[1])
            end = row_offset + len(batch_data)
            struct_pos, struct_neg, struct_mask = struct_store.get_block(row_offset, end, width)
            time_pos, time_neg, time_mask = time_store.get_block(row_offset, end, width)
            valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), neg_mask), axis=1)
            valid &= np.concatenate((np.ones((len(batch_data), 1), dtype=bool), struct_mask), axis=1)
            valid &= np.concatenate((np.ones((len(batch_data), 1), dtype=bool), time_mask), axis=1)
            cand_ids = np.concatenate((batch_data[:, 2:3], neg_arr), axis=1)
            cand_ids = np.where(valid, cand_ids, -1).astype(np.int64, copy=False)
            struct_scores = candidate_scores(struct_pos, struct_neg, valid)
            time_scores = candidate_scores(time_pos, time_neg, valid)
            base_scores = make_base_scores(struct_scores, time_scores, valid)
            features = context.feature_builder.make(
                struct_scores,
                time_scores,
                base_scores,
                valid,
                batch_data=batch_data,
                cand_ids=cand_ids,
            )
            yield valid, struct_scores, time_scores, base_scores, features
            row_offset = end

    if row_offset != struct_store.num_rows:
        raise ValueError(f"structure row count mismatch for {split}: stream={row_offset}, store={struct_store.num_rows}")
    if row_offset != time_store.num_rows:
        raise ValueError(f"time row count mismatch for {split}: stream={row_offset}, store={time_store.num_rows}")


def build_ranker_matrix(context, split, args, topk, query_start=0, query_stop=None, require_full_width=False):
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0
    seen_queries = 0
    expected_width = int(args.ns_q) + 1 if int(args.ns_q) > 0 else None

    for valid, struct_scores, time_scores, base_scores, features in iter_hybrid_blocks(context, split, args):
        selected = selected_for_training(struct_scores, time_scores, base_scores, valid, topk)
        for row in range(selected.shape[0]):
            query_idx = seen_queries
            seen_queries += 1
            if query_idx < int(query_start):
                continue
            if query_stop is not None and query_idx >= int(query_stop):
                continue
            cols = np.flatnonzero(selected[row])
            if require_full_width and expected_width is not None and len(cols) != expected_width:
                raise ValueError(
                    f"{split} query {query_idx} has {len(cols)} valid candidates, expected {expected_width}"
                )
            if len(cols) <= 1:
                continue
            labels = np.zeros(len(cols), dtype=np.float32)
            pos = np.flatnonzero(cols == 0)
            if len(pos) != 1:
                raise ValueError(f"{split} query {query_idx} missing positive candidate")
            labels[int(pos[0])] = 1.0
            X_parts.append(features[row, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)

    if not X_parts:
        raise ValueError(f"no {split} matrix rows built")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {
            "split": split,
            "queries": int(queries),
            "rows": int(rows),
            "source_queries_seen": int(seen_queries),
            "selection": "all_candidates" if int(topk) < 0 else "struct_time_base_topk",
            "topk": int(topk),
            "query_start": int(query_start),
            "query_stop": None if query_stop is None else int(query_stop),
        },
    )


def build_train_matrix(context, args):
    return build_ranker_matrix(context, "train", args, int(args.top_hybrid_train))


def build_lgbm_eval_matrix(context, args):
    fraction = float(args.lgbm_eval_tail_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--lgbm_eval_tail_fraction must be in (0, 1]")
    total = split_query_count(context.data, "val")
    start = int(total * (1.0 - fraction))
    topk = int(args.lgbm_eval_topk)
    X, y, group, info = build_ranker_matrix(context, "val", args, topk, query_start=start)
    info.update(
        {
            "mode": "val_tail_lgbm_eval_set",
            "tail_fraction": fraction,
            "total_val_queries": int(total),
        }
    )
    return (X, y, group), info


def evaluate_hybrid(context, model, split, args):
    sums = {}
    query_idx = 0
    expected_width = int(args.ns_q) + 1 if split in ("val", "test") and int(args.ns_q) > 0 else None
    for valid, _, _, _, features in iter_hybrid_blocks(context, split, args):
        X_parts = []
        slices = []
        cursor = 0
        for row in range(valid.shape[0]):
            cols = np.flatnonzero(valid[row])
            if expected_width is not None and len(cols) != expected_width:
                raise ValueError(
                    f"{split} query {query_idx} has {len(cols)} valid candidates, expected {expected_width}"
                )
            query_idx += 1
            X_parts.append(features[row, cols, :])
            slices.append((cursor, cursor + len(cols)))
            cursor += len(cols)

        if not X_parts:
            continue
        X = np.vstack(X_parts).astype(np.float32, copy=False)
        pred = model.predict(X).astype(np.float32, copy=False)
        if not np.all(np.isfinite(pred)):
            bad = int(np.size(pred) - np.sum(np.isfinite(pred)))
            raise RuntimeError(f"{split} hybrid prediction has {bad} non-finite values")
        loose = []
        strict = []
        avg = []
        for start, end in slices:
            scores = pred[start:end]
            pos_score = scores[0]
            neg = scores[1:]
            l_rank = 1 + int(np.sum(neg > pos_score))
            s_rank = 1 + int(np.sum(neg >= pos_score))
            loose.append(l_rank)
            strict.append(s_rank)
            avg.append((l_rank + s_rank) * 0.5)
        add_rank_sums(
            sums,
            np.asarray(loose, dtype=np.int64),
            np.asarray(strict, dtype=np.int64),
            np.asarray(avg, dtype=np.float64),
        )
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    return metrics


def evaluate_flat_predictions(pred, group):
    pred = np.asarray(pred, dtype=np.float32)
    group = np.asarray(group, dtype=np.int32)
    if not np.all(np.isfinite(pred)):
        bad = int(np.size(pred) - np.sum(np.isfinite(pred)))
        raise RuntimeError(f"flat hybrid prediction has {bad} non-finite values")
    loose = []
    strict = []
    avg = []
    cursor = 0
    for g in group:
        end = cursor + int(g)
        scores = pred[cursor:end]
        if len(scores) <= 1:
            cursor = end
            continue
        pos_score = scores[0]
        neg = scores[1:]
        l_rank = 1 + int(np.sum(neg > pos_score))
        s_rank = 1 + int(np.sum(neg >= pos_score))
        loose.append(l_rank)
        strict.append(s_rank)
        avg.append((l_rank + s_rank) * 0.5)
        cursor = end
    sums = {}
    add_rank_sums(
        sums,
        np.asarray(loose, dtype=np.int64),
        np.asarray(strict, dtype=np.int64),
        np.asarray(avg, dtype=np.float64),
    )
    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    return metrics


def fit_lgbm_ranker(X, y, group, feature_names, args, params, eval_data=None):
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("train_new_hybrid.py requires lightgbm") from exc

    use_custom_hit10_eval = eval_data is not None
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="None" if use_custom_hit10_eval else "ndcg",
        boosting_type="gbdt",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        min_split_gain=float(params["min_split_gain"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        random_state=int(args.seed),
        n_jobs=int(args.num_threads),
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    fit_kwargs = {"group": group.tolist(), "feature_name": feature_names}
    if eval_data is not None:
        X_val, y_val, group_val = eval_data

        def hit10_eval(y_true, y_pred, *unused):
            metrics = evaluate_flat_predictions(np.asarray(y_pred, dtype=np.float32), group_val)
            return "hit10", float(metrics["hit@10_strict"]), True

        fit_kwargs.update(
            {
                "eval_set": [(X_val, y_val)],
                "eval_group": [group_val.tolist()],
                "eval_at": [10],
                "eval_metric": hit10_eval,
                "callbacks": [
                    lgb.early_stopping(
                        int(args.lgbm_early_stopping_rounds),
                        verbose=False,
                    )
                ],
            }
        )
    model.fit(X, y, **fit_kwargs)
    return model


def matrix_diagnostics(name, X, y, group):
    nonfinite = int(np.size(X) - np.sum(np.isfinite(X)))
    positives = int(np.sum(y > 0.0))
    rows = int(X.shape[0])
    queries = int(len(group))
    group_min = int(np.min(group)) if len(group) else 0
    group_max = int(np.max(group)) if len(group) else 0
    print(
        f"[NewHybrid] {name} diagnostics: rows={rows} queries={queries} "
        f"positives={positives} nonfinite_features={nonfinite} "
        f"group_min={group_min} group_max={group_max}",
        flush=True,
    )
    if positives != queries:
        raise RuntimeError(f"{name} matrix expected one positive per query, got positives={positives}, queries={queries}")
    if nonfinite:
        raise RuntimeError(f"{name} matrix has {nonfinite} non-finite feature values")


def save_lgbm_model(model, path):
    booster = getattr(model, "booster_", model)
    booster.save_model(path)


def pair_key(args, struct_run, time_run):
    return stable_hash(
        {
            "fit_protocol": FIT_PROTOCOL,
            "dataset": args.dataset,
            "config_hash": getattr(args, "config_hash", ""),
            "seed": args.seed,
            "ns_q": args.ns_q,
            "ns_seed": args.ns_seed,
            "train_predict_ratio": args.train_predict_ratio,
            "top_hybrid_train": args.top_hybrid_train,
            "lgbm_eval_topk": args.lgbm_eval_topk,
            "lgbm_eval_tail_fraction": args.lgbm_eval_tail_fraction,
            "lgbm_early_stopping_rounds": args.lgbm_early_stopping_rounds,
            "n_trials": args.n_trials,
            "focus_test_metric": args.focus_test_metric,
            "struct_id": struct_run["id"],
            "struct_dir": struct_run["dir"],
            "time_id": time_run["id"],
            "time_dir": time_run["dir"],
        },
        length=16,
    )


def make_out_dir(args):
    h = stable_hash(
        {
            "fit_protocol": FIT_PROTOCOL,
            "dataset": args.dataset,
            "config_hash": getattr(args, "config_hash", ""),
            "seed": args.seed,
            "ns_q": args.ns_q,
            "ns_seed": args.ns_seed,
            "train_predict_ratio": args.train_predict_ratio,
            "top_hybrid_train": args.top_hybrid_train,
            "lgbm_eval_topk": args.lgbm_eval_topk,
            "lgbm_eval_tail_fraction": args.lgbm_eval_tail_fraction,
            "lgbm_early_stopping_rounds": args.lgbm_early_stopping_rounds,
            "n_trials": args.n_trials,
            "focus_test_metric": args.focus_test_metric,
        },
        length=12,
    )
    return osp.join(args.output_root, args.dataset, f"seed{args.seed}", f"new_hybrid_{h}")


def cache_dir(out_dir, key):
    return osp.join(out_dir, "pair_cache", f"pair-{key}")


def load_pair_cache(out_dir, key):
    path = osp.join(cache_dir(out_dir, key), "record.json")
    model_path = osp.join(cache_dir(out_dir, key), "model.txt")
    if not osp.isfile(path) or not osp.isfile(model_path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "new_hybrid_pair_v3" or payload.get("pair_key") != key:
        return None
    return payload["record"], model_path


def save_pair_cache(out_dir, key, record, model):
    cdir = ensure_dir(cache_dir(out_dir, key))
    model_path = osp.join(cdir, "model.txt")
    save_lgbm_model(model, model_path)
    with open(osp.join(cdir, "record.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "new_hybrid_pair_v3",
                "pair_key": key,
                "record": record,
                "model_path": model_path,
            },
            f,
            indent=2,
        )


def complete_score_dir(out_dir):
    return is_run_complete(out_dir, modes=("train", "val", "test"))


def require_score_dir(out_dir, label):
    missing = []
    for split in ("train", "val", "test"):
        for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json"):
            path = osp.join(out_dir, f"{split}_{suffix}")
            if not osp.isfile(path):
                missing.append(path)
    if missing:
        text = "\n  ".join(missing[:12])
        extra = "" if len(missing) <= 12 else f"\n  ... and {len(missing) - 12} more"
        raise FileNotFoundError(f"{label} score files are incomplete under {out_dir}:\n  {text}{extra}")


def require_train_new_structure_impl():
    impl = getattr(train_new_structure, "NEW_STRUCTURE_IMPL", None)
    if impl is not None:
        if impl != EXPECTED_STRUCTURE_IMPL:
            raise RuntimeError(
                "train_new_hybrid.py requires train_new_structure.NEW_STRUCTURE_IMPL "
                f"to be {EXPECTED_STRUCTURE_IMPL!r}, got {impl!r}"
            )
        return

    try:
        source = inspect.getsource(train_new_structure.make_new_result_dir)
    except (OSError, TypeError) as exc:
        raise RuntimeError("cannot inspect train_new_structure.make_new_result_dir") from exc
    if EXPECTED_STRUCTURE_IMPL not in source:
        raise RuntimeError(
            "train_new_hybrid.py requires train_new_structure.make_new_result_dir "
            f"to use impl={EXPECTED_STRUCTURE_IMPL!r}"
        )


def assert_score_store_finite(out_dir, label, splits=("train", "val", "test")):
    for split in splits:
        store = ScoreStore(out_dir, split)
        pos_nonfinite = int(np.size(store.pos) - np.sum(np.isfinite(store.pos)))
        neg_data = store.neg.data
        neg_nonfinite = int(np.size(neg_data) - np.sum(np.isfinite(neg_data)))
        if pos_nonfinite or neg_nonfinite:
            raise RuntimeError(
                f"{label} {split} has non-finite raw scores: "
                f"pos_nonfinite={pos_nonfinite} neg_nonfinite={neg_nonfinite} dir={out_dir}"
            )


def inspect_score_store_against_data(out_dir, label, data, args, sample_queries=None):
    expected_negs = int(args.ns_q)
    if expected_negs <= 0:
        raise ValueError("train_new_hybrid requires sampled negatives with ns_q > 0")
    sample_queries = int(getattr(args, "score_check_queries", 1) if sample_queries is None else sample_queries)
    sample_queries = max(1, sample_queries)

    for split in ("train", "val", "test"):
        expected_rows = split_query_count(data, split)
        store = ScoreStore(out_dir, split)
        meta_path = osp.join(out_dir, f"{split}_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if int(meta.get("num_rows", -1)) != expected_rows:
            raise RuntimeError(
                f"{label} {split} meta num_rows mismatch: "
                f"meta={meta.get('num_rows')} expected={expected_rows}"
            )
        if store.num_rows != expected_rows:
            raise RuntimeError(
                f"{label} {split} score rows mismatch: store={store.num_rows} expected={expected_rows}"
            )
        if store.pos.shape != (expected_rows, 1):
            raise RuntimeError(
                f"{label} {split} pos shape mismatch: got={store.pos.shape} expected={(expected_rows, 1)}"
            )
        if int(store.max_negs) < expected_negs:
            raise RuntimeError(
                f"{label} {split} max_negs mismatch: got={store.max_negs} expected at least {expected_negs}"
            )
        if store.valid_lens.shape[0] != expected_rows:
            raise RuntimeError(
                f"{label} {split} valid_lens rows mismatch: "
                f"got={store.valid_lens.shape[0]} expected={expected_rows}"
            )
        bad_lens = int(np.sum(store.valid_lens != expected_negs))
        if bad_lens:
            examples = store.valid_lens[store.valid_lens != expected_negs][:8].tolist()
            raise RuntimeError(
                f"{label} {split} valid_lens mismatch: {bad_lens} rows are not ns_q={expected_negs}; "
                f"examples={examples}"
            )

        sampled = 0
        sampled_nonfinite = 0
        row_offset = 0
        neg_sampler = data["negative_sampler"]
        for events, _, t_orig in split_snapshots(data, split):
            for batch_data, neg_arr, neg_mask in collect_eval_batch(
                events,
                t_orig,
                neg_sampler,
                split,
                max(1, min(sample_queries, int(args.query_batch_size))),
            ):
                width = int(neg_arr.shape[1])
                if width != expected_negs:
                    raise RuntimeError(
                        f"{label} {split} sampled negative width mismatch at row {row_offset}: "
                        f"got={width} expected={expected_negs}"
                    )
                take = min(int(len(batch_data)), sample_queries - sampled)
                pos, neg, mask = store.get_block(row_offset, row_offset + take, width)
                if pos.shape != (take, 1) or neg.shape != (take, expected_negs) or mask.shape != (take, expected_negs):
                    raise RuntimeError(
                        f"{label} {split} sampled block shape mismatch at row {row_offset}: "
                        f"pos={pos.shape} neg={neg.shape} mask={mask.shape}"
                    )
                if not np.all(mask):
                    raise RuntimeError(f"{label} {split} sampled block has invalid negatives at row {row_offset}")
                sampled_nonfinite += int(np.size(pos) - np.sum(np.isfinite(pos)))
                sampled_nonfinite += int(np.size(neg[mask]) - np.sum(np.isfinite(neg[mask])))
                sampled += take
                row_offset += int(len(batch_data))
                if sampled >= sample_queries:
                    break
            if sampled >= sample_queries:
                break

        if expected_rows > 0 and sampled == 0:
            raise RuntimeError(f"{label} {split} expected rows but sampled no query for format check")
        print(
            f"[NewHybrid] verified {label} {split}: rows={expected_rows} "
            f"negatives_per_query={expected_negs} sampled={sampled} "
            f"sampled_nonfinite_scores={sampled_nonfinite}",
            flush=True,
        )


def make_structure_args(args, cfg):
    common = args.dataset_common
    payload = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "gpu": int(args.gpu),
        "ns_q": int(common["ns_q"]),
        "ns_seed": int(args.ns_seed),
        "train_predict_ratio": float(args.train_predict_ratio),
        "batch_size": int(cfg.get("batch_size", common["batch_size"])),
        "max_events_in_single_batch": int(cfg["max_events_in_single_batch"]),
        "source_join_threads": int(args.source_join_threads),
        "source_join_log_batches": int(args.source_join_log_batches),
        "close_update_backward": bool(cfg.get("close_update_backward", False)),
        "dict_mode": cfg["dict_mode"],
        "shared_w": cfg["shared_w"],
        "per_rel_use_mtrans": bool(cfg["per_rel_use_mtrans"]),
        "ppr_k": int(cfg["ppr_k"]),
        "top_k_relation": int(cfg["top_k_relation"]),
        "ppr_alpha": float(cfg["ppr_alpha"]),
        "ppr_beta": float(cfg["ppr_beta"]),
        "gamma": float(cfg["gamma"]),
        "direct_single_hop": float(cfg["direct_single_hop"]),
        "decay_direct": float(cfg["decay_direct"]),
        "top_share": int(cfg["top_share"]),
        "top_direct": int(cfg["top_direct"]),
        "decay_rel_trans": float(cfg["decay_rel_trans"]),
        "window_semantic_sim": float(cfg["window_semantic_sim"]),
        "window_trans": float(cfg["window_trans"]),
        "skip_val_eval": False,
        "output_root": args.structure_output_root,
    }
    return argparse.Namespace(**payload)


def prepare_structure_runs(args, inputs):
    require_train_new_structure_impl()
    runs = []
    for cfg in structure_configs_from_config(inputs, args.dataset):
        sargs = make_structure_args(args, cfg)
        out_dir = train_new_structure.make_new_result_dir(sargs)
        print(f"[NewHybrid] structure {cfg['id']} -> {out_dir}", flush=True)
        if not complete_score_dir(out_dir):
            print(
                f"[NewHybrid] running full structure inference for {cfg['id']} "
                f"into hybrid-only dir {out_dir}",
                flush=True,
            )
            train_new_structure.main(sargs)
        require_score_dir(out_dir, f"structure {cfg['id']}")
        assert_score_store_finite(out_dir, f"structure {cfg['id']}")
        run = copy.deepcopy(cfg)
        run["dir"] = out_dir
        run["args"] = vars(sargs)
        runs.append(run)
    return runs


def prepare_time_runs(args, data, inputs):
    runs = []
    for raw in time_runs_from_config(inputs, args.dataset):
        run = copy.deepcopy(raw)
        if args.time_root:
            rel = raw["dir"]
            parts = rel.replace("\\", "/").split("/")
            if parts and parts[0] == "results_time_tkg_single":
                rel = "/".join(parts[1:])
            run["dir"] = osp.join(args.time_root, rel)
        require_score_dir(run["dir"], f"time {run['id']}")
        assert_score_store_finite(run["dir"], f"time {run['id']}")
        inspect_score_store_against_data(run["dir"], f"time {run['id']}", data, args)
        run["test_metrics_computed"] = print_score_store_baseline(
            run["dir"],
            f"time {run['id']}",
            data,
            args,
            split="test",
        )
        runs.append(run)
        print(f"[NewHybrid] time {run['id']} -> {run['dir']}", flush=True)
    return runs


def verify_structure_runs(struct_runs, data, args):
    for run in struct_runs:
        require_score_dir(run["dir"], f"structure {run['id']}")
        assert_score_store_finite(run["dir"], f"structure {run['id']}")
        inspect_score_store_against_data(run["dir"], f"structure {run['id']}", data, args)
        run["test_metrics_computed"] = print_score_store_baseline(
            run["dir"],
            f"structure {run['id']}",
            data,
            args,
            split="test",
        )


def make_context(args, data, struct_run, time_run):
    return SimpleNamespace(
        data=data,
        struct=struct_run,
        time=time_run,
        struct_dir=struct_run["dir"],
        time_dir=time_run["dir"],
        feature_builder=NewHybridFeatureBuilder(data["num_rels"]),
    )


def suggest_lgbm_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [7, 15, 23, 31, 47, 63, 127]),
        "max_depth": trial.suggest_int("max_depth", 2, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 1000, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
    }


def tune_pair(context, args):
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("train_new_hybrid.py requires optuna for the current tuning mode") from exc

    print("[NewHybrid] building train matrix", flush=True)
    X_train, y_train, group_train, train_info = build_train_matrix(context, args)
    matrix_diagnostics("train", X_train, y_train, group_train)
    print(
        f"[NewHybrid] train matrix queries={train_info['queries']} rows={train_info['rows']} "
        f"features={X_train.shape[1]} topk={train_info['topk']}",
        flush=True,
    )
    print("[NewHybrid] building val-tail matrix for early stopping", flush=True)
    (X_val, y_val, group_val), eval_info = build_lgbm_eval_matrix(context, args)
    matrix_diagnostics("val_tail", X_val, y_val, group_val)
    print(
        f"[NewHybrid] val-tail matrix queries={eval_info['queries']} rows={eval_info['rows']} "
        f"tail_fraction={eval_info['tail_fraction']} topk={eval_info['topk']}",
        flush=True,
    )

    best = {
        "score": -float("inf"),
        "model": None,
        "params": None,
        "test_metrics": None,
        "val_tail_metrics": None,
        "best_iteration": None,
        "trial": None,
    }
    trial_records = []

    def objective(trial):
        params = suggest_lgbm_params(trial)
        model = fit_lgbm_ranker(
            X_train,
            y_train,
            group_train,
            context.feature_builder.feature_names,
            args,
            params,
            eval_data=(X_val, y_val, group_val),
        )
        val_pred = model.predict(X_val).astype(np.float32, copy=False)
        val_tail_metrics = evaluate_flat_predictions(val_pred, group_val)
        test_metrics = evaluate_hybrid(context, model, "test", args)
        score = metric_value(test_metrics, args.focus_test_metric)
        best_iteration = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
        print(
            f"[NewHybrid][trial {trial.number}] "
            f"val_tail_mrr={val_tail_metrics['mrr_strict']:.5f} "
            f"val_tail_hr10={val_tail_metrics['hit@10_strict']:.5f} "
            f"test_mrr={test_metrics['mrr_strict']:.5f} "
            f"test_hr1={test_metrics['hit@1_strict']:.5f} "
            f"test_hr10={test_metrics['hit@10_strict']:.5f} "
            f"objective_{args.focus_test_metric}={score:.5f} "
            f"best_iteration={best_iteration} params={params}",
            flush=True,
        )
        record = (
            {
                "trial": int(trial.number),
                "params": dict(params),
                "best_iteration": int(best_iteration),
                "selection_score": float(score),
                "val_tail_metrics": val_tail_metrics,
                "test_metrics": test_metrics,
            }
        )
        trial_records.append(record)
        if score > best["score"]:
            if best["model"] is not None:
                del best["model"]
                gc.collect()
            best.update(
                {
                    "score": float(score),
                    "model": model,
                    "params": dict(params),
                    "test_metrics": test_metrics,
                    "val_tail_metrics": val_tail_metrics,
                    "best_iteration": int(best_iteration),
                    "trial": int(trial.number),
                }
            )
        else:
            del model
            gc.collect()
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(args.n_trials), gc_after_trial=True)
    if best["model"] is None:
        raise RuntimeError("Optuna produced no valid LGBM model")

    return best["model"], {
        "format": "new_hybrid_pair_result_v1",
        "fit_protocol": FIT_PROTOCOL,
        "struct_id": context.struct["id"],
        "struct_dir": context.struct_dir,
        "struct_source_log": context.struct.get("source_log"),
        "time_id": context.time["id"],
        "time_dir": context.time_dir,
        "time_reference_metrics": {
            "test_mrr": context.time.get("test_mrr"),
            "test_hit10": context.time.get("test_hit10"),
        },
        "selection_metric": args.focus_test_metric,
        "selection_score": float(best["score"]),
        "best_trial": int(best["trial"]),
        "best_iteration": int(best["best_iteration"]),
        "best_params": best["params"],
        "trial_records": trial_records,
        "optuna_best_trial": int(study.best_trial.number),
        "optuna_best_value": float(study.best_value),
        "train_info": train_info,
        "eval_info": eval_info,
        "train_metrics": None,
        "val_tail_metrics": best["val_tail_metrics"],
        "test_metrics": best["test_metrics"],
        "objective_split": "test",
        "uses_early_stopping": True,
        "early_stopping_eval": "val_tail_last_fraction",
        "early_stopping_native_metric": "custom_hit@10",
    }


def normalize_gpu_arg(args):
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    visible_count = int(torch.cuda.device_count())
    requested = int(args.gpu)
    if requested < 0:
        raise ValueError("--gpu must be >= 0")
    if requested < visible_count:
        return
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_count == 1:
        print(
            f"[NewHybrid] requested --gpu {requested}, but only one CUDA device is visible "
            f"(CUDA_VISIBLE_DEVICES={visible_env!r}); using local cuda:0",
            flush=True,
        )
        args.gpu = 0
        return
    raise ValueError(
        f"--gpu {requested} is invalid because torch sees {visible_count} CUDA devices "
        f"(CUDA_VISIBLE_DEVICES={visible_env!r}); use a local ordinal in [0, {visible_count - 1}]"
    )


def validate_args(args, inputs):
    if args.dataset not in inputs["dataset_common"]:
        raise ValueError(f"--dataset must be one of {sorted(inputs['dataset_common'])}")
    args.dataset_common = dataset_common_from_config(inputs, args.dataset)
    args.structure_output_root = getattr(args, "structure_output_root", "") or inputs.get(
        "structure_output_root", "results_new_hybrid_structure"
    )
    args.ns_q = int(args.dataset_common["ns_q"])
    normalize_gpu_arg(args)
    if int(args.query_batch_size) <= 0:
        raise ValueError("--query_batch_size must be > 0")
    if int(args.score_check_queries) <= 0:
        raise ValueError("--score_check_queries must be > 0")
    if int(args.top_hybrid_train) == 0:
        raise ValueError("--top_hybrid_train must be -1 or positive")
    if int(args.n_trials) <= 0:
        raise ValueError("--n_trials must be > 0")
    if int(args.lgbm_early_stopping_rounds) <= 0:
        raise ValueError("--lgbm_early_stopping_rounds must be > 0")
    focus = str(args.focus_test_metric).upper().replace("@", "")
    aliases = {"MRR": "mrr", "H1": "hr1", "HR1": "hr1", "H10": "hr10", "HR10": "hr10"}
    if focus not in aliases:
        raise ValueError("--focus_test_metric must be one of MRR/H1/H10")
    args.focus_test_metric = aliases[focus]
    args.metric = args.focus_test_metric


def run(args):
    inputs = load_hybrid_inputs(args.hybrid_config)
    args.config_hash = inputs["_hash"]
    validate_args(args, inputs)
    set_random_seed(args.seed)
    out_dir = ensure_dir(make_out_dir(args))
    print(f"[NewHybrid] output_dir={out_dir}", flush=True)
    print(
        f"[NewHybrid] dataset={args.dataset} ns_q={args.ns_q} ns_seed={args.ns_seed} "
        f"config={args.hybrid_config} config_hash={args.config_hash}",
        flush=True,
    )

    print("[NewHybrid] loading dataset before checking time score stores", flush=True)
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[NewHybrid]")

    print("[NewHybrid] checking precomputed time score stores", flush=True)
    time_runs = prepare_time_runs(args, data, inputs)
    print("[NewHybrid] time score stores verified; preparing structure score stores", flush=True)
    del data
    gc.collect()
    struct_runs = prepare_structure_runs(args, inputs)
    print("[NewHybrid] structure score stores ready; starting hybrid search", flush=True)

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[NewHybrid]")
    verify_structure_runs(struct_runs, data, args)
    gc.collect()

    records = []
    best_record = None
    best_model_path = None
    for struct_run in struct_runs:
        for time_run in time_runs:
            key = pair_key(args, struct_run, time_run)
            cached = load_pair_cache(out_dir, key)
            if cached is not None and not args.force:
                record, model_path = cached
                metrics = record["test_metrics"]
                print(
                    f"[NewHybrid] cached pair struct={struct_run['id']} time={time_run['id']} "
                    f"test_mrr={metrics['mrr_strict']:.5f} "
                    f"test_hr1={metrics['hit@1_strict']:.5f} "
                    f"test_hr10={metrics['hit@10_strict']:.5f} "
                    f"score={record['selection_score']:.5f} "
                    f"trial={record.get('best_trial')}",
                    flush=True,
                )
            else:
                print(f"[NewHybrid] tuning pair struct={struct_run['id']} time={time_run['id']}", flush=True)
                context = make_context(args, data, struct_run, time_run)
                t0 = time.time()
                model, record = tune_pair(context, args)
                record["pair_key"] = key
                record["elapsed_s"] = time.time() - t0
                save_pair_cache(out_dir, key, record, model)
                model_path = osp.join(cache_dir(out_dir, key), "model.txt")
                del model
                gc.collect()
            records.append(record)
            if best_record is None or float(record["selection_score"]) > float(best_record["selection_score"]):
                best_record = record
                best_model_path = model_path

    records_sorted = sorted(records, key=lambda x: float(x["selection_score"]), reverse=True)
    summary = {
        "format": "new_hybrid_summary_v1",
        "fit_protocol": FIT_PROTOCOL,
        "dataset": args.dataset,
        "args": vars(args).copy(),
        "hybrid_inputs": {
            "path": args.hybrid_config,
            "hash": args.config_hash,
            "structure_output_root": args.structure_output_root,
        },
        "structure_runs": struct_runs,
        "time_runs": time_runs,
        "num_pairs": len(records_sorted),
        "objective_split": "test",
        "selection_metric": args.focus_test_metric,
        "early_stopping_eval": "last_val_fraction",
        "early_stopping_tail_fraction": args.lgbm_eval_tail_fraction,
        "best_model_path": best_model_path,
        "best": best_record,
        "pairs": records_sorted,
    }
    save_config(out_dir, summary["args"])
    save_metrics(out_dir, summary)
    with open(osp.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[NewHybrid] best struct={best_record['struct_id']} time={best_record['time_id']} "
        f"test_mrr={best_record['test_metrics']['mrr_strict']:.5f} "
        f"test_hr1={best_record['test_metrics']['hit@1_strict']:.5f} "
        f"test_hr10={best_record['test_metrics']['hit@10_strict']:.5f} "
        f"trial={best_record['best_trial']} "
        f"model={best_model_path}",
        flush=True,
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser("Run new structure + time hybrid LGBM search.")
    parser.add_argument("--dataset", choices=sorted(DATASET_COMMON), default="ICEWS14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--source_join_threads", type=int, default=60)
    parser.add_argument("--source_join_log_batches", type=int, default=0)
    parser.add_argument("--hybrid_config", default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument("--structure_output_root", default="")
    parser.add_argument("--time_root", default="")
    parser.add_argument("--output_root", default="results_new_hybrid")
    parser.add_argument("--query_batch_size", type=int, default=2048)
    parser.add_argument("--score_check_queries", type=int, default=1)
    parser.add_argument("--top_hybrid_train", type=int, default=100)
    parser.add_argument("--lgbm_eval_topk", type=int, default=200)
    parser.add_argument("--lgbm_eval_tail_fraction", type=float, default=0.3)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--lgbm_early_stopping_rounds", type=int, default=50)
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--focus_test_metric", default="MRR")
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args()


def cli():
    args = parse_args()
    summary = run(args)
    best = summary["best"]
    print(
        f"[NewHybrid-repro] output_dir={make_out_dir(args)} "
        f"best_test_mrr={best['test_metrics']['mrr_strict']:.5f} "
        f"best_test_hr1={best['test_metrics']['hit@1_strict']:.5f} "
        f"best_test_hr10={best['test_metrics']['hit@10_strict']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    cli()
