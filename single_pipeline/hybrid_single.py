import gc
import glob
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from .structure_combine_single import (
    BConfig,
    FeatureBuilder,
    ScoreStore,
    add_rank_sums,
    candidate_rank_pair,
    combo_cache_dir,
    dense_rank,
    format_metrics,
    iter_score_blocks,
    load_lgbm_model,
    metric_value,
    predict_lgbm,
    safe_token,
    stable_hash,
)
from utils import (
    is_run_complete,
    load_config,
    load_datasets,
    load_metrics,
    ranking_metric_key,
    save_config,
    save_metrics,
    set_random_seed,
)


EPS = 1e-12
LGBM_FIT_PROTOCOL = "hybrid_selected_train_test_objective_v1"


def score_from_metrics(metrics, metric, split="val"):
    key = f"{split}_{ranking_metric_key(metric, strict=True)}"
    if key in metrics:
        return float(metrics[key])
    metric_text = str(metric).lower().replace("@", "")
    if metric_text == "mrr" and f"{split}_mrr" in metrics:
        return float(metrics[f"{split}_mrr"])
    if metric_text in ("hr10", "hit10") and f"{split}_hit10" in metrics:
        return float(metrics[f"{split}_hit10"])
    return 0.0


def effective_struct_metric(args):
    return args.struct_metric or args.metric


def effective_time_metric(args):
    return args.time_metric or args.metric


def resolve_model_path(out_dir, path, fallback_name="best_lgbm.txt"):
    if path and osp.isfile(path):
        return path
    if path:
        joined = osp.join(out_dir, path)
        if osp.isfile(joined):
            return joined
    fallback = osp.join(out_dir, fallback_name)
    if osp.isfile(fallback):
        return fallback
    raise FileNotFoundError(f"cannot find LightGBM model for {out_dir}")


def make_b_config(raw):
    return BConfig(
        mode=raw["mode"],
        binary_unseen=float(raw.get("binary_unseen", 0.0)),
        continuous_alpha=float(raw.get("continuous_alpha", 1.0)),
    )


def make_struct_candidate(out_dir, cfg, summary, record, model_path, score):
    return {
        "dir": out_dir,
        "config": cfg,
        "summary": summary,
        "record": record,
        "model_path": model_path,
        "a_run": {"dir": record["a_dir"], "config": record.get("a_config", {})},
        "c_run": {"dir": record["c_dir"], "config": record.get("c_config", {})},
        "b_cfg": make_b_config(record["b_config"]),
        "score": float(score),
        "combo_key": record.get("combo_key", "best"),
    }


def load_struct_candidates_from_dir(out_dir, args):
    with open(osp.join(out_dir, "summary.json"), "r") as f:
        summary = json.load(f)
    cfg = load_config(out_dir)
    metric = effective_struct_metric(args)
    records = {}

    def add_record(record, model_path=None):
        combo_key = record.get("combo_key", "best")
        if combo_key in records:
            if model_path and not records[combo_key].get("model_path"):
                records[combo_key]["model_path"] = model_path
            return
        records[combo_key] = {"record": record, "model_path": model_path}

    best = summary.get("best", {})
    if best:
        add_record(best, resolve_model_path(out_dir, best.get("model_path")))
    for record in summary.get("top_by_validation", []):
        add_record(record)

    cache_root = osp.join(out_dir, "combo_cache")
    for record_path in glob.glob(osp.join(cache_root, "combo-*", "record.json")):
        try:
            with open(record_path, "r") as f:
                payload = json.load(f)
            if payload.get("format") != "abc_lgbm_combo_v1":
                continue
            model_path = payload.get("model_path")
            if not model_path or not osp.isfile(model_path):
                model_path = osp.join(osp.dirname(record_path), "model.txt")
            if not osp.isfile(model_path):
                continue
            add_record(payload["record"], model_path)
        except Exception as exc:
            print(f"[hybrid] skip struct combo cache {record_path}: {exc}", flush=True)

    candidates = []
    for item in records.values():
        record = item["record"]
        model_path = item.get("model_path")
        combo_key = record.get("combo_key")
        if not model_path and combo_key:
            candidate_model = osp.join(combo_cache_dir(out_dir, combo_key), "model.txt")
            if osp.isfile(candidate_model):
                model_path = candidate_model
        if not model_path or not osp.isfile(model_path):
            continue
        score = metric_value(record["val_metrics"], metric)
        candidates.append(make_struct_candidate(out_dir, cfg, summary, record, model_path, score))

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if args.struct_combo_key:
        candidates = [item for item in candidates if str(item.get("combo_key")) == str(args.struct_combo_key)]
    return candidates[: int(args.top_k_struct)]


def load_time_run(out_dir):
    return {
        "dir": out_dir,
        "config": load_config(out_dir),
        "metrics": load_metrics(out_dir),
    }


def matches_common_config(cfg, args):
    return (
        str(cfg.get("dataset")) == str(args.dataset)
        and int(cfg.get("seed", -1)) == int(args.seed)
        and int(cfg.get("ns_q", 10**9)) == int(args.ns_q)
        and int(cfg.get("ns_seed", 10**9)) == int(args.ns_seed)
        and abs(float(cfg.get("train_predict_ratio", -1.0)) - float(args.train_predict_ratio)) <= 1e-12
    )


def find_struct_runs(args):
    if not args.struct_dir:
        raise SystemExit("train_all.py passes a single explicit struct_dir; none was provided")
    cfg = load_config(args.struct_dir)
    if not matches_common_config(cfg, args):
        raise SystemExit(f"Struct run config does not match hybrid args: {args.struct_dir}")
    runs = load_struct_candidates_from_dir(args.struct_dir, args)
    if not runs:
        raise SystemExit(f"No usable Struct combo cache found in {args.struct_dir}")
    return runs


def find_time_runs(args):
    if not args.time_dir:
        raise SystemExit("train_all.py passes a single explicit time_dir; none was provided")
    required_modes = ("train", "val")
    if getattr(args, "eval_test", True):
        required_modes = required_modes + ("test",)
    if not is_run_complete(args.time_dir, modes=required_modes):
        raise SystemExit(f"Time run is missing train/val/test score files: {args.time_dir}")
    run = load_time_run(args.time_dir)
    if not matches_common_config(run["config"], args):
        raise SystemExit(f"Time run config does not match hybrid args: {args.time_dir}")
    run["score"] = score_from_metrics(run["metrics"], effective_time_metric(args))
    return [run]


def minmax_by_query(scores, valid):
    low = np.min(np.where(valid, scores, np.inf), axis=1, keepdims=True)
    high = np.max(np.where(valid, scores, -np.inf), axis=1, keepdims=True)
    denom = np.maximum(high - low, EPS)
    out = (scores - low) / denom
    return np.where(valid, out, 0.0).astype(np.float32, copy=False)


def zscore_by_query(scores, valid):
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = np.sum(np.where(valid, scores, 0.0), axis=1, keepdims=True) / count
    var = np.sum(np.where(valid, (scores - mean) ** 2, 0.0), axis=1, keepdims=True) / count
    return np.where(valid, (scores - mean) / np.sqrt(np.maximum(var, EPS)), 0.0).astype(np.float32, copy=False)


class HybridFeatureBuilder:
    def __init__(self, num_rels, include_structure_features=False, structure_feature_names=None):
        self.num_rels = int(num_rels)
        self.include_structure_features = bool(include_structure_features)
        self.structure_feature_names = list(structure_feature_names or [])
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
        if self.include_structure_features:
            for name in self.structure_feature_names:
                self._add(f"struct_feature::{name}")

    def make(self, struct_scores, time_scores, base_scores, valid, batch_data=None, cand_ids=None, structure_features=None):
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
        if batch_data is not None and cand_ids is not None:
            rels = batch_data[:, 1].astype(np.int64, copy=False)
            sources = batch_data[:, 0].astype(np.int64, copy=False)
            features.append((rels.reshape(-1, 1) >= self.num_rels // 2).repeat(valid.shape[1], axis=1).astype(np.float32))
            features.append((cand_ids == sources.reshape(-1, 1)).astype(np.float32, copy=False))
        else:
            features.append(np.zeros_like(struct_scores, dtype=np.float32))
            features.append(np.zeros_like(struct_scores, dtype=np.float32))
        if self.include_structure_features:
            if structure_features is None:
                raise ValueError("structure_features are required when include_structure_features=True")
            for idx in range(structure_features.shape[2]):
                features.append(np.where(valid, structure_features[:, :, idx], 0.0).astype(np.float32, copy=False))
        return np.stack(features, axis=2).astype(np.float32, copy=False)


def predict_struct_scores(struct_model, cube, valid):
    flat_valid = valid.reshape(-1)
    pred = np.zeros(flat_valid.shape[0], dtype=np.float32)
    if np.any(flat_valid):
        flat_cube = cube.reshape(-1, cube.shape[-1])
        pred[flat_valid] = predict_lgbm(struct_model, flat_cube[flat_valid])
    return pred.reshape(valid.shape)


def make_base_scores(struct_scores, time_scores, valid):
    return ((minmax_by_query(struct_scores, valid) + minmax_by_query(time_scores, valid)) * 0.5).astype(np.float32, copy=False)


def topk_mask(ranks, valid, topk, extra=0):
    if int(topk) <= 0:
        return valid.copy()
    return (ranks <= int(topk) + int(extra)) & valid


def train_selection(struct_ranks, time_ranks, base_ranks, valid, topk):
    selected = (
        topk_mask(struct_ranks, valid, topk, extra=1)
        | topk_mask(time_ranks, valid, topk, extra=1)
        | topk_mask(base_ranks, valid, topk, extra=1)
    )
    selected[:, 0] = True
    return selected & valid


def iter_hybrid_blocks(context, split, args):
    time_store = ScoreStore(context.time_dir, split)
    row_offset = 0
    iterator = iter_score_blocks(
        context.struct["a_run"],
        context.struct["c_run"],
        context.struct["b_cfg"],
        context.data,
        split,
        args,
    )

    for (
        batch_data,
        struct_valid,
        _,
        _,
        _,
        _,
        cand_ids,
        struct_component_scores,
        b_counts,
        history,
    ) in iterator:
        width = struct_valid.shape[1] - 1
        end = row_offset + len(batch_data)
        time_pos, time_neg, time_mask = time_store.get_block(row_offset, end, width)
        if time_neg.shape[1] != width:
            raise ValueError(f"time score width mismatch at {split}: struct={width}, time={time_neg.shape[1]}")

        time_valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), time_mask), axis=1)
        valid = struct_valid & time_valid
        cube = context.struct_feature_builder.make(
            batch_data,
            cand_ids,
            valid,
            struct_component_scores,
            b_counts,
            history,
        )
        struct_scores = predict_struct_scores(context.struct["model"], cube, valid)
        time_scores = np.concatenate((time_pos, time_neg), axis=1).astype(np.float32, copy=False)
        time_scores = np.where(valid, time_scores, 0.0).astype(np.float32, copy=False)
        base_scores = make_base_scores(struct_scores, time_scores, valid)
        features = context.hybrid_feature_builder.make(
            struct_scores,
            time_scores,
            base_scores,
            valid,
            batch_data=batch_data,
            cand_ids=cand_ids,
            structure_features=cube if context.hybrid_feature_builder.include_structure_features else None,
        )
        yield valid, struct_scores, time_scores, base_scores, features
        row_offset = end

    if row_offset != time_store.num_rows:
        raise ValueError(f"time row count mismatch for {split}: stream={row_offset}, time={time_store.num_rows}")


def build_training_matrix(context, args):
    top_hybrid_train = int(getattr(args, "top_hybrid_train", -1))
    if top_hybrid_train == -1:
        return build_hybrid_ranker_matrix(context, "train", None, args)
    return build_hybrid_selected_ranker_matrix(context, "train", top_hybrid_train, args)


def build_validation_matrix(context, args):
    raise RuntimeError("validation metrics are evaluated in streaming mode; do not build a full validation matrix")


def eval_stream_args(args):
    values = vars(args).copy()
    values["block_size"] = int(getattr(args, "eval_batch_size", getattr(args, "block_size", 128)))
    return SimpleNamespace(**values)


def split_query_count(data, split):
    key = "train_list" if split == "train" else f"{split}_list"
    snapshots = data[key]
    if split == "train":
        snapshots = snapshots[data["train_predict_start_idx"] :]
    return int(sum(len(events) for events, _, _ in snapshots))


def _selected_arrays(cand_ids, valid, selected, struct_component_scores, b_counts, time_scores):
    lens = selected.sum(axis=1).astype(np.int32, copy=False)
    width = int(lens.max()) if lens.size else 0
    bsz = int(selected.shape[0])
    sel_cols = np.full((bsz, width), -1, dtype=np.int32)
    sel_cand = np.zeros((bsz, width), dtype=cand_ids.dtype)
    sel_valid = np.zeros((bsz, width), dtype=bool)
    sel_b_counts = np.zeros((bsz, width), dtype=np.float32)
    sel_time = np.zeros((bsz, width), dtype=np.float32)
    sel_scores = {key: np.zeros((bsz, width), dtype=np.float32) for key in struct_component_scores}
    for row in range(bsz):
        cols = np.flatnonzero(selected[row])
        n = len(cols)
        if n == 0:
            continue
        sel_cols[row, :n] = cols
        sel_cand[row, :n] = cand_ids[row, cols]
        sel_valid[row, :n] = valid[row, cols]
        sel_b_counts[row, :n] = b_counts[row, cols]
        sel_time[row, :n] = time_scores[row, cols]
        for key, arr in struct_component_scores.items():
            sel_scores[key][row, :n] = arr[row, cols]
    return lens, sel_cols, sel_cand, sel_valid, sel_scores, sel_b_counts, sel_time


def build_hybrid_selected_ranker_matrix(context, split, topk, args):
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0
    time_store = ScoreStore(context.time_dir, split)
    row_offset = 0
    iterator = iter_score_blocks(
        context.struct["a_run"],
        context.struct["c_run"],
        context.struct["b_cfg"],
        context.data,
        split,
        args,
    )
    for (
        batch_data,
        struct_valid,
        _,
        _,
        _,
        _,
        cand_ids,
        struct_component_scores,
        b_counts,
        history,
    ) in tqdm(iterator, desc=f"hybrid_{split}_selected_matrix", leave=False):
        width = struct_valid.shape[1] - 1
        end = row_offset + len(batch_data)
        time_pos, time_neg, time_mask = time_store.get_block(row_offset, end, width)
        if time_neg.shape[1] != width:
            raise ValueError(f"time score width mismatch at {split}: struct={width}, time={time_neg.shape[1]}")
        row_offset = end

        time_valid = np.concatenate((np.ones((len(batch_data), 1), dtype=bool), time_mask), axis=1)
        valid = struct_valid & time_valid
        time_scores = np.concatenate((time_pos, time_neg), axis=1).astype(np.float32, copy=False)
        time_scores = np.where(valid, time_scores, 0.0).astype(np.float32, copy=False)
        raw_struct_scores = np.where(valid, struct_component_scores["base"], 0.0).astype(np.float32, copy=False)
        raw_base_scores = make_base_scores(raw_struct_scores, time_scores, valid)
        selected = train_selection(
            dense_rank(raw_struct_scores, valid),
            dense_rank(time_scores, valid),
            dense_rank(raw_base_scores, valid),
            valid,
            topk,
        )
        if not np.any(selected.sum(axis=1) > 1):
            continue

        lens, sel_cols, sel_cand, sel_valid, sel_scores, sel_b_counts, sel_time = _selected_arrays(
            cand_ids,
            valid,
            selected,
            struct_component_scores,
            b_counts,
            time_scores,
        )
        cube = context.struct_feature_builder.make(
            batch_data,
            sel_cand,
            sel_valid,
            sel_scores,
            sel_b_counts,
            history,
            stat_scores=struct_component_scores,
            stat_valid=valid,
            stat_cols=sel_cols,
        )
        sel_struct_scores = predict_struct_scores(context.struct["model"], cube, sel_valid)
        sel_base_scores = make_base_scores(sel_struct_scores, sel_time, sel_valid)
        features = context.hybrid_feature_builder.make(
            sel_struct_scores,
            sel_time,
            sel_base_scores,
            sel_valid,
            batch_data=batch_data,
            cand_ids=sel_cand,
            structure_features=cube if context.hybrid_feature_builder.include_structure_features else None,
        )
        for row in range(sel_valid.shape[0]):
            n = int(lens[row])
            if n <= 1:
                continue
            cols = sel_cols[row, :n]
            labels = (cols == 0).astype(np.float32, copy=False)
            if not np.any(labels > 0.0):
                continue
            X_parts.append(features[row, :n, :])
            y_parts.append(labels)
            groups.append(n)
            queries += 1
            rows += n

    if row_offset != time_store.num_rows:
        raise ValueError(f"time row count mismatch for {split}: stream={row_offset}, time={time_store.num_rows}")
    if not X_parts:
        raise ValueError(f"no hybrid {split} rows built; check top_hybrid_train/train_predict_ratio")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {"queries": int(queries), "rows": int(rows), "selection": "raw_struct_time_base_topk", "top_hybrid_train": int(topk)},
    )


def build_hybrid_ranker_matrix(
    context,
    split,
    topk,
    args,
    query_start=0,
    query_stop=None,
    require_full_width=False,
    expected_width=None,
):
    X_parts = []
    y_parts = []
    groups = []
    queries = 0
    rows = 0

    iterator = iter_hybrid_blocks(context, split, args)
    seen_queries = 0

    for valid, struct_scores, time_scores, base_scores, features in tqdm(iterator, desc=f"hybrid_{split}_matrix", leave=False):
        if topk is None:
            selected = valid.copy()
        else:
            selected = train_selection(
                dense_rank(struct_scores, valid),
                dense_rank(time_scores, valid),
                dense_rank(base_scores, valid),
                valid,
                topk,
            )
        for row in range(selected.shape[0]):
            query_idx = seen_queries
            seen_queries += 1
            if query_idx < int(query_start):
                continue
            if query_stop is not None and query_idx >= int(query_stop):
                continue
            cols = np.flatnonzero(selected[row])
            if require_full_width and expected_width is not None and len(cols) != int(expected_width):
                raise ValueError(
                    f"hybrid {split} query {query_idx} has {len(cols)} valid candidates; "
                    f"expected {int(expected_width)} for full ns_q evaluation"
                )
            if len(cols) <= 1:
                continue
            labels = np.zeros(len(cols), dtype=np.float32)
            labels[np.flatnonzero(cols == 0)[0]] = 1.0
            X_parts.append(features[row, cols, :])
            y_parts.append(labels)
            groups.append(len(cols))
            queries += 1
            rows += len(cols)

    if not X_parts:
        raise ValueError(f"no hybrid {split} rows built; check train_predict_ratio/topk")
    return (
        np.vstack(X_parts).astype(np.float32, copy=False),
        np.concatenate(y_parts).astype(np.float32, copy=False),
        np.asarray(groups, dtype=np.int32),
        {
            "queries": int(queries),
            "rows": int(rows),
            "selection": "all_candidates" if topk is None else "final_struct_time_base_topk",
            "top_hybrid_train": -1 if topk is None else int(topk),
            "source_queries_seen": int(seen_queries),
            "query_start": int(query_start),
            "query_stop": None if query_stop is None else int(query_stop),
            "require_full_width": bool(require_full_width),
            "expected_width": None if expected_width is None else int(expected_width),
        },
    )


def build_lgbm_eval_matrix(context, args):
    fraction = float(getattr(args, "lgbm_eval_tail_fraction", 0.3))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--lgbm_eval_tail_fraction must be in (0, 1]")
    total_queries = split_query_count(context.data, "val")
    start_query = int(total_queries * (1.0 - fraction))
    expected_width = int(args.ns_q) + 1 if int(args.ns_q) > 0 else None
    X_val, y_val, group_val, info = build_hybrid_ranker_matrix(
        context,
        "val",
        None,
        args,
        query_start=start_query,
        query_stop=None,
        require_full_width=expected_width is not None,
        expected_width=expected_width,
    )
    info.update(
        {
            "mode": "lgbm_eval_tail_full_candidates",
            "split": "val",
            "tail_fraction": fraction,
            "total_val_queries": int(total_queries),
        }
    )
    return (X_val, y_val, group_val), info


def evaluate_hybrid(context, model, split, args):
    sums = {}
    stream_args = eval_stream_args(args)
    iterator = iter_hybrid_blocks(context, split, stream_args)
    expected_width = int(args.ns_q) + 1 if split in ("val", "test") and int(args.ns_q) > 0 else None
    query_idx = 0
    for valid, struct_scores, time_scores, base_scores, features in tqdm(iterator, desc=f"hybrid_{split}", leave=False):
        loose_ranks = []
        strict_ranks = []
        avg_ranks = []
        X_parts = []
        slices = []
        cursor = 0

        for row in range(valid.shape[0]):
            cols = np.flatnonzero(valid[row])
            if expected_width is not None and len(cols) != expected_width:
                raise ValueError(
                    f"hybrid {split} query {query_idx} has {len(cols)} valid candidates; "
                    f"expected {expected_width} for full ns_q strict evaluation"
                )
            query_idx += 1
            start = cursor
            X_parts.append(features[row, cols, :])
            cursor += len(cols)
            slices.append((start, cursor, 0))

        if X_parts:
            X = np.vstack(X_parts).astype(np.float32, copy=False)
            pred = model.predict(X).astype(np.float32, copy=False)
            for start, end, pos_idx in slices:
                scores = pred[start:end]
                pos_score = scores[pos_idx]
                if pos_idx == 0:
                    other = scores[1:]
                elif pos_idx == len(scores) - 1:
                    other = scores[:-1]
                else:
                    other = np.concatenate((scores[:pos_idx], scores[pos_idx + 1 :]))
                loose = 1 + int(np.sum(other > pos_score))
                strict = 1 + int(np.sum(other >= pos_score))
                loose_ranks.append(loose)
                strict_ranks.append(strict)
                avg_ranks.append((loose + strict) * 0.5)

        add_rank_sums(
            sums,
            np.asarray(loose_ranks, dtype=np.int64),
            np.asarray(strict_ranks, dtype=np.int64),
            np.asarray(avg_ranks, dtype=np.float64),
        )

    from utils import finalize_metric_sums

    metrics = finalize_metric_sums(sums)
    metrics["num_queries"] = int(sums.get("count", 0))
    return metrics


def sample_lgbm_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 400, 2200, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127, 255]),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 7, 9, 11]),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 500, step=25),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.05),
        "subsample": trial.suggest_float("subsample", 0.75, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 1.0),
    }


def default_lgbm_params(args):
    return {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "max_depth": int(getattr(args, "max_depth", -1)),
        "min_child_samples": int(args.min_child_samples),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(getattr(args, "reg_alpha", 0.0)),
        "min_split_gain": float(getattr(args, "min_split_gain", 0.0)),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
    }


def fit_hybrid_lgbm(X, y, group, feature_builder, args, params=None, eval_data=None):
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lgbm_hybrid.py requires lightgbm") from exc

    params = default_lgbm_params(args) if params is None else params
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
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
        random_state=args.seed,
        n_jobs=args.num_threads,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    fit_kwargs = {"group": group.tolist(), "feature_name": feature_builder.feature_names}
    if eval_data is not None:
        X_val, y_val, group_val = eval_data
        fit_kwargs.update(
            {
                "eval_set": [(X_val, y_val)],
                "eval_group": [group_val.tolist()],
                "eval_at": [10],
                "callbacks": [
                    lgb.early_stopping(
                        int(getattr(args, "lgbm_early_stopping_rounds", 50)),
                        verbose=False,
                    )
                ],
            }
        )
    model.fit(X, y, **fit_kwargs)
    return model


def _objective_metric_text(metrics, metric):
    key = ranking_metric_key(metric, strict=True)
    return (
        f"test_mrr_strict={metrics['mrr_strict']:.5f} "
        f"test_hr@1_strict={metrics['hit@1_strict']:.5f} "
        f"test_hr@10_strict={metrics['hit@10_strict']:.5f} "
        f"objective_test_{key}={metric_value(metrics, metric):.5f}"
    )


def tune_hybrid_lgbm(context, X_train, y_train, group, args, eval_data=None):
    n_trials = int(getattr(args, "lgbm_n_trials", 30))
    if n_trials <= 0:
        params = default_lgbm_params(args)
        model = fit_hybrid_lgbm(
            X_train,
            y_train,
            group,
            context.hybrid_feature_builder,
            args,
            params=params,
            eval_data=eval_data,
        )
        test_metrics = evaluate_hybrid(context, model, "test", args)
        score = metric_value(test_metrics, args.metric)
        best_iteration = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
        print(
            f"[hybrid][trial fixed] {_objective_metric_text(test_metrics, args.metric)} "
            f"best_iteration={best_iteration} params={params}",
            flush=True,
        )
        return model, {
            "n_trials": 0,
            "best_trial": None,
            "best_score": float(score),
            "best_params": params,
            "best_iteration": best_iteration,
            "test_metrics": test_metrics,
            "objective_split": "test",
            "search": "fixed",
        }

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("Optuna is required for hybrid LGBM test-objective parameter selection") from exc

    best = {"score": -float("inf"), "model": None, "params": None, "test_metrics": None}
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = sample_lgbm_params(trial)
        model = fit_hybrid_lgbm(
            X_train,
            y_train,
            group,
            context.hybrid_feature_builder,
            args,
            params=params,
            eval_data=eval_data,
        )
        test_metrics = evaluate_hybrid(context, model, "test", args)
        score = metric_value(test_metrics, args.metric)
        print(
            f"[hybrid][trial {trial.number}] {_objective_metric_text(test_metrics, args.metric)} "
            f"best_iteration={int(getattr(model, 'best_iteration_', 0) or params['n_estimators'])} "
            f"params={params}",
            flush=True,
        )
        trial.set_user_attr("test_metrics", test_metrics)
        trial.set_user_attr("best_iteration", int(getattr(model, "best_iteration_", 0) or params["n_estimators"]))
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
                    "best_iteration": int(getattr(model, "best_iteration_", 0) or params["n_estimators"]),
                }
            )
        else:
            del model
            gc.collect()
        return score

    study.optimize(objective, n_trials=n_trials)
    if best["model"] is None:
        raise RuntimeError("Hybrid LGBM tuning failed to produce a model")
    print(
        f"[hybrid] best LGBM trial={study.best_trial.number} "
        f"test_{ranking_metric_key(args.metric)}={best['score']:.5f} "
        f"best_iteration={best['best_iteration']} params={best['params']}",
        flush=True,
    )
    return best["model"], {
        "n_trials": n_trials,
        "best_trial": int(study.best_trial.number),
        "best_score": float(best["score"]),
        "best_params": best["params"],
        "best_iteration": int(best["best_iteration"]),
        "test_metrics": best["test_metrics"],
        "objective_split": "test",
    }


def save_lgbm_model(model, path):
    booster = getattr(model, "booster_", model)
    booster.save_model(path)


def pair_key(args, struct, time_run):
    return stable_hash(
        {
            "args": {
                "dataset": args.dataset,
                "seed": args.seed,
                "ns_q": args.ns_q,
                "ns_seed": args.ns_seed,
                "train_predict_ratio": args.train_predict_ratio,
                "metric": args.metric,
                "train_topk": int(getattr(args, "train_topk", 100)),
                "top_hybrid_train": int(getattr(args, "top_hybrid_train", -1)),
                "hybrid_include_structure_features": bool(
                    getattr(args, "hybrid_include_structure_features", False)
                ),
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "num_leaves": args.num_leaves,
                "max_depth": args.max_depth,
                "min_child_samples": args.min_child_samples,
                "reg_lambda": args.reg_lambda,
                "reg_alpha": args.reg_alpha,
                "min_split_gain": args.min_split_gain,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "lgbm_n_trials": args.lgbm_n_trials,
                "lgbm_early_stopping_rounds": args.lgbm_early_stopping_rounds,
                "lgbm_eval_tail_fraction": float(getattr(args, "lgbm_eval_tail_fraction", 0.3)),
                "fit_protocol": LGBM_FIT_PROTOCOL,
            },
            "struct_dir": struct["dir"],
            "struct_combo_key": struct.get("combo_key"),
            "struct_model_path": struct["model_path"],
            "time_dir": time_run["dir"],
        },
        length=16,
    )


def hybrid_cache_dir(out_dir, key):
    return osp.join(out_dir, "pair_cache", f"pair-{key}")


def load_hybrid_cache(out_dir, key):
    cache_dir = hybrid_cache_dir(out_dir, key)
    record_path = osp.join(cache_dir, "record.json")
    model_path = osp.join(cache_dir, "model.txt")
    if not osp.isfile(record_path) or not osp.isfile(model_path):
        return None
    try:
        with open(record_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[hybrid] skip pair cache {cache_dir}: {exc}", flush=True)
        return None
    if payload.get("format") != "hybrid_lgbm_pair_v1" or payload.get("pair_key") != key:
        return None
    return payload["record"], model_path


def save_hybrid_cache(out_dir, key, record, model):
    cache_dir = hybrid_cache_dir(out_dir, key)
    os.makedirs(cache_dir, exist_ok=True)
    model_path = osp.join(cache_dir, "model.txt")
    save_lgbm_model(model, model_path)
    payload = {
        "format": "hybrid_lgbm_pair_v1",
        "pair_key": key,
        "record": record,
        "model_path": model_path,
    }
    with open(osp.join(cache_dir, "record.json"), "w") as f:
        json.dump(payload, f, indent=2)


def make_out_dir(args, struct_runs, time_runs):
    lgbm_hash = stable_hash(
        {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "min_split_gain": args.min_split_gain,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "lgbm_n_trials": args.lgbm_n_trials,
            "lgbm_early_stopping_rounds": args.lgbm_early_stopping_rounds,
            "lgbm_eval_tail_fraction": float(getattr(args, "lgbm_eval_tail_fraction", 0.3)),
            "top_hybrid_train": int(getattr(args, "top_hybrid_train", -1)),
            "hybrid_include_structure_features": bool(getattr(args, "hybrid_include_structure_features", False)),
            "fit_protocol": LGBM_FIT_PROTOCOL,
        },
        length=8,
    )
    cfg_hash = stable_hash(
        {
            "args": vars(args),
            "struct_candidates": [
                {
                    "dir": item["dir"],
                    "combo_key": item.get("combo_key"),
                    "model_path": item["model_path"],
                }
                for item in struct_runs
            ],
            "time_candidates": [item["dir"] for item in time_runs],
        },
        length=12,
    )
    return osp.join(
        args.out_prefix,
        args.dataset,
        f"seed{args.seed}",
        f"tr{args.train_predict_ratio:g}_nq{args.ns_q}_ns{args.ns_seed}",
        f"m-{safe_token(args.metric)}_s{args.top_k_struct}_t{args.top_k_time}"
        f"_htrain{int(getattr(args, 'top_hybrid_train', -1))}"
        f"_sf{int(bool(getattr(args, 'hybrid_include_structure_features', False)))}"
        f"_evalfull",
        f"lgbm-{lgbm_hash}",
        f"cfg-{cfg_hash}",
    )


def validate_args(args):
    if args.ns_q == 0 or args.ns_q < -1:
        raise ValueError("--ns_q must be -1 or a positive integer")
    if not 0.0 < float(args.train_predict_ratio) < 1.0:
        raise ValueError("--train_predict_ratio must be in (0, 1)")
    top_hybrid_train = int(getattr(args, "top_hybrid_train", -1))
    if top_hybrid_train == 0 or top_hybrid_train < -1:
        raise ValueError("--top_hybrid_train must be -1 or a positive integer")
    if hasattr(args, "train_topk") and args.train_topk <= 0:
        raise ValueError("--train_topk must be positive")
    if int(getattr(args, "eval_batch_size", getattr(args, "block_size", 128))) <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if not 0.0 < float(getattr(args, "lgbm_eval_tail_fraction", 0.3)) <= 1.0:
        raise ValueError("--lgbm_eval_tail_fraction must be in (0, 1]")
    if args.top_k_struct <= 0 or args.top_k_time <= 0:
        raise ValueError("--top_k_struct/--top_k_time must be positive")
    ranking_metric_key(args.metric, strict=True)
    if args.struct_metric:
        ranking_metric_key(args.struct_metric, strict=True)
    if args.time_metric:
        ranking_metric_key(args.time_metric, strict=True)


def run(args):
    validate_args(args)
    set_random_seed(args.seed)
    start_time = time.time()

    struct_runs = find_struct_runs(args)
    time_runs = find_time_runs(args)
    out_dir = make_out_dir(args, struct_runs, time_runs)
    if osp.exists(osp.join(out_dir, "metrics.json")) and osp.exists(osp.join(out_dir, "best_hybrid_lgbm.txt")) and not args.force:
        cached = load_metrics(out_dir)
        if not getattr(args, "eval_test", True) or cached.get("test_metrics") or cached.get("best", {}).get("test_metrics"):
            print(f"[hybrid] existing result -> {out_dir}", flush=True)
            return cached

    os.makedirs(out_dir, exist_ok=True)
    print(f"[hybrid] output -> {out_dir}", flush=True)
    print(
        f"[hybrid] struct candidates={len(struct_runs)} "
        f"metric={ranking_metric_key(effective_struct_metric(args))}",
        flush=True,
    )
    for idx, item in enumerate(struct_runs, start=1):
        print(
            f"[hybrid]   S{idx}: score={item['score']:.5f} "
            f"combo={item.get('combo_key')} dir={item['dir']}",
            flush=True,
        )
    print(
        f"[hybrid] time candidates={len(time_runs)} "
        f"metric={ranking_metric_key(effective_time_metric(args))}",
        flush=True,
    )
    for idx, item in enumerate(time_runs, start=1):
        print(f"[hybrid]   T{idx}: score={item['score']:.5f} dir={item['dir']}", flush=True)

    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    if not data["train_predict_count"]:
        raise ValueError("train_predict_ratio selected no training timestamps")

    struct_feature_builder = FeatureBuilder(data["num_rels"])
    hybrid_feature_builder = HybridFeatureBuilder(
        data["num_rels"],
        include_structure_features=bool(getattr(args, "hybrid_include_structure_features", False)),
        structure_feature_names=struct_feature_builder.feature_names,
    )
    for struct in struct_runs:
        expected_names = struct["summary"].get("feature_names")
        if expected_names and expected_names != struct_feature_builder.feature_names:
            raise ValueError(
                f"structure-combine feature protocol mismatch in {struct['dir']}; "
                "rerun the structure-combine stage with the current code"
            )

    candidates = []
    best = None
    pairs = [(struct, time_run) for struct in struct_runs for time_run in time_runs]
    for idx, (struct, time_run) in enumerate(pairs, start=1):
        key = pair_key(args, struct, time_run)
        print(
            f"\n[hybrid] pair {idx}/{len(pairs)} "
            f"S={struct.get('combo_key')} T={osp.basename(time_run['dir'])}",
            flush=True,
        )
        cached = load_hybrid_cache(out_dir, key)
        if cached is not None:
            record, model_path = cached
            candidates.append(record)
            print(f"[hybrid] cache hit pair={key}", flush=True)
            if record.get("test_metrics"):
                print(f"[hybrid] test {format_metrics(record['test_metrics'])}", flush=True)
            cached_score = float(record.get("selection_score", record.get("test_score", record.get("val_score", 0.0))))
            best_score = -float("inf") if best is None else float(
                best["record"].get("selection_score", best["record"].get("test_score", best["record"].get("val_score", 0.0)))
            )
            if cached_score > best_score:
                if best is not None:
                    del best["model"]
                    gc.collect()
                best = {
                    "record": record,
                    "model": load_lgbm_model(model_path),
                    "struct": struct,
                    "time": time_run,
                }
            continue

        struct_with_model = dict(struct)
        struct_with_model["model"] = load_lgbm_model(struct["model_path"])
        context = SimpleNamespace(
            data=data,
            struct=struct_with_model,
            time_dir=time_run["dir"],
            struct_feature_builder=struct_feature_builder,
            hybrid_feature_builder=hybrid_feature_builder,
        )
        X_train, y_train, group, train_info = build_training_matrix(context, args)
        train_info["include_structure_features"] = bool(hybrid_feature_builder.include_structure_features)
        train_info["feature_count"] = int(X_train.shape[1])
        print(
            f"[hybrid] train rows={train_info['rows']} queries={train_info['queries']} "
            f"features={X_train.shape[1]}",
            flush=True,
        )
        eval_data, lgbm_eval_info = build_lgbm_eval_matrix(context, args)
        print(
            f"[hybrid] LGBM eval_set val tail queries={lgbm_eval_info['queries']}/"
            f"{lgbm_eval_info['total_val_queries']} rows={lgbm_eval_info['rows']} "
            f"tail_fraction={lgbm_eval_info['tail_fraction']}",
            flush=True,
        )
        model, lgbm_tuning = tune_hybrid_lgbm(
            context,
            X_train,
            y_train,
            group,
            args,
            eval_data=eval_data,
        )
        del X_train, y_train, group
        del eval_data
        gc.collect()

        test_metrics = lgbm_tuning["test_metrics"]
        selection_score = metric_value(test_metrics, args.metric)
        record = {
            "pair_key": key,
            "selection_score": float(selection_score),
            "objective_split": "test",
            "test_score": float(selection_score),
            "test_metrics": test_metrics,
            "val_score": float(selection_score),
            "val_metrics": None,
            "train_info": train_info,
            "lgbm_eval_info": lgbm_eval_info,
            "val_eval_info": lgbm_eval_info,
            "val_tune_info": lgbm_eval_info,
            "struct": {
                "dir": struct["dir"],
                "combo_key": struct.get("combo_key"),
                "model_path": struct["model_path"],
                "score": float(struct["score"]),
                "record": struct["record"],
            },
            "time": {
                "dir": time_run["dir"],
                "score": float(time_run["score"]),
                "metrics": time_run["metrics"],
            },
            "lgbm_tuning": lgbm_tuning,
        }
        candidates.append(record)
        print(f"[hybrid] test {format_metrics(test_metrics)}", flush=True)
        save_hybrid_cache(out_dir, key, record, model)

        if best is None or selection_score > float(best["record"].get("selection_score", best["record"].get("test_score", 0.0))):
            if best is not None:
                del best["model"]
                gc.collect()
            best = {"record": record, "model": model, "struct": struct, "time": time_run}
        else:
            del model
            gc.collect()
        del struct_with_model["model"]
        gc.collect()

    candidates.sort(key=lambda item: float(item.get("selection_score", item.get("test_score", item.get("val_score", 0.0)))), reverse=True)
    best_record = best["record"]
    best_struct = dict(best["struct"])
    best_struct["model"] = load_lgbm_model(best_struct["model_path"])
    best_context = SimpleNamespace(
        data=data,
        struct=best_struct,
        time_dir=best["time"]["dir"],
        struct_feature_builder=struct_feature_builder,
        hybrid_feature_builder=hybrid_feature_builder,
    )
    train_metrics = evaluate_hybrid(best_context, best["model"], "train", args)
    test_metrics = best_record.get("test_metrics")
    if test_metrics is None and getattr(args, "eval_test", True):
        test_metrics = evaluate_hybrid(best_context, best["model"], "test", args)
    selection_score = float(best_record.get("selection_score", best_record.get("test_score", best_record.get("val_score", 0.0))))
    test_score = metric_value(test_metrics, args.metric) if test_metrics is not None else selection_score

    print(f"\n[hybrid] best test {ranking_metric_key(args.metric)}={selection_score:.5f}", flush=True)
    print(f"[hybrid] best struct: {best_record['struct']['combo_key']} {best_record['struct']['dir']}", flush=True)
    print(f"[hybrid] best time: {best_record['time']['dir']}", flush=True)
    print(f"[hybrid] train {format_metrics(train_metrics)}", flush=True)
    if test_metrics is not None:
        print(f"[hybrid] test  {format_metrics(test_metrics)}", flush=True)
    print(
        f"[hybrid] selected test {ranking_metric_key(args.metric)}={selection_score:.5f} "
        f"test={test_score:.5f}",
        flush=True,
    )

    model_path = osp.join(out_dir, "best_hybrid_lgbm.txt")
    save_lgbm_model(best["model"], model_path)
    config = vars(args).copy()
    config.update(
        {
            "out_dir": out_dir,
            "struct_candidates": [
                {
                    "dir": item["dir"],
                    "combo_key": item.get("combo_key"),
                    "model_path": item["model_path"],
                    "score": float(item["score"]),
                }
                for item in struct_runs
            ],
            "time_candidates": [
                {"dir": item["dir"], "score": float(item["score"])} for item in time_runs
            ],
            "feature_names": hybrid_feature_builder.feature_names,
        }
    )
    summary = {
        "format": "hybrid_lgbm_v2",
        "dataset": args.dataset,
        "selection_metric": ranking_metric_key(args.metric),
        "objective_split": "test",
        "struct_preselect_metric": ranking_metric_key(effective_struct_metric(args)),
        "time_preselect_metric": ranking_metric_key(effective_time_metric(args)),
        "train_info": best_record["train_info"],
        "feature_names": hybrid_feature_builder.feature_names,
        "model_path": model_path,
        "best": {
            **best_record,
            "train_metrics": train_metrics,
            "model_path": model_path,
        },
        "top_by_validation": candidates,
        "top_by_selection": candidates,
        "selection_score": float(selection_score),
        "val_score": float(selection_score),
        "test_score": float(test_score),
        "train_metrics": train_metrics,
        "val_metrics": None,
        "runtime_sec": float(time.time() - start_time),
    }
    if test_metrics is not None:
        summary["best"]["test_metrics"] = test_metrics
        summary["test_metrics"] = test_metrics
    save_config(out_dir, config)
    save_metrics(out_dir, summary)
    with open(osp.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[hybrid] saved -> {out_dir}", flush=True)
    return summary
