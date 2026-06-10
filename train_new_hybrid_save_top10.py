import argparse
import copy
import gc
import hashlib
import json
import os
import os.path as osp
import time
from types import SimpleNamespace

import numpy as np

from new_single_pipeline.structure_lgbm import (
    BConfig,
    HybridFeatureBuilder,
    StructureFeatureBuilder,
    build_hybrid_matrix,
    build_structure_matrix,
    ensure_dir,
    evaluate_hybrid_model,
    evaluate_score_store,
    evaluate_structure_model,
    fit_lgbm_ranker,
    format_metrics,
    metric_value,
    save_component_score_stores,
    save_lgbm_model,
)
from utils import describe_loaded_data, load_datasets, ranking_metric_key, save_config, save_metrics, select_torch_device, set_random_seed


PROTOCOL = "new_hybrid_save_top10_two_stage_v1"
DEFAULT_CONFIG = osp.join("configs", "new_hybrid_inputs.json")


STRUCTURE_PARAM_PRESETS = [
    {
        "n_estimators": 8,
        "learning_rate": 0.015116739956471236,
        "num_leaves": 15,
        "max_depth": 11,
        "min_child_samples": 147,
        "reg_lambda": 0.8241925264876453,
        "reg_alpha": 0.28383821193536135,
        "min_split_gain": 0.007404465173409036,
        "subsample": 0.8075397185632818,
        "colsample_bytree": 0.7347607178575388,
    },
    {
        "n_estimators": 4,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 8,
        "min_child_samples": 120,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "min_split_gain": 0.005,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 16,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": 10,
        "min_child_samples": 180,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "min_split_gain": 0.01,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
    },
]


HYBRID_PARAM_PRESETS = [
    {
        "n_estimators": 1,
        "learning_rate": 0.03753885387101247,
        "num_leaves": 21,
        "max_depth": 12,
        "min_child_samples": 39,
        "reg_lambda": 0.6376811127061687,
        "reg_alpha": 0.0013242769886098894,
        "min_split_gain": 0.00023728487317629075,
        "subsample": 0.990325975281362,
        "colsample_bytree": 0.7627169011118449,
    },
    {
        "n_estimators": 2,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "max_depth": 8,
        "min_child_samples": 60,
        "reg_lambda": 0.8,
        "reg_alpha": 0.01,
        "min_split_gain": 0.001,
        "subsample": 0.95,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 4,
        "learning_rate": 0.015,
        "num_leaves": 31,
        "max_depth": 10,
        "min_child_samples": 100,
        "reg_lambda": 1.5,
        "reg_alpha": 0.05,
        "min_split_gain": 0.005,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    },
]


def stable_hash(payload, length=12):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[: int(length)]


def load_inputs(path):
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


def dataset_common(inputs, dataset):
    try:
        return inputs["dataset_common"][dataset]
    except KeyError as exc:
        raise ValueError(f"dataset {dataset!r} is missing from hybrid input config") from exc


def structure_configs(inputs, dataset):
    configs = inputs["structure_top_configs"].get(dataset, [])
    if not configs:
        raise ValueError(f"no structure configs for dataset {dataset!r} in {inputs['_path']}")
    return configs


def time_runs(inputs, dataset):
    runs = inputs["time_runs"].get(dataset, [])
    if not runs:
        raise ValueError(f"no time runs for dataset {dataset!r} in {inputs['_path']}")
    return runs


def resolve_time_dir(raw_dir, time_root):
    if not time_root:
        return raw_dir
    rel = raw_dir.replace("\\", "/")
    parts = rel.split("/")
    if parts and parts[0] == "results_time_tkg_single":
        rel = "/".join(parts[1:])
    return osp.join(time_root, rel)


def make_structure_args(args, cfg, common):
    payload = dict(cfg)
    payload.update(
        {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "gpu": int(args.gpu),
            "ns_q": int(common["ns_q"]),
            "ns_seed": int(args.ns_seed),
            "train_predict_ratio": float(args.train_predict_ratio),
            "batch_size": int(cfg.get("batch_size", common.get("batch_size", 4096))),
            "query_batch_size": int(args.query_batch_size),
            "source_join_threads": int(args.source_join_threads),
            "source_join_log_batches": int(args.source_join_log_batches),
            "b_cfg": BConfig(
                mode=args.b_mode,
                binary_unseen=float(args.b_binary_unseen),
                continuous_alpha=float(args.b_continuous_alpha),
            ),
        }
    )
    return SimpleNamespace(**payload)


def make_out_dir(args, inputs):
    h = stable_hash(
        {
            "protocol": PROTOCOL,
            "dataset": args.dataset,
            "config_hash": inputs["_hash"],
            "seed": args.seed,
            "ns_seed": args.ns_seed,
            "train_predict_ratio": args.train_predict_ratio,
            "structure_topk": args.structure_train_topk,
            "hybrid_topk": args.hybrid_train_topk,
            "focus_metric": args.focus_metric,
            "b": {
                "mode": args.b_mode,
                "binary_unseen": args.b_binary_unseen,
                "continuous_alpha": args.b_continuous_alpha,
            },
        },
        12,
    )
    return osp.join(args.output_root, args.dataset, f"seed{args.seed}", f"save_top10_{h}")


def require_time_scores(time_dir, label):
    missing = []
    for split in ("train", "val", "test"):
        for suffix in ("pos.npy", "neg.npz", "valid_lens.npy", "meta.json"):
            path = osp.join(time_dir, f"{split}_{suffix}")
            if not osp.isfile(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(f"{label} missing time score files, first missing: {missing[0]}")


def train_best_structure(data, sargs, args, device, out_dir, struct_id):
    feature_builder = StructureFeatureBuilder(data["num_rels"])
    print(f"[SaveTop10][structure] build train matrix struct={struct_id}", flush=True)
    X_train, y_train, group, train_info = build_structure_matrix(
        data,
        "train",
        sargs,
        feature_builder,
        device,
        int(args.structure_train_topk),
    )
    print(
        f"[SaveTop10][structure] train rows={train_info['rows']} queries={train_info['queries']} "
        f"features={X_train.shape[1]}",
        flush=True,
    )
    best = None
    records = []
    for idx, params in enumerate(STRUCTURE_PARAM_PRESETS[: int(args.structure_param_presets)], start=1):
        print(f"[SaveTop10][structure] preset {idx} params={params}", flush=True)
        model = fit_lgbm_ranker(
            X_train,
            y_train,
            group,
            feature_builder.feature_names,
            feature_builder.categorical_indices,
            args,
            params,
        )
        val_metrics = evaluate_structure_model(data, "val", sargs, feature_builder, model, device)
        score = metric_value(val_metrics, args.structure_metric)
        print(f"[SaveTop10][structure] preset {idx} val {format_metrics(val_metrics)} score={score:.5f}", flush=True)
        rec = {
            "preset": idx,
            "params": dict(params),
            "val_metrics": val_metrics,
            "score": float(score),
            "train_info": train_info,
        }
        records.append(rec)
        if best is None or score > best["score"]:
            if best is not None:
                del best["model"]
                gc.collect()
            best = {"model": model, "score": float(score), "record": rec}
        else:
            del model
            gc.collect()
    del X_train, y_train, group
    gc.collect()
    model_path = osp.join(out_dir, "structure_models", f"{struct_id}.txt")
    save_lgbm_model(best["model"], model_path)
    test_metrics = evaluate_structure_model(data, "test", sargs, feature_builder, best["model"], device)
    print(f"[SaveTop10][structure] best struct={struct_id} test {format_metrics(test_metrics)}", flush=True)
    best["record"]["test_metrics"] = test_metrics
    best["record"]["model_path"] = model_path
    return best["model"], feature_builder, {"best": best["record"], "all": records}


def train_best_hybrid(data, sargs, args, device, out_dir, struct_id, time_run, structure_model, structure_feature_builder):
    hybrid_feature_builder = HybridFeatureBuilder(data["num_rels"])
    time_dir = time_run["dir"]
    print(f"[SaveTop10][hybrid] build train matrix struct={struct_id} time={time_run['id']}", flush=True)
    X_train, y_train, group, train_info = build_hybrid_matrix(
        data,
        "train",
        sargs,
        structure_feature_builder,
        hybrid_feature_builder,
        structure_model,
        time_dir,
        device,
        int(args.hybrid_train_topk),
    )
    print(
        f"[SaveTop10][hybrid] train rows={train_info['rows']} queries={train_info['queries']} "
        f"features={X_train.shape[1]}",
        flush=True,
    )
    best = None
    records = []
    for idx, params in enumerate(HYBRID_PARAM_PRESETS[: int(args.hybrid_param_presets)], start=1):
        print(f"[SaveTop10][hybrid] preset {idx} params={params}", flush=True)
        model = fit_lgbm_ranker(
            X_train,
            y_train,
            group,
            hybrid_feature_builder.feature_names,
            [],
            args,
            params,
        )
        val_metrics = evaluate_hybrid_model(
            data,
            "val",
            sargs,
            structure_feature_builder,
            hybrid_feature_builder,
            structure_model,
            model,
            time_dir,
            device,
        )
        score = metric_value(val_metrics, args.focus_metric)
        print(f"[SaveTop10][hybrid] preset {idx} val {format_metrics(val_metrics)} score={score:.5f}", flush=True)
        rec = {
            "preset": idx,
            "params": dict(params),
            "val_metrics": val_metrics,
            "score": float(score),
            "train_info": train_info,
        }
        records.append(rec)
        if best is None or score > best["score"]:
            if best is not None:
                del best["model"]
                gc.collect()
            best = {"model": model, "score": float(score), "record": rec}
        else:
            del model
            gc.collect()
    del X_train, y_train, group
    gc.collect()
    pair_id = f"{struct_id}__{time_run['id']}"
    model_path = osp.join(out_dir, "hybrid_models", f"{pair_id}.txt")
    save_lgbm_model(best["model"], model_path)
    top10_path = osp.join(out_dir, "top10", f"{pair_id}.test_top10.jsonl")
    test_metrics = evaluate_hybrid_model(
        data,
        "test",
        sargs,
        structure_feature_builder,
        hybrid_feature_builder,
        structure_model,
        best["model"],
        time_dir,
        device,
        save_top10_path=top10_path,
    )
    print(f"[SaveTop10][hybrid] best pair={pair_id} test {format_metrics(test_metrics)} top10={top10_path}", flush=True)
    best["record"].update(
        {
            "test_metrics": test_metrics,
            "model_path": model_path,
            "top10_path": top10_path,
            "pair_id": pair_id,
            "struct_id": struct_id,
            "time_id": time_run["id"],
            "time_dir": time_dir,
        }
    )
    return best["record"], records


def validate_args(args, inputs):
    if args.dataset not in inputs["dataset_common"]:
        raise ValueError(f"dataset {args.dataset!r} is not present in {args.hybrid_config}")
    focus = str(args.focus_metric).upper().replace("@", "")
    aliases = {"MRR": "mrr", "H1": "hr1", "HR1": "hr1", "H10": "hr10", "HR10": "hr10"}
    if focus not in aliases:
        raise ValueError("--focus_metric must be one of MRR/H1/H10")
    args.focus_metric = aliases[focus]
    sm = str(args.structure_metric).upper().replace("@", "")
    if sm not in aliases:
        raise ValueError("--structure_metric must be one of MRR/H1/H10")
    args.structure_metric = aliases[sm]
    ranking_metric_key(args.focus_metric, strict=True)
    ranking_metric_key(args.structure_metric, strict=True)
    if int(args.query_batch_size) <= 0:
        raise ValueError("--query_batch_size must be > 0")
    if int(args.structure_train_topk) == 0 or int(args.hybrid_train_topk) == 0:
        raise ValueError("topk must be -1 or positive")


def run(args):
    inputs = load_inputs(args.hybrid_config)
    validate_args(args, inputs)
    set_random_seed(args.seed)
    device = select_torch_device(args.gpu)
    common = dataset_common(inputs, args.dataset)
    args.ns_q = int(common["ns_q"])
    out_dir = ensure_dir(make_out_dir(args, inputs))
    print(f"[SaveTop10] output -> {out_dir}", flush=True)
    print(
        f"[SaveTop10] protocol={PROTOCOL} dataset={args.dataset} ns_q={args.ns_q} "
        f"focus={args.focus_metric} structure_metric={args.structure_metric}",
        flush=True,
    )
    data = load_datasets(
        args.dataset,
        q=args.ns_q,
        load_train_ratio=args.train_predict_ratio,
        load_eval_neg=True,
        ns_seed=args.ns_seed,
    )
    describe_loaded_data(data, prefix="[SaveTop10]")

    resolved_time_runs = []
    for raw in time_runs(inputs, args.dataset):
        tr = copy.deepcopy(raw)
        tr["dir"] = resolve_time_dir(tr["dir"], args.time_root)
        require_time_scores(tr["dir"], f"time {tr['id']}")
        test_metrics = evaluate_score_store(tr["dir"], data, "test", SimpleNamespace(query_batch_size=args.query_batch_size))
        tr["computed_test_metrics"] = test_metrics
        print(f"[SaveTop10][time] {tr['id']} test {format_metrics(test_metrics)} dir={tr['dir']}", flush=True)
        resolved_time_runs.append(tr)

    structure_records = []
    pair_records = []
    best_pair = None
    for cfg in structure_configs(inputs, args.dataset)[: int(args.max_structure_configs)]:
        sargs = make_structure_args(args, cfg, common)
        struct_id = cfg["id"]
        print(f"[SaveTop10] structure config {struct_id}: {cfg}", flush=True)
        t0 = time.time()
        component_dir, component_metrics = save_component_score_stores(
            data,
            sargs,
            device,
            out_dir,
            struct_id,
        )
        raw_test = component_metrics["splits"]["test"]["metrics"]["structure_raw"]
        print(
            f"[SaveTop10][structure_raw] struct={struct_id} test {format_metrics(raw_test)} "
            f"component_scores={component_dir}",
            flush=True,
        )
        structure_model, structure_feature_builder, struct_record = train_best_structure(
            data, sargs, args, device, out_dir, struct_id
        )
        struct_record["id"] = struct_id
        struct_record["config"] = cfg
        struct_record["component_score_dir"] = component_dir
        struct_record["component_metrics"] = component_metrics
        struct_record["train_new_structure_style_test_metrics"] = raw_test
        struct_record["elapsed_s"] = time.time() - t0
        structure_records.append(struct_record)
        for tr in resolved_time_runs[: int(args.max_time_configs)]:
            pair_t0 = time.time()
            best_record, all_records = train_best_hybrid(
                data,
                sargs,
                args,
                device,
                out_dir,
                struct_id,
                tr,
                structure_model,
                structure_feature_builder,
            )
            best_record["all_hybrid_presets"] = all_records
            best_record["elapsed_s"] = time.time() - pair_t0
            pair_records.append(best_record)
            score = metric_value(best_record["test_metrics"], args.focus_metric)
            if best_pair is None or score > metric_value(best_pair["test_metrics"], args.focus_metric):
                best_pair = best_record
        del structure_model
        gc.collect()

    summary = {
        "format": "new_hybrid_save_top10_summary_v1",
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "args": vars(args).copy(),
        "hybrid_config": {"path": args.hybrid_config, "hash": inputs["_hash"]},
        "time_runs": resolved_time_runs,
        "structure_records": structure_records,
        "pair_records": sorted(
            pair_records,
            key=lambda r: metric_value(r["test_metrics"], args.focus_metric),
            reverse=True,
        ),
        "best": best_pair,
    }
    save_config(out_dir, summary["args"])
    save_metrics(out_dir, summary)
    with open(osp.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[SaveTop10] best pair={best_pair['pair_id']} "
        f"test {format_metrics(best_pair['test_metrics'])} top10={best_pair['top10_path']}",
        flush=True,
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser("Two-stage new hybrid pipeline that saves test top10 per query.")
    parser.add_argument("--dataset", default="ICEWS14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--hybrid_config", default=DEFAULT_CONFIG)
    parser.add_argument("--time_root", default="")
    parser.add_argument("--output_root", default="results_new_hybrid_save_top10")
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--source_join_threads", type=int, default=60)
    parser.add_argument("--source_join_log_batches", type=int, default=0)
    parser.add_argument("--structure_train_topk", type=int, default=100)
    parser.add_argument("--hybrid_train_topk", type=int, default=100)
    parser.add_argument("--structure_metric", default="H10")
    parser.add_argument("--focus_metric", default="H10")
    parser.add_argument("--structure_param_presets", type=int, default=3)
    parser.add_argument("--hybrid_param_presets", type=int, default=3)
    parser.add_argument("--max_structure_configs", type=int, default=3)
    parser.add_argument("--max_time_configs", type=int, default=3)
    parser.add_argument("--num_threads", type=int, default=60)
    parser.add_argument("--b_mode", choices=("continuous", "binary"), default="continuous")
    parser.add_argument("--b_binary_unseen", type=float, default=0.0)
    parser.add_argument("--b_continuous_alpha", type=float, default=0.0001)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
