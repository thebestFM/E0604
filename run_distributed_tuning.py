import argparse
import hashlib
import json
import os
import os.path as osp

from train_all import common_params, make_a_args, make_c_args, make_time_args, parse_param_overrides
from single_pipeline import a_single, c_single, time_single
from single_pipeline.a_single import tune_a
from single_pipeline.c_single import tune_c
from single_pipeline.tuning import run_hybrid, run_hybrid_from_manifest, run_struct, strict_val_score, tune_b, tune_time
from utils import load_metrics, ranking_metric_key


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[distributed] saved -> {path}", flush=True)


def stable_hash(payload, length=8):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[: int(length)]


def metric_token(metric):
    key = ranking_metric_key(metric, strict=True)
    if key == "mrr_strict":
        return "mrr"
    if key.startswith("hit@") and key.endswith("_strict"):
        return "h" + key[len("hit@") : -len("_strict")]
    return key.replace("@", "").replace("_strict", "")


def protocol_token(args):
    return (
        f"p-nq{args.ns_q}-ns{args.ns_seed}"
        f"_tr{args.train_predict_ratio:g}"
        f"_cub{int(bool(getattr(args, 'close_update_backward', False)))}"
    )


def search_payload(args):
    return {
        "component_metric": ranking_metric_key(args.component_metric, strict=True),
        "time_metric": ranking_metric_key(args.time_metric, strict=True),
        "struct_metric": ranking_metric_key(args.struct_metric, strict=True),
        "hybrid_metric": ranking_metric_key(args.hybrid_metric, strict=True),
        "top_k_a": int(args.top_k_a),
        "top_k_b": int(args.top_k_b),
        "top_k_c": int(args.top_k_c),
        "top_k_time": int(args.top_k_time),
        "top_k_struct": int(args.top_k_struct),
        "all_trials": int(args.all_trials),
        "train_topk": int(args.train_topk),
        "top_structure_combine_train": int(getattr(args, "top_structure_combine_train", -1)),
        "top_hybrid_train": int(getattr(args, "top_hybrid_train", -1)),
        "hybrid_include_structure_features": bool(getattr(args, "hybrid_include_structure_features", False)),
        "ignore_val_structure_combine": bool(getattr(args, "ignore_val_structure_combine", False)),
        "lgbm_n_trials": int(args.lgbm_n_trials),
        "lgbm_early_stopping_rounds": int(args.lgbm_early_stopping_rounds),
        "lgbm_eval_tail_fraction": float(getattr(args, "lgbm_eval_tail_fraction", 0.3)),
        "time_param": list(getattr(args, "time_param", []) or []),
        "a_param": list(getattr(args, "a_param", []) or []),
        "c_param": list(getattr(args, "c_param", []) or []),
    }


def search_token(args):
    payload = search_payload(args)
    return (
        f"m-c{metric_token(args.component_metric)}-t{metric_token(args.time_metric)}"
        f"-s{metric_token(args.struct_metric)}-h{metric_token(args.hybrid_metric)}"
        f"_k-a{args.top_k_a}-b{args.top_k_b}-c{args.top_k_c}"
        f"-tt{args.top_k_time}-ts{args.top_k_struct}"
        f"_r{stable_hash(payload)}"
    )


def record_dir(args):
    return osp.join(
        args.records_dir,
        args.dataset,
        f"seed{args.seed}",
        protocol_token(args),
        search_token(args),
    )


def record_path(args, name):
    return osp.join(record_dir(args), name)


def save_records(args, name, records):
    payload = {"top_by_validation": records}
    save_json(record_path(args, name), payload)
    return payload


def load_record_payload(args, name):
    path = record_path(args, name)
    if not osp.isfile(path):
        root = osp.join(args.records_dir, args.dataset, f"seed{args.seed}", protocol_token(args))
        matches = []
        for dirpath, _, filenames in os.walk(root) if osp.isdir(root) else []:
            if name in filenames:
                matches.append(osp.join(dirpath, name))
        if len(matches) == 1:
            path = matches[0]
            print(f"[distributed] using compatible {name} -> {path}", flush=True)
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"{record_path(args, name)} not found, and multiple compatible {name} files exist under {root}: "
                + ", ".join(matches)
            )
    return load_json(path)


def load_records(args, name):
    return load_record_payload(args, name)["top_by_validation"]


def save_record_metadata(args):
    save_json(
        record_path(args, "metadata.json"),
        {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "ns_q": int(args.ns_q),
            "ns_seed": int(args.ns_seed),
            "train_predict_ratio": float(args.train_predict_ratio),
            "close_update_backward": bool(getattr(args, "close_update_backward", False)),
            "search": search_payload(args),
            "record_dir": record_dir(args),
        },
    )


def run_stage(args):
    save_record_metadata(args)
    common = common_params(args)

    if args.stage == "run_a":
        params = parse_param_overrides(args.a_param)
        a_args = make_a_args(args, common, params)
        metrics = a_single.main(a_args)
        score = strict_val_score(metrics, args.component_metric)
        record = {
            "rank_source": "single_validation",
            "score": score,
            "metric": f"val_{ranking_metric_key(args.component_metric, strict=True)}",
            "params": params,
            "out_dir": a_single.get_out_dir(a_args),
            "args": vars(a_args).copy(),
        }
        return save_records(args, "a_top.json", [record])

    if args.stage == "tune_a":
        a_args = make_a_args(args, common, {})
        metric_key = f"val_{ranking_metric_key(args.component_metric, strict=True)}"
        records = tune_a(a_args, top_k=args.top_k_a, metric=metric_key)
        return save_records(args, "a_top.json", records)

    if args.stage == "run_c":
        params = parse_param_overrides(args.c_param)
        c_args = make_c_args(args, common, params)
        metrics = c_single.main(c_args)
        score = strict_val_score(metrics, args.component_metric)
        record = {
            "rank_source": "single_validation",
            "score": score,
            "metric": f"val_{ranking_metric_key(args.component_metric, strict=True)}",
            "params": params,
            "out_dir": c_single.make_c_result_dir(c_args, c_args.gamma),
            "args": vars(c_args).copy(),
        }
        return save_records(args, "c_top.json", [record])

    if args.stage == "tune_c":
        c_args = make_c_args(args, common, {})
        records = tune_c(
            c_args,
            all_trials=args.all_trials,
            top_k=args.top_k_c,
            metric=f"val_{ranking_metric_key(args.component_metric, strict=True)}",
        )
        return save_records(args, "c_top.json", records)

    if args.stage == "tune_b":
        records = tune_b(
            args,
            top_k=args.top_k_b,
            metric=args.component_metric,
            out_dir=osp.join(record_dir(args), "b_raw"),
        )
        return save_records(args, "b_top.json", records)

    if args.stage == "run_time":
        params = parse_param_overrides(args.time_param)
        time_args = make_time_args(args, common, params)
        time_single.main(time_args)
        out_dir = time_single.get_out_dir(time_args)
        metrics = load_metrics(out_dir)
        score = strict_val_score(metrics, args.time_metric)
        record = {
            "rank_source": "single_validation",
            "score": score,
            "metric": f"val_{ranking_metric_key(args.time_metric, strict=True)}",
            "params": params,
            "out_dir": out_dir,
            "args": vars(time_args).copy(),
        }
        return save_records(args, "time_top.json", [record])

    if args.stage == "tune_time":
        time_args = make_time_args(args, common, {})
        records = tune_time(time_args, top_k=args.top_k_time, metric=args.time_metric)
        return save_records(args, "time_top.json", records)

    if args.stage == "run_struct":
        a_records = load_records(args, "a_top.json")
        b_records = load_records(args, "b_top.json")
        c_records = load_records(args, "c_top.json")
        summary = run_struct(
            args,
            a_records,
            b_records,
            c_records,
            top_k_a=args.top_k_a,
            top_k_b=args.top_k_b,
            top_k_c=args.top_k_c,
            metric=args.struct_metric,
        )
        save_json(record_path(args, "struct_top.json"), summary)
        return summary

    if args.stage == "run_hybrid":
        if args.hybrid_manifest and osp.isfile(args.hybrid_manifest):
            manifest_payload = load_json(args.hybrid_manifest)
            manifest_datasets = manifest_payload.get("datasets", manifest_payload)
            if args.dataset in manifest_datasets:
                summary = run_hybrid_from_manifest(
                    args,
                    args.hybrid_manifest,
                    top_k_time=args.top_k_time,
                    top_k_struct=args.top_k_struct,
                    metric=args.hybrid_metric,
                )
                save_json(record_path(args, "hybrid_top.json"), summary)
                return summary
            print(
                f"[distributed] {args.dataset} not in hybrid manifest {args.hybrid_manifest}; "
                "falling back to records_dir",
                flush=True,
            )
        time_records = load_records(args, "time_top.json")
        struct_summary = load_record_payload(args, "struct_top.json")
        summary = run_hybrid(
            args,
            time_records,
            struct_summary,
            top_k_time=args.top_k_time,
            top_k_struct=args.top_k_struct,
            metric=args.hybrid_metric,
        )
        save_json(record_path(args, "hybrid_top.json"), summary)
        return summary

    raise ValueError(f"unknown stage: {args.stage}")


def load_args():
    parser = argparse.ArgumentParser(
        "Distributed validation-only tuning entry points for the single EAGLE++ pipeline."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("run_a", "run_c", "run_time", "tune_a", "tune_c", "tune_b", "tune_time", "run_struct", "run_hybrid"),
    )
    parser.add_argument("--dataset", type=str, default="tkgl-icews")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=2000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--records_dir", type=str, default="tuning_records")
    parser.add_argument("--hybrid_manifest", type=str, default="best_hyper_params/hybrid_component_paths.json")

    parser.add_argument("--top_k_a", type=int, default=3)
    parser.add_argument("--top_k_b", type=int, default=3)
    parser.add_argument("--top_k_c", type=int, default=10)
    parser.add_argument("--top_k_time", type=int, default=3)
    parser.add_argument("--top_k_struct", type=int, default=3)
    parser.add_argument("--all_trials", type=int, default=10)

    parser.add_argument("--component_metric", type=str, default="mrr")
    parser.add_argument("--struct_metric", type=str, default="hr10")
    parser.add_argument("--time_metric", type=str, default="mrr")
    parser.add_argument("--hybrid_metric", type=str, default="mrr")

    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--train_topk", type=int, default=100)
    parser.add_argument("--top_structure_combine_train", type=int, default=-1)
    parser.add_argument("--top_hybrid_train", type=int, default=200)
    parser.add_argument("--hybrid_include_structure_features", action="store_true", default=False)
    parser.add_argument("--ignore_val_structure_combine", action="store_true", default=False)
    parser.add_argument("--lgbm_n_trials", type=int, default=None)
    parser.add_argument("--lgbm_early_stopping_rounds", type=int, default=50)
    parser.add_argument("--lgbm_eval_tail_fraction", type=float, default=0.3)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--min_child_samples", type=int, default=50)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--min_split_gain", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)

    parser.add_argument("--time_batch_size", type=int, default=4096)
    parser.add_argument("--time_eval_batch_size", type=int, default=512)
    parser.add_argument("--time_eval_neg_chunk", type=int, default=5000)
    parser.add_argument("--time_max_eval_pairs", type=int, default=2500000)
    parser.add_argument("--time_quick_val_fraction", type=float, default=0.3)
    parser.add_argument("--time_eval_node_preload_chunk", type=int, default=131072)
    parser.add_argument("--time_max_eval_node_cache_mb", type=float, default=8192.0)
    parser.add_argument("--time_train_group_matrix_mb", type=float, default=2048.0)
    parser.add_argument("--a_batch_size", type=int, default=1024)
    parser.add_argument("--c_batch_size", type=int, default=1024)
    parser.add_argument("--c_max_events_in_single_batch", type=int, default=20000)
    parser.add_argument("--source_join_threads", type=int, default=0)
    parser.add_argument("--source_join_log_batches", type=int, default=0)
    parser.add_argument("--force_time", action="store_true", default=False)
    parser.add_argument("--force_hybrid", action="store_true", default=False)
    parser.add_argument("--close_update_backward", action="store_true", default=False)
    parser.add_argument("--time_param", action="append", default=[])
    parser.add_argument("--a_param", action="append", default=[])
    parser.add_argument("--c_param", action="append", default=[])
    args = parser.parse_args()
    if args.lgbm_n_trials is None:
        args.lgbm_n_trials = 10 if args.dataset == "ICEWS14" else 5
    return args


if __name__ == "__main__":
    run_stage(load_args())
