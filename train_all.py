import argparse
import copy
import json
import os
import os.path as osp
from types import SimpleNamespace

PROTOCOL_KEYS = {
    "dataset",
    "seed",
    "ns_q",
    "ns_seed",
    "train_predict_ratio",
    "gpu",
    "batch_size",
    "eval_batch_size",
    "eval_neg_chunk",
    "eval_node_preload_chunk",
    "max_eval_node_cache_mb",
    "train_group_matrix_mb",
    "max_events_in_single_batch",
    "source_join_threads",
    "source_join_log_batches",
    "num_threads",
    "close_update_backward",
}


def load_best_params(dataset, path=""):
    if not path:
        path = osp.join("best_hyper_params", dataset, "structure_best_params.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("dataset") and payload["dataset"] != dataset:
        raise ValueError(f"best params dataset mismatch: {payload['dataset']} vs {dataset}")
    payload["_path"] = path
    return payload


def parse_param_value(text):
    lowered = str(text).lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def parse_param_overrides(items):
    params = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"override key is empty: {item}")
        if key in PROTOCOL_KEYS:
            raise ValueError(f"{key} is a protocol/runtime setting; use the dedicated CLI argument instead")
        params[key] = parse_param_value(value.strip())
    return params


def merge_overrides(params, overrides):
    merged = dict(params)
    merged.update(overrides)
    return merged


def single_params(payload, component):
    raw = payload.get(component)
    if raw is None:
        return {}
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(f"{component} must contain exactly one entry; found {len(raw)}")
        raw = raw[0]
    if isinstance(raw, dict) and "params" in raw:
        params = dict(raw["params"])
        return {key: value for key, value in params.items() if key not in PROTOCOL_KEYS}
    if isinstance(raw, dict):
        return {key: value for key, value in raw.items() if key not in PROTOCOL_KEYS}
    raise ValueError(f"unsupported {component} params format: {type(raw).__name__}")


def set_attrs(namespace, **kwargs):
    for key, value in kwargs.items():
        setattr(namespace, key, value)
    return namespace


def apply_params(namespace, params):
    for key, value in params.items():
        setattr(namespace, key, value)
    return namespace


def common_params(cli):
    return {
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "ns_q": int(cli.ns_q),
        "ns_seed": int(cli.ns_seed),
        "train_predict_ratio": float(cli.train_predict_ratio),
    }


def make_time_args(cli, common, params):
    args = SimpleNamespace(
        dataset=common["dataset"],
        seed=common["seed"],
        gpu=cli.gpu,
        batch_size=cli.time_batch_size,
        eval_batch_size=cli.time_eval_batch_size,
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        cache_eval_source=False,
        topk=15,
        train_num_neg=4,
        hard_neg_ratio=0.5,
        train_sampler="grouped_exact",
        train_group_matrix_mb=cli.time_train_group_matrix_mb,
        use_neighbor_id=False,
        use_abs_time=False,
        abs_time_periods=None,
        abs_time_harmonics=1,
        abs_time_use_raw=False,
        use_query_gate=False,
        query_gate_type="channel",
        use_rank_pos=False,
        multi_windows="",
        use_cross_history=False,
        cross_heads=2,
        event_encoder="mixer",
        transformer_heads=2,
        transformer_ff_dim=None,
        time_dim=64,
        rel_dim=64,
        node_dim=64,
        event_dim=96,
        hidden_dim=192,
        num_layers=1,
        dropout=0.1,
        time_min=1.0,
        token_expansion_factor=0.5,
        channel_expansion_factor=4.0,
        use_single_layer=False,
        predictor_mode="diag",
        num_epochs=50,
        patience=5,
        selection_metric="mrr",
        quick_val_events=0,
        quick_val_fraction=cli.time_quick_val_fraction,
        lr=1e-3,
        weight_decay=5e-5,
        temperature=1.0,
        train_loss="margin",
        rank_margin=1.0,
        grad_clip=1.0,
        tolerance=1e-8,
        curriculum_decay=0.0,
        curriculum_raw_age=False,
        eval_neg_chunk=cli.time_eval_neg_chunk,
        max_eval_pairs=cli.time_max_eval_pairs,
        eval_node_preload_chunk=cli.time_eval_node_preload_chunk,
        preload_eval_nodes=True,
        dense_eval_node_cache=True,
        max_eval_node_cache_mb=cli.time_max_eval_node_cache_mb,
        profile_sync=False,
        force=cli.force_time,
    )
    apply_params(args, params)
    return set_attrs(
        args,
        dataset=common["dataset"],
        seed=common["seed"],
        gpu=cli.gpu,
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        force=cli.force_time,
    )


def make_a_args(cli, common, params):
    args = SimpleNamespace(
        dataset=common["dataset"],
        seed=common["seed"],
        batch_size=cli.a_batch_size,
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        close_update_backward=bool(getattr(cli, "close_update_backward", False)),
        a_mode="rank",
        decay_a=1.0,
        ppr_beta=0.9,
    )
    apply_params(args, params)
    return set_attrs(
        args,
        dataset=common["dataset"],
        seed=common["seed"],
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        close_update_backward=bool(getattr(cli, "close_update_backward", False)),
        batch_size=cli.a_batch_size,
    )


def make_c_args(cli, common, params):
    args = SimpleNamespace(
        dataset=common["dataset"],
        seed=common["seed"],
        gpu=cli.gpu,
        batch_size=cli.c_batch_size,
        max_events_in_single_batch=cli.c_max_events_in_single_batch,
        source_join_threads=cli.source_join_threads,
        source_join_log_batches=cli.source_join_log_batches,
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        close_update_backward=bool(getattr(cli, "close_update_backward", False)),
        c_storage="tag_sum",
        shared_w="dual_msim",
        per_rel_use_mtrans=False,
        ppr_k=1500,
        top_k_relation=0,
        ppr_alpha=0.03,
        ppr_beta=0.93,
        gamma=0.01,
        top_share=100,
        top_direct=-1,
        decay_rel_trans=0.05,
        window_semantic_sim=5.0,
        window_trans=5.0,
        decay_level="timestamp",
    )
    apply_params(args, params)
    return set_attrs(
        args,
        dataset=common["dataset"],
        seed=common["seed"],
        gpu=cli.gpu,
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        close_update_backward=bool(getattr(cli, "close_update_backward", False)),
        batch_size=cli.c_batch_size,
        max_events_in_single_batch=cli.c_max_events_in_single_batch,
    )


def make_combine_args(cli, common, a_dir, c_dir, b_params):
    args = SimpleNamespace(
        dataset=common["dataset"],
        seed=common["seed"],
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        metric="hr10",
        component_metric="mrr",
        # These two prefixes are part of the original combo-key hash. The actual
        # score files are still read from the explicit *_single a_dir/c_dir below.
        a_prefix="results_a",
        c_prefix="results_c",
        out_prefix="results_lgbm_single",
        a_dir=a_dir,
        c_dir=c_dir,
        top_a=3,
        top_b=3,
        top_c=8,
        close_update_backward=bool(getattr(cli, "close_update_backward", False)),
        block_size=256,
        train_topk=100,
        top_structure_combine_train=cli.top_structure_combine_train,
        ignore_val=bool(cli.ignore_val_structure_combine),
        eval_batch_size=cli.eval_batch_size,
        b_modes="binary,continuous",
        b_mode="continuous",
        b_binary_unseen=0.0,
        b_continuous_alpha=0.0001,
        binary_unseen_grid="0,0.0003,0.001,0.003,0.01,0.03,0.1",
        continuous_alpha_grid="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_depth=-1,
        min_split_gain=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        lgbm_n_trials=0,
        lgbm_early_stopping_rounds=50,
        num_threads=cli.num_threads,
        print_top=20,
    )
    mapping = {
        "mode": "b_mode",
        "binary_unseen": "b_binary_unseen",
        "continuous_alpha": "b_continuous_alpha",
    }
    for src, dst in mapping.items():
        if src in b_params:
            setattr(args, dst, b_params[src])
    return args


def make_hybrid_args(cli, common, struct_dir, time_dir, struct_combo_key):
    return SimpleNamespace(
        dataset=common["dataset"],
        seed=common["seed"],
        ns_q=common["ns_q"],
        ns_seed=common["ns_seed"],
        train_predict_ratio=common["train_predict_ratio"],
        metric="hr10",
        struct_metric="",
        time_metric="",
        struct_prefix="results_lgbm_single",
        time_prefix="results_time_tkg_single",
        out_prefix="results_hybrid_lgbm_single",
        struct_dir=struct_dir,
        struct_combo_key=struct_combo_key,
        time_dir=time_dir,
        block_size=128,
        top_k_struct=1,
        top_k_time=1,
        train_topk=100,
        top_hybrid_train=cli.top_hybrid_train,
        hybrid_include_structure_features=bool(cli.hybrid_include_structure_features),
        eval_batch_size=cli.eval_batch_size,
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_depth=-1,
        min_split_gain=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        lgbm_n_trials=0,
        lgbm_early_stopping_rounds=50,
        num_threads=cli.num_threads,
        force=cli.force_hybrid,
    )


def protocol_dir_name(common, close_update_backward=False):
    return (
        f"p-nq{common['ns_q']}-ns{common['ns_seed']}"
        f"_tr{common['train_predict_ratio']:g}"
        f"_cub{int(bool(close_update_backward))}"
    )


def save_pipeline_summary(common, close_update_backward, summary):
    out_dir = osp.join(
        "results_all_single",
        common["dataset"],
        f"seed{common['seed']}",
        protocol_dir_name(common, close_update_backward),
    )
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[all] saved pipeline summary -> {path}", flush=True)


def run_pipeline(cli):
    from single_pipeline import a_single, c_single, hybrid_single, structure_combine_single, time_single

    payload = (
        {"dataset": cli.dataset, "_path": "<defaults-plus-cli-overrides>"}
        if cli.ignore_best_params
        else load_best_params(cli.dataset, cli.best_params_file)
    )
    time_params = single_params(payload, "time")
    a_params = single_params(payload, "a")
    b_params = single_params(payload, "b")
    c_params = single_params(payload, "c")
    time_params = merge_overrides(time_params, parse_param_overrides(cli.time_param))
    a_params = merge_overrides(a_params, parse_param_overrides(cli.a_param))
    b_params = merge_overrides(b_params, parse_param_overrides(cli.b_param))
    c_params = merge_overrides(c_params, parse_param_overrides(cli.c_param))
    combine_params = parse_param_overrides(cli.combine_param)
    hybrid_params = parse_param_overrides(cli.hybrid_param)
    common = common_params(cli)

    time_args = make_time_args(cli, common, time_params)
    a_args = make_a_args(cli, common, a_params)
    c_args = make_c_args(cli, common, c_params)

    time_dir = time_single.get_out_dir(copy.deepcopy(time_args))
    a_dir = a_single.get_out_dir(a_args)
    c_dir = c_single.make_c_result_dir(c_args, c_args.gamma)

    print(f"[all] params file -> {payload['_path']}", flush=True)
    print(f"[all] A -> {a_dir}", flush=True)
    a_metrics = a_single.main(a_args)

    print(f"[all] C -> {c_dir}", flush=True)
    c_metrics = c_single.main(c_args)

    combine_args = make_combine_args(cli, common, a_dir, c_dir, b_params)
    apply_params(combine_args, combine_params)
    struct_dir = structure_combine_single.make_out_dir(combine_args)
    print(f"[all] structure combine -> {struct_dir}", flush=True)
    combine_summary = structure_combine_single.run_search(combine_args)
    struct_combo_key = combine_summary.get("best", {}).get("combo_key")
    if not struct_combo_key:
        raise RuntimeError("structure combine did not return a best combo_key")

    print(f"[all] time -> {time_dir}", flush=True)
    time_result = time_single.main(time_args)
    time_dir = time_single.get_out_dir(time_args)

    hybrid_args = make_hybrid_args(cli, common, struct_dir, time_dir, struct_combo_key)
    apply_params(hybrid_args, hybrid_params)
    print(f"[all] hybrid with combo={struct_combo_key}", flush=True)
    hybrid_summary = hybrid_single.run(hybrid_args)
    hybrid_model_path = hybrid_summary.get("model_path", "")

    pipeline_summary = {
        "format": "eagle_single_pipeline_v1",
        "dataset": common["dataset"],
        "seed": common["seed"],
        "ns_q": common["ns_q"],
        "ns_seed": common["ns_seed"],
        "train_predict_ratio": common["train_predict_ratio"],
        "close_update_backward": bool(getattr(cli, "close_update_backward", False)),
        "best_params_file": payload["_path"],
        "cli_param_overrides": {
            "time": parse_param_overrides(cli.time_param),
            "a": parse_param_overrides(cli.a_param),
            "b": parse_param_overrides(cli.b_param),
            "c": parse_param_overrides(cli.c_param),
            "combine": parse_param_overrides(cli.combine_param),
            "hybrid": parse_param_overrides(cli.hybrid_param),
        },
        "time_dir": time_dir,
        "a_dir": a_dir,
        "c_dir": c_dir,
        "structure_dir": struct_dir,
        "structure_combo_key": struct_combo_key,
        "hybrid_dir": osp.dirname(hybrid_model_path) if hybrid_model_path else "",
        "hybrid_model_path": hybrid_model_path,
        "time_result": time_result,
        "a_metrics": a_metrics,
        "c_metrics": c_metrics,
        "structure_test_metrics": combine_summary.get("best", {}).get("test_metrics", {}),
        "hybrid_test_metrics": hybrid_summary.get("test_metrics", {}),
    }
    save_pipeline_summary(common, getattr(cli, "close_update_backward", False), pipeline_summary)
    return pipeline_summary


def load_args():
    parser = argparse.ArgumentParser("Run the single-best EAGLE++ TKG pipeline end to end.")
    parser.add_argument("--dataset", type=str, default="ICEWS14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, default=1000)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--best_params_file", type=str, default="")
    parser.add_argument("--ignore_best_params", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--top_structure_combine_train", type=int, default=-1)
    parser.add_argument("--ignore_val_structure_combine", action="store_true", default=False)
    parser.add_argument("--top_hybrid_train", type=int, default=200)
    parser.add_argument("--hybrid_include_structure_features", action="store_true", default=False)
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
    parser.add_argument("--b_param", action="append", default=[])
    parser.add_argument("--c_param", action="append", default=[])
    parser.add_argument("--combine_param", action="append", default=[])
    parser.add_argument("--hybrid_param", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(load_args())
