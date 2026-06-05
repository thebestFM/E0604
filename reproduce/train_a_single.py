import argparse
import json
import os.path as osp

import numpy as np

from single_pipeline import a_single


EXPECTED_EVAL_QUERIES = {
    "ICEWS14": {"val": 27646, "test": 26444},
    "GDELT": {"val": 477530, "test": 610482},
    "tkgl-polecat": {"val": 533472, "test": 532636},
}


def strict_value(metrics, split, name):
    key = f"{split}_{name}_strict"
    if key not in metrics:
        raise KeyError(f"missing metric {key!r} in metrics.json")
    return float(metrics[key])


def verify_split(out_dir, split, ns_q, dataset):
    lens_path = osp.join(out_dir, f"{split}_valid_lens.npy")
    meta_path = osp.join(out_dir, f"{split}_meta.json")
    if not osp.isfile(lens_path) or not osp.isfile(meta_path):
        raise FileNotFoundError(f"missing score output files for split={split!r} under {out_dir}")

    lens = np.load(lens_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_rows = EXPECTED_EVAL_QUERIES.get(dataset, {}).get(split)
    if expected_rows is not None and int(meta.get("num_rows", -1)) != int(expected_rows):
        raise RuntimeError(
            f"{split} query count mismatch: got {meta.get('num_rows')}, expected {expected_rows}"
        )
    if expected_rows is not None and int(lens.shape[0]) != int(expected_rows):
        raise RuntimeError(f"{split} valid_lens rows mismatch: got {lens.shape[0]}, expected {expected_rows}")

    if int(ns_q) > 0 and (lens.shape[0] == 0 or not np.all(lens == int(ns_q))):
        bad = int(np.sum(lens != int(ns_q)))
        raise RuntimeError(f"{split} negative count mismatch: {bad} rows do not have ns_q={int(ns_q)}")

    print(
        f"[A-repro] verified {split}: queries={int(meta.get('num_rows', lens.shape[0]))} "
        f"negatives_per_query={int(ns_q)}",
        flush=True,
    )


def print_strict_metrics(metrics, splits):
    for split in splits:
        print(
            f"[A-repro] {split}_strict "
            f"MRR={strict_value(metrics, split, 'mrr'):.5f} "
            f"HR@1={strict_value(metrics, split, 'hit@1'):.5f} "
            f"HR@10={strict_value(metrics, split, 'hit@10'):.5f}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser("Run one A-component configuration and report strict ranking metrics.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ns_q", type=int, required=True)
    parser.add_argument("--ns_seed", type=int, default=42)
    parser.add_argument("--train_predict_ratio", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--a_mode", choices=("energy", "rank"), required=True)
    parser.add_argument("--decay_a", type=float, required=True)
    parser.add_argument("--ppr_beta", type=float, required=True)
    parser.add_argument("--a_score_topk", type=int, default=0)
    parser.add_argument("--close_update_backward", action="store_true", default=False)
    parser.add_argument("--no_eval_test", action="store_true", default=False)
    args = parser.parse_args()
    args.eval_test = not args.no_eval_test
    return args


def main():
    args = parse_args()
    metrics = a_single.main(args)
    out_dir = a_single.get_out_dir(args)
    splits = ["val"] + (["test"] if args.eval_test else [])
    for split in splits:
        verify_split(out_dir, split, args.ns_q, args.dataset)
    print_strict_metrics(metrics, splits)
    print(f"[A-repro] output_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
