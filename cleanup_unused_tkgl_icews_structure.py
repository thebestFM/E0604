#!/usr/bin/env python3
"""Delete tkgl-icews structure runs not used by the current fast hybrid config.

Dry run by default:

    python cleanup_unused_tkgl_icews_structure.py

Actually delete, expecting the current server state of 21 removable runs:

    python cleanup_unused_tkgl_icews_structure.py --delete

The keep set is read from configs/new_hybrid_inputs_tkgl_icews_fast.json, which
currently keeps only the two pure-DSH runs used by tkgl-icews hybrid:
decay_direct=0.5 and decay_direct=0.8.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


COMPARE_KEYS = (
    "batch_size",
    "max_events_in_single_batch",
    "dict_mode",
    "shared_w",
    "per_rel_use_mtrans",
    "ppr_k",
    "top_k_relation",
    "ppr_alpha",
    "ppr_beta",
    "gamma",
    "direct_single_hop",
    "decay_direct",
    "top_share",
    "top_direct",
    "decay_rel_trans",
    "window_semantic_sim",
    "window_trans",
    "close_update_backward",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def equivalent(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return str(a) == str(b)


def matches_spec(config: dict, spec: dict) -> bool:
    for key in COMPARE_KEYS:
        if key not in config:
            return False
        if not equivalent(config[key], spec[key]):
            return False
    return True


def matching_spec_id(config: dict, specs: list[dict]) -> str | None:
    for spec in specs:
        if matches_spec(config, spec):
            return str(spec["id"])
    return None


def load_keep_specs(path: Path) -> list[dict]:
    payload = load_json(path)
    if payload.get("format") != "new_hybrid_inputs_v1":
        raise ValueError(f"unsupported hybrid config format: {path}")
    specs = list(payload.get("structure_top_configs", {}).get("tkgl-icews", []))
    if not specs:
        raise ValueError(f"no tkgl-icews structure specs found in {path}")
    return specs


def iter_tkgl_icews_runs(root: Path):
    dataset_dir = root / "tkgl-icews"
    if not dataset_dir.is_dir():
        return
    for seed_dir in sorted(dataset_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        for run_dir in sorted(seed_dir.iterdir()):
            if run_dir.is_dir():
                yield seed_dir.name, run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete unused results_new_structure/tkgl-icews runs."
    )
    parser.add_argument("--root", default="results_new_structure")
    parser.add_argument(
        "--hybrid-config",
        default="configs/new_hybrid_inputs_tkgl_icews_fast.json",
    )
    parser.add_argument("--delete", action="store_true")
    parser.add_argument(
        "--expected-remove",
        type=int,
        default=21,
        help="Safety check for the current server snapshot. Use -1 to disable.",
    )
    parser.add_argument(
        "--delete-unknown",
        action="store_true",
        help="Also delete run directories without a readable config.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    keep_specs = load_keep_specs(Path(args.hybrid_config))

    keep = []
    remove = []
    unknown = []

    for seed, run_dir in iter_tkgl_icews_runs(root):
        cfg_path = run_dir / "config.json"
        if not cfg_path.is_file():
            unknown.append((seed, run_dir, "missing config.json"))
            continue
        try:
            config = load_json(cfg_path)
        except Exception as exc:
            unknown.append((seed, run_dir, f"bad config.json: {exc}"))
            continue

        spec_id = matching_spec_id(config, keep_specs)
        if spec_id is None:
            remove.append((seed, run_dir))
        else:
            keep.append((seed, run_dir, spec_id))

    print(f"[cleanup-tkgl-icews] root={root}")
    print(f"[cleanup-tkgl-icews] hybrid_config={args.hybrid_config}")
    print(f"[cleanup-tkgl-icews] keep={len(keep)} remove={len(remove)} unknown={len(unknown)}")
    for seed, run_dir, spec_id in keep:
        print(f"[KEEP] tkgl-icews/{seed}/{run_dir.name}  # {spec_id}")
    for seed, run_dir in remove:
        print(f"[REMOVE] tkgl-icews/{seed}/{run_dir.name}")
    for seed, run_dir, reason in unknown:
        tag = "REMOVE_UNKNOWN" if args.delete_unknown else "KEEP_UNKNOWN"
        print(f"[{tag}] tkgl-icews/{seed}/{run_dir.name}  # {reason}")

    if not args.delete:
        print("[cleanup-tkgl-icews] dry-run only. Re-run with --delete to remove [REMOVE] entries.")
        return

    if args.expected_remove >= 0 and len(remove) != args.expected_remove:
        raise SystemExit(
            f"refusing to delete: expected {args.expected_remove} removable runs, found {len(remove)}. "
            "Inspect the dry-run output, then pass --expected-remove -1 if this is intentional."
        )

    for _, run_dir in remove:
        shutil.rmtree(run_dir)
    if args.delete_unknown:
        for _, run_dir, _ in unknown:
            shutil.rmtree(run_dir)
    print("[cleanup-tkgl-icews] delete complete.")


if __name__ == "__main__":
    main()
