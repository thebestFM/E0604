import argparse
import os
import shutil
from pathlib import Path


TRASH_DIR = "useless_0609"


KEEP_WHOLE_DIRS = {
    ".git",
    "data",
}


KEEP_FILES = {
    ".gitignore",
    "train_new_structure.py",
    "train_new_hybrid.py",
    "utils.py",
    "cleanup_trusted_0609.py",
    "check_new_hybrid_icews14_latest_scores.py",
    "check_structure_score_anomalies.py",
    "check_icews14_new_a_scores.py",
    "single_pipeline/structure_combine_single.py",
}


KEEP_TIME_RUNS = {
    "results_time_tkg_single/ICEWS14/seed42/"
    "r9eb5b85515d8_topk30_mw5-15-30_ed96_hd192_bs4096_ebs384_neg6_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin",
    "results_time_tkg_single/ICEWS14/seed42/"
    "r210529791eed_topk40_mw5-15-30-60_ed96_hd192_bs4096_ebs384_neg8_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin",
    "results_time_tkg_single/ICEWS14/seed42/"
    "r041812cea350_topk70_mw5-15-30-60-120_ed96_hd192_bs4096_ebs384_neg4_samgroup_nsq6000_nss42_tpr0.3_abs1r0_gatechannel_rank1_lossmargin",
    "results_time_tkg_single/GDELT/seed42/"
    "rec9ebf5506ad_topk60_mw7-30_ed64_hd128_bs2048_ebs192_neg4_samgroup_nsq5000_nss42_tpr0.3_abs1r0_gateoff_rank1_lossmargin",
    "results_time_tkg_single/tkgl-polecat/seed42/"
    "r59bae37154aa_topk80_mw30_ed64_hd128_bs2048_ebs128_neg2_samgroup_nsq5000_nss42_tpr0.3_abs1r0_gateoff_rank1_lossmargin",
}


KEEP_STRUCTURE_RUNS = {
    "results_new_structure/ICEWS14/seed42/"
    "decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.95_gamma=0_impl=new_structure_hb8a1ef33f08b",
    "results_new_structure/ICEWS14/seed42/"
    "decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.9_gamma=0_impl=new_structure__h0cd799355526",
    "results_new_structure/ICEWS14/seed42/"
    "decay_direct=1_decay_rt=0.05_dict_mode=tag_sum_direct_single_hop=0.8_gamma=0_impl=new_structure__h13a42747ada6",
}


KEEP_PATHS = KEEP_FILES | KEEP_TIME_RUNS | KEEP_STRUCTURE_RUNS


def norm_rel(path):
    return path.as_posix().strip("/")


def is_same_or_child(rel, parent):
    return rel == parent or rel.startswith(parent + "/")


def has_kept_descendant(rel):
    return any(is_same_or_child(keep, rel) for keep in KEEP_PATHS)


def should_keep(path, rel):
    if rel == TRASH_DIR or rel.startswith(TRASH_DIR + "/"):
        return True
    if rel in KEEP_FILES or rel in KEEP_TIME_RUNS or rel in KEEP_STRUCTURE_RUNS:
        return True
    if path.is_dir() and rel in KEEP_WHOLE_DIRS:
        return True
    if path.is_dir() and has_kept_descendant(rel):
        return True
    return False


def unique_destination(root, rel):
    dst = root / TRASH_DIR / rel
    if not dst.exists():
        return dst
    base = dst
    i = 1
    while True:
        candidate = Path(f"{base}.moved{i}")
        if not candidate.exists():
            return candidate
        i += 1


def move_path(root, path, rel, apply):
    dst = unique_destination(root, rel)
    print(f"[MOVE] {rel} -> {norm_rel(dst.relative_to(root))}")
    if not apply:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dst))


def clean_dir(root, current, apply):
    for child in sorted(current.iterdir(), key=lambda p: p.name):
        rel = norm_rel(child.relative_to(root))
        if should_keep(child, rel):
            if child.is_dir() and rel not in KEEP_WHOLE_DIRS and rel != TRASH_DIR:
                clean_dir(root, child, apply)
            continue
        move_path(root, child, rel, apply)


def remove_empty_dirs(root, apply):
    preserved = {TRASH_DIR}
    preserved.update(KEEP_WHOLE_DIRS)
    preserved.update(str(Path(p).parent).replace("\\", "/") for p in KEEP_PATHS)

    for current, dirs, files in os.walk(root, topdown=False):
        path = Path(current)
        if path == root:
            continue
        rel = norm_rel(path.relative_to(root))
        if rel in preserved or rel.startswith(TRASH_DIR + "/"):
            continue
        try:
            if not any(path.iterdir()):
                print(f"[RMDIR] {rel}")
                if apply:
                    path.rmdir()
        except OSError:
            pass


def validate_kept_paths(root):
    missing = []
    for rel in sorted(KEEP_TIME_RUNS | KEEP_STRUCTURE_RUNS):
        if not (root / rel).exists():
            missing.append(rel)
    if missing:
        print("[WARN] Some trusted result paths do not exist under this root:")
        for rel in missing:
            print(f"  - {rel}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Move untrusted or unnecessary files into useless_0609 while preserving "
            "trusted time results, verified ICEWS14 structure-v2 scores, and the "
            "current new-structure/new-hybrid code."
        )
    )
    parser.add_argument("--root", default=".", help="Repository/result root to clean.")
    parser.add_argument("--apply", action="store_true", help="Actually move files. Default is dry-run.")
    parser.add_argument(
        "--remove_empty_dirs",
        action="store_true",
        help="After moving, remove empty directories outside the preserved tree.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"root is not a directory: {root}")
    trash = root / TRASH_DIR
    if args.apply:
        trash.mkdir(exist_ok=True)

    print(f"[ROOT] {root}")
    print(f"[MODE] {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"[TRASH] {trash}")
    validate_kept_paths(root)
    clean_dir(root, root, args.apply)
    if args.remove_empty_dirs:
        remove_empty_dirs(root, args.apply)
    print("[DONE]")


if __name__ == "__main__":
    main()
