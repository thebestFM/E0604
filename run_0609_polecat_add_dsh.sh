#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs-0609

COMMON="--dataset tkgl-polecat --seed 42 --gpu 0 --ns_q 5000 --ns_seed 42 --train_predict_ratio 0.3 --batch_size 4096 --max_events_in_single_batch 60000 --source_join_threads 60 --source_join_log_batches 0 --skip_val_eval"
DMH_SHARED_FIXED="--dict_mode tag_sum --shared_w dual_msim --ppr_k 1000 --top_k_relation 0 --ppr_alpha 0.01579502319249557 --ppr_beta 0.9343207039457382 --gamma 0.01 --top_share 100 --top_direct -1 --decay_rel_trans 0.05 --window_semantic_sim 5.0 --window_trans 5.0"

run_case() {
  local case_id="$1"
  local direct_single_hop="$2"
  local decay_direct="$3"
  local log_file="logs-0609/polecat_add_dsh_${case_id}_dsh${direct_single_hop}_decay${decay_direct}.log"

  echo "[run] ${log_file}"
  nohup python train_new_structure.py \
    ${COMMON} \
    ${DMH_SHARED_FIXED} \
    --direct_single_hop "${direct_single_hop}" \
    --decay_direct "${decay_direct}" \
    > "${log_file}" 2>&1 &
  wait $!
}

for dsh in 0.05 0.10 0.20; do
  run_case "a01" "${dsh}" 0.005
  run_case "a02" "${dsh}" 0.01
  run_case "a03" "${dsh}" 0.05
  run_case "a04" "${dsh}" 0.10
  run_case "a05" "${dsh}" 0.50
done
