#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs-0609

COMMON="--dataset ICEWS14 --seed 42 --gpu 0 --ns_q 6000 --ns_seed 42 --train_predict_ratio 0.3 --batch_size 8192 --max_events_in_single_batch 60000 --source_join_threads 60 --source_join_log_batches 0 --skip_val_eval"
DSH_FIXED="--direct_single_hop PLACEHOLDER_DSH --decay_direct 1.0"
SHARED_FIXED="--gamma 0 --top_share 100 --top_direct -1 --decay_rel_trans 0.05 --window_semantic_sim 5.0 --window_trans 5.0 --shared_w dual_msim"

run_case() {
  local case_id="$1"
  local direct_single_hop="$2"
  local dict_mode="$3"
  local ppr_alpha="$4"
  local ppr_beta="$5"
  local top_k_relation="$6"
  local log_file="logs-0609/icews14_add_dmh_${case_id}_dsh${direct_single_hop}_${dict_mode}_a${ppr_alpha}_b${ppr_beta}.log"

  local dsh_args="${DSH_FIXED/PLACEHOLDER_DSH/${direct_single_hop}}"
  echo "[run] ${log_file}"
  nohup python train_new_structure.py \
    ${COMMON} \
    ${dsh_args} \
    ${SHARED_FIXED} \
    --dict_mode "${dict_mode}" \
    --ppr_k 1000 \
    --top_k_relation "${top_k_relation}" \
    --ppr_alpha "${ppr_alpha}" \
    --ppr_beta "${ppr_beta}" \
    > "${log_file}" 2>&1 &
  wait $!
}

for dsh in 0.95 0.90 0.80; do
  run_case "c01" "${dsh}" "tag_sum" 0.012 0.92 0
  run_case "c02" "${dsh}" "tag_sum" 0.020 0.95 0
  run_case "c03" "${dsh}" "tag_max" 0.015 0.93 0
  run_case "c04" "${dsh}" "tag_sum" 0.035 0.97 0
  run_case "c05" "${dsh}" "tag_max" 0.050 0.90 0
done
