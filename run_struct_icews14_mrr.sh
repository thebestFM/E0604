#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

COMMON="--dataset ICEWS14 --seed 42 --ns_q 6000 --ns_seed 42 --train_predict_ratio 0.3 --records_dir tuning_records_struct_icews14_mrr_q6k --component_metric mrr --struct_metric mrr --num_threads 40 --block_size 1024 --eval_batch_size 1024 --train_topk 300 --top_structure_combine_train 1200 --lgbm_n_trials 12 --lgbm_early_stopping_rounds 80 --n_estimators 2200 --learning_rate 0.025 --num_leaves 127 --max_depth -1 --min_child_samples 80 --reg_lambda 1.0 --reg_alpha 0.001 --subsample 0.9 --colsample_bytree 0.9 --a_batch_size 8192 --c_batch_size 4096 --c_max_events_in_single_batch 40000 --source_join_threads 64 --source_join_log_batches 0"

nohup python run_distributed_tuning.py $COMMON --stage tune_a > logs/struct_icews14_mrr_q6k_tune_a.log 2>&1 &
PID_A=$!

nohup python run_distributed_tuning.py $COMMON --stage tune_b > logs/struct_icews14_mrr_q6k_tune_b.log 2>&1 &
PID_B=$!

nohup python run_distributed_tuning.py $COMMON --stage tune_c --all_trials 12 --top_k_c 8 > logs/struct_icews14_mrr_q6k_tune_c.log 2>&1 &
PID_C=$!

wait $PID_A
wait $PID_B
wait $PID_C

nohup python run_distributed_tuning.py $COMMON --stage run_struct --top_k_a 3 --top_k_b 3 --top_k_c 8 > logs/struct_icews14_mrr_q6k_run_struct.log 2>&1
