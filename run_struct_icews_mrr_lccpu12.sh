#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

COMMON="--dataset tkgl-icews --seed 42 --ns_q 5000 --ns_seed 42 --train_predict_ratio 0.3 --records_dir tuning_records_struct_icews_mrr --component_metric mrr --struct_metric mrr --num_threads 112 --block_size 1536 --eval_batch_size 1536 --train_topk 500 --top_structure_combine_train 1200 --lgbm_n_trials 10 --lgbm_early_stopping_rounds 80 --n_estimators 2000 --learning_rate 0.025 --num_leaves 127 --max_depth -1 --min_child_samples 100 --reg_lambda 1.0 --reg_alpha 0.001 --subsample 0.9 --colsample_bytree 0.9 --a_batch_size 8192 --c_batch_size 4096 --c_max_events_in_single_batch 60000 --source_join_threads 112 --source_join_log_batches 0"

nohup python run_distributed_tuning.py $COMMON --stage tune_a > logs/struct_icews_tune_a.log 2>&1 &
PID_A=$!

nohup python run_distributed_tuning.py $COMMON --stage tune_b > logs/struct_icews_tune_b.log 2>&1 &
PID_B=$!

nohup python run_distributed_tuning.py $COMMON --stage tune_c --all_trials 12 --top_k_c 8 --c_param top_direct=500 > logs/struct_icews_tune_c.log 2>&1 &
PID_C=$!

wait $PID_A
wait $PID_B
wait $PID_C

nohup python run_distributed_tuning.py $COMMON --stage run_struct --top_k_a 3 --top_k_b 3 --top_k_c 8 > logs/struct_icews_run_struct.log 2>&1
