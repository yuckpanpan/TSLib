#!/usr/bin/env bash
set -euo pipefail

PY="/home/wangpenglei/micromamba/envs/torch_base/bin/python"
RUN="run.py"

DATA="windows_t0"
ROOT="./dataset/"
FILE="14_14_with_Y_PV.npz"

SEQ=336
PRED=24

ENC=100
DEC=100
COUT=1

NUM_WORKERS=2

# 通用默认（你可以改）
EPOCHS=40
PATIENCE=8
BS=16
LR=0.0003

COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --data "${DATA}"
  --root_path "${ROOT}"
  --data_path "${FILE}"
  --features MS
  --seq_len "${SEQ}"
  --pred_len "${PRED}"
  --enc_in "${ENC}"
  --dec_in "${DEC}"
  --c_out "${COUT}"
  --train_epochs "${EPOCHS}"
  --batch_size "${BS}"
  --patience "${PATIENCE}"
  --learning_rate "${LR}"
  --num_workers "${NUM_WORKERS}"
  --inverse 1
  --use_amp 0
)

run_one () {
  local model="$1"
  local model_id="$2"
  shift 2

  local logdir="./sweep_logs"
  mkdir -p "${logdir}"
  local logfile="${logdir}/${model_id}.log"

  echo "============================================================"
  echo "[START] $(date '+%F %T')  model=${model}  model_id=${model_id}"
  echo "log -> ${logfile}"
  echo "============================================================"

  # 串行执行：前一个结束后才会执行下一个
  ${PY} -u ${RUN} \
    "${COMMON[@]}" \
    --model "${model}" \
    --model_id "${model_id}" \
    "$@" 2>&1 | tee "${logfile}"

  echo "============================================================"
  echo "[DONE ] $(date '+%F %T')  model=${model}  model_id=${model_id}"
  echo "============================================================"
  echo
}

# =========================
# 1) 线性基线（强烈建议先看）
# =========================
run_one DLinear "pv_dlinear_ll0_lr3e-4_bs16" \
  --label_len 0

# 如果你仓库有 NLinear，取消注释
# run_one NLinear "pv_nlinear_ll0_lr3e-4_bs16" \
#   --label_len 0

# =========================
# 2) Autoformer（小模型 / 中模型 + label_len 扫描）
# =========================
run_one Autoformer "pv_autoformer_small_ll24" \
  --label_len 24 \
  --d_model 32 --n_heads 4 --e_layers 1 --d_layers 1 --d_ff 128 --dropout 0.1

run_one Autoformer "pv_autoformer_mid_ll48" \
  --label_len 48 \
  --d_model 64 --n_heads 4 --e_layers 2 --d_layers 1 --d_ff 256 --dropout 0.1

run_one Autoformer "pv_autoformer_mid_ll0" \
  --label_len 0 \
  --d_model 64 --n_heads 4 --e_layers 2 --d_layers 1 --d_ff 256 --dropout 0.1

run_one Autoformer "pv_autoformer_mid_ll96" \
  --label_len 96 \
  --d_model 64 --n_heads 4 --e_layers 2 --d_layers 1 --d_ff 256 --dropout 0.1

# =========================
# 3) Informer / FEDformer / iTransformer（如果仓库包含）
# =========================
run_one Informer "pv_informer_small_ll24" \
  --label_len 24 \
  --d_model 32 --n_heads 4 --e_layers 1 --d_layers 1 --d_ff 128 --dropout 0.1

run_one FEDformer "pv_fedformer_small_ll24" \
  --label_len 24 \
  --d_model 32 --n_heads 4 --e_layers 1 --d_layers 1 --d_ff 128 --dropout 0.1

run_one iTransformer "pv_itransformer_small_ll24" \
  --label_len 24 \
  --d_model 32 --n_heads 4 --e_layers 2 --d_ff 128 --dropout 0.1

echo "ALL DONE. Logs are in ./sweep_logs/"