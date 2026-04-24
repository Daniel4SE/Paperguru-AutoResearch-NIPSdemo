#!/bin/bash
# E1 main CIFAR-10 comparison watcher.
# Runs the four quantizer configurations serially on a single GPU.
# Survives the death of any individual run: the next run starts only
# when the previous ckpt_final.pt appears.
#
# Usage:
#   nohup bash scripts/run_e1.sh > /tmp/e1.log 2>&1 &
#
# Output:
#   results/cifar10_{vanilla,rotation,fsq,gumbel}_e1/{tb,ckpt_*.pt,config.yaml}
#   results/E1_logs/{vanilla,rotation,fsq,gumbel}.log
#   results/E1_logs/DONE      (touched when all four runs complete)

set -u
cd "$(dirname "$0")/.."

source ~/venv/bin/activate 2>/dev/null || true
unset LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

mkdir -p results/E1_logs

COMMON_OVERRIDE="train.max_steps=12000 \
                 train.log_every=200 \
                 train.val_every=1000 \
                 train.save_every=6000 \
                 train.max_val_batches=20 \
                 train.batch_size=1024 \
                 train.lr=1.6e-3 \
                 train.lr_min=1.0e-6 \
                 train.amp=false \
                 train.grad_clip=1.0 \
                 data.num_workers=0"

wait_for_ckpt() {
  local ckpt=$1
  local max_wait=3000  # 50-minute guard
  local t=0
  while [ ! -f "$ckpt" ] && [ $t -lt $max_wait ]; do
    sleep 30; t=$((t + 30))
  done
  if [ -f "$ckpt" ]; then
    echo "$(date +%T) DETECTED $ckpt after ${t}s" >> /tmp/watcher.log
  else
    echo "$(date +%T) TIMEOUT waiting for $ckpt"  >> /tmp/watcher.log
  fi
}

launch() {
  local cfg=$1
  echo "$(date +%T) starting $cfg" >> /tmp/watcher.log
  python -u train.py --config configs/${cfg}.yaml --device cuda \
    --override $COMMON_OVERRIDE \
               logging.out_dir=~/vq-rotation/results/${cfg}_e1 \
    > results/E1_logs/${cfg}.log 2>&1
  echo "$(date +%T) finished $cfg" >> /tmp/watcher.log
}

echo "=== run_e1 started $(date +%T) ===" > /tmp/watcher.log

# If vanilla is already running (e.g., re-attach scenario), wait for it;
# otherwise launch it.
if [ ! -f "results/cifar10_vanilla_e1/ckpt_final.pt" ]; then
  if ! pgrep -f "configs/cifar10_vanilla.yaml" >/dev/null; then
    launch cifar10_vanilla
  else
    wait_for_ckpt results/cifar10_vanilla_e1/ckpt_final.pt
  fi
fi

wait_for_ckpt results/cifar10_vanilla_e1/ckpt_final.pt
launch cifar10_rotation

wait_for_ckpt results/cifar10_rotation_e1/ckpt_final.pt
launch cifar10_fsq

wait_for_ckpt results/cifar10_fsq_e1/ckpt_final.pt
launch cifar10_gumbel

echo "ALL_E1_DONE $(date +%T)" > results/E1_logs/DONE
echo "=== run_e1 done $(date +%T) ===" >> /tmp/watcher.log
