#!/bin/bash
# scripts/monitor_training.sh
# Watch training progress in real-time
# Usage: bash scripts/monitor_training.sh

LOG="logs/train_walk_full.log"
CKPT_DIR="runs/h1_walk_full/checkpoints"

echo "========================================"
echo "  H1 Walk Training Monitor"
echo "  Log:  $LOG"
echo "  Ctrl+C to stop"
echo "========================================"

while true; do
    clear
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo ""

    # Process info
    PID=$(pgrep -f "train_walk.py" 2>/dev/null)
    if [ -z "$PID" ]; then
        echo "⚠ Training process NOT running!"
    else
        CPU=$(ps -p $PID -o %cpu --no-headers 2>/dev/null | tr -d ' ')
        MEM=$(ps -p $PID -o %mem --no-headers 2>/dev/null | tr -d ' ')
        echo "Process: PID=$PID  CPU=${CPU}%  RAM=${MEM}%"
    fi
    echo ""

    # Last 15 lines of log
    echo "--- Training Log (last 15 lines) ---"
    tail -15 "$LOG" 2>/dev/null || echo "(log not found)"
    echo ""

    # Checkpoints
    echo "--- Checkpoints ---"
    ls -lh "$CKPT_DIR"/checkpoint_step_*.pt 2>/dev/null | tail -5 || echo "(none yet)"
    echo ""

    # GPU usage
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | \
            awk -F',' '{printf "GPU: util=%s%%  mem=%s/%sMB  temp=%s°C\n", $1, $2, $3, $4}'
    fi

    sleep 30
done
