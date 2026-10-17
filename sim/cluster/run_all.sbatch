#!/bin/bash
# SLURM job for the full evaluation on a 20 GB H100 slice.
# Fill in the placeholders (<...>) for the KTH cluster before submitting:  sbatch cluster/run_all.sbatch
#SBATCH --job-name=hidden-object-eval
#SBATCH --partition=<partition>
#SBATCH --account=<project-account>
#SBATCH --gres=gpu:<gpu-slice-name>:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"            # the sim/ folder
source .venv/bin/activate
mkdir -p logs

DATA=${DATA:-out/test}

# All stages in one job; the job resumes from results/trials/*.json if it is restarted.
python run_all.py --data "$DATA" \
    --vlm Qwen/Qwen3.5-9B Qwen/Qwen3-VL-8B-Instruct \
    --primary-vlm Qwen/Qwen3.5-9B \
    --load-in-8bit
