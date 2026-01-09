#!/usr/bin/env bash
# Submit every *.py in the current directory as its own Slurm job using run_one.slurm
set -euo pipefail

mkdir -p logs

for f in *.py; do
  # Skip if no matches (bash globbing edge case)
  [[ -e "$f" ]] || { echo "No .py files found."; exit 0; }

  name="${f%.py}"
  echo "Submitting $f"
  sbatch \
    -J "$name" \
    -o "logs/%x-%j.out" \
    --export=ALL,FILE="$f" \
    run_one.slurm
  sleep 0.2
done

echo "Done. Check queue with: squeue -u $USER"
