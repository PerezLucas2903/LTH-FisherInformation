#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --partition=p_general
#SBATCH --job-name=SVHN
#SBATCH --output=./logs/%x-%j.out

# Fail fast and be strict
set -euo pipefail

# Ensure logs directory exists
mkdir -p logs

# ------------- LOG DE QUAL MÁQUINA ESTAMOS -----------------
node="$(hostname -s).sense.dcc.ufmg.br"
echo -e "Alocado na máquina: ${node}\n"
# -----------------------------------------------------------

# Print conda version (good sanity check)
conda --version

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create / activate env (idempotent if already created)
conda create -y -n ml_script python=3.10 || true
conda activate ml_script

# Remove any prior conda torch bits (ignore errors)
conda remove -y pytorch pytorch-cuda torchaudio torchvision 2>/dev/null || true

# Install PyTorch cu121 wheels via pip
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121

# Your other deps
pip install numpy==2.0 wandb==0.21.3 h5py==3.9.0 singd==0.0.5 matplotlib tqdm pytorch_msssim lpips

# Some clusters set this; make sure it's clean
unset LD_PRELOAD || true

# NCCL / networking environment (adjust if your fabric differs)
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=^mlx5_2:1

# Show primary node IP (optional debug)
# shellcheck disable=SC2207
nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes[0]}" hostname --ip-address)
echo "Rank 0: Node ${nodes[0]}, IP: ${head_node_ip}"

# ----- Run the specific Python file passed via --export=FILE=... -----
if [[ -z "${FILE:-}" ]]; then
  echo "ERROR: FILE env var not set. Submit with: sbatch --export=ALL,FILE=<your.py> run_one.slurm"
  exit 2
fi

echo ">>> Running: ${FILE}"
srun python "${FILE}"
