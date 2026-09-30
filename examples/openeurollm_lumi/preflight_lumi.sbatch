#!/usr/bin/env bash
#SBATCH --job-name=verl-oellm-preflight
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=mi250:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.preflight.out
#SBATCH --error=logs/%j.preflight.err

set -euo pipefail
ROOT=${OELLM_VERL_ROOT:-/scratch/project_465002530/users/bmoell/verl-oellm-experiment}
EXPERIMENT="$ROOT/verl/examples/openeurollm_lumi"
IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
MODEL=/scratch/project_465002530/users/bmoell/oellm-rlvr/outputs/grpo-math-oellm9b-instruct-canary-16step-v1/lumi_grpo_math_oellm9b_instruct_canary_16step_v1__42__1790587530_checkpoints/step_16
BINDS=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
export SINGULARITYENV_PYTHONPATH="$EXPERIMENT:/opt/venv/lib/python3.12/site-packages"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT/verl"
srun --gpu-bind=none singularity exec -B "$BINDS" "$IMAGE" \
  bash "$EXPERIMENT/rocm_ray_env.sh" "$ROOT/venv/bin/python" -c \
  'import torch, ray, vllm, verl, verl.trainer.main_ppo; from transformers import AutoConfig; c=AutoConfig.from_pretrained("'$MODEL'", local_files_only=True); print("gpu_count", torch.cuda.device_count()); print("gpu", torch.cuda.get_device_name(0)); print("torch",torch.__version__,"hip",torch.version.hip,"ray",ray.__version__,"vllm",vllm.__version__,"verl",verl.__file__,"model_type",c.model_type); assert torch.cuda.device_count()==1'
