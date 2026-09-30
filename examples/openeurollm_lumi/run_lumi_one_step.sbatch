#!/usr/bin/env bash
# Standalone verl GRPO compatibility qualification; no oellm-rlvr runtime.
#SBATCH --job-name=verl-oellm9b-math-1step
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=mi250:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

ROOT=${OELLM_VERL_ROOT:-/scratch/project_465002530/users/bmoell/verl-oellm-experiment}
EXPERIMENT="$ROOT/verl/examples/openeurollm_lumi"
MODEL=/scratch/project_465002530/users/bmoell/oellm-rlvr/outputs/grpo-math-oellm9b-instruct-canary-16step-v1/lumi_grpo_math_oellm9b_instruct_canary_16step_v1__42__1790587530_checkpoints/step_16
IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
RUN="$ROOT/runs/${SLURM_JOB_ID}"
BINDS=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd

test -f "$MODEL/.checkpoint_complete"
test -s "$MODEL/model.safetensors"
test -s "$ROOT/data-20260930-v2/train.parquet"
test -s "$ROOT/data-20260930-v2/validation.parquet"
test -f "$IMAGE"
mkdir -p "$RUN" "$JOB_TMPDIR/verl-model" "$JOB_TMPDIR/verl-ray"
git -C "$ROOT/verl" rev-parse HEAD
sha256sum "$ROOT/data-20260930-v2/train.parquet" "$ROOT/data-20260930-v2/validation.parquet" "$EXPERIMENT/reward.py"

# Copy one immutable 9B export to node-local storage before worker initialization.
cp -a "$MODEL"/. "$JOB_TMPDIR/verl-model"/
test -s "$JOB_TMPDIR/verl-model/model.safetensors"

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$EXPERIMENT:/opt/venv/lib/python3.12/site-packages"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export RAY_TMPDIR="$JOB_TMPDIR/verl-ray"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

cd "$ROOT/verl"
srun --gpu-bind=none singularity exec -B "$BINDS" "$IMAGE" \
  bash "$EXPERIMENT/rocm_ray_env.sh" "$ROOT/venv/bin/python" -c \
  'import torch, ray, vllm, verl; print("GPU", torch.cuda.device_count(), torch.cuda.get_device_name(0)); print("Ray", ray.__version__, "vLLM", vllm.__version__)'

srun --gpu-bind=none singularity exec -B "$BINDS" "$IMAGE" \
  bash "$EXPERIMENT/rocm_ray_env.sh" "$ROOT/venv/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$ROOT/data-20260930-v2/train.parquet" \
  data.val_files="$ROOT/data-20260930-v2/validation.parquet" \
  data.train_batch_size=8 data.max_prompt_length=1024 data.max_response_length=2048 \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=False \
  reward.custom_reward_function.path="$EXPERIMENT/reward.py" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$JOB_TMPDIR/verl-model" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  trainer.nnodes=1 trainer.n_gpus_per_node=8 \
  trainer.project_name=oellm_standalone_verl \
  trainer.experiment_name="math_one_step_${SLURM_JOB_ID}" \
  trainer.logger='["console"]' trainer.val_before_train=False \
  trainer.test_freq=-1 trainer.save_freq=1 \
  trainer.total_epochs=1 trainer.total_training_steps=1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$RUN/checkpoints" \
  trainer.rollout_data_dir="$RUN/rollouts" \
  ray_kwargs.ray_init.num_cpus=48 \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
  ray_kwargs.ray_init.runtime_env.py_executable="$ROOT/venv/bin/python"
