# OpenEuroLLM 9B math GRPO on LUMI, directly with verl

This example uses verl's own `verl.trainer.main_ppo` entry point. It does **not** import, launch,
or require `oellm-rlvr`. The fork branch is based on verl `v0.9.1` at
`1876b06d0a3e4e71e06230be10af14492ca8a75b`; all OpenEuroLLM-specific files are in this
folder. The public upstream ROCm guide validates MI300/MI350, not LUMI's MI250X, so this is a
compatibility qualification, not a production recipe yet.

The first run starts from the step-16 OpenEuroLLM 9B instruct-SFT math canary export (job
`22398668`). The first batch is six EU-language and two English-replay prompts from
`birgermoell/oellm-math-rlvr`, each sampled eight times. The standalone reward awards 1 only
when the last complete boxed integer or rational equals ground truth; response format alone earns
no reward. One optimizer update and a readable checkpoint are the target, **not** a quality claim.

## Layout and inputs

The scripts assume a LUMI scratch workspace, defaulting to
`ROOT=/scratch/project_465002530/users/bmoell/verl-oellm-experiment`, with the fork checked out
at `$ROOT/verl`, an isolated overlay at `$ROOT/venv`, generated Parquets at
`$ROOT/data-20260930-v2`, and logs at `$ROOT/logs`. The batch scripts accept
`OELLM_VERL_ROOT` to change the root, but the checkpoint, image, account, and source paths are
currently specific to this LUMI project; inspect and change them for another allocation.

The source model, EU math pool, English replay pool, and frozen EU evaluation pool must already
exist on LUMI scratch. Compute nodes run offline. The data converter refuses to overwrite
existing output and rejects train/evaluation semantic-group overlap. Its train Parquet has 16
rows, ordered so the first eight-row batch contains six EU and two English prompts. The
validation Parquet has 16 disjoint EU prompts; the one-step job disables validation because its
purpose is backend qualification. Evaluate a successful checkpoint separately before scaling.

## Environment and execution

Use LUMI's AI Factory ROCm image; **do not run verl's CUDA-oriented `uv sync`** on MI250X.
The current qualification uses the image's optimized PyTorch 2.10 and vLLM 0.22.1, with an
isolated Python overlay for verl and missing pure-Python packages. The tested overlay used
Ray 2.55.1, Hydra 1.3.7, TensorDict 0.10.0, TorchData 0.11.0, Codetiming 1.4.0,
Pylatexenc 2.11, and Mathruler 0.1.0. Install verl editable with `--no-deps`; allowing pip or
uv to replace the image's ROCm GPU packages would invalidate the qualification.

Run `prepare_data.py` inside the image with `--eu-train`, `--en-train`, `--holdout`, and
`--output-dir "$ROOT/data-20260930-v2"`. Record its printed input/output hashes, then run
`python -m unittest discover -s tests` from this folder. Submit the small import/GPU check
first, then inspect its result before submitting training:

```bash
cd "$ROOT"
mkdir -p logs
sbatch verl/examples/openeurollm_lumi/preflight_lumi.sbatch
# Only after that job completes 0:0 and reports the allocated MI250X:
sbatch verl/examples/openeurollm_lumi/run_lumi_one_step.sbatch
```

The preflight takes one GCD for at most 10 minutes. The training script takes eight GCDs on
one LUMI-G node for at most 90 minutes, copies the immutable 9B checkpoint to node-local
storage, and writes to `$ROOT/runs/JOB_ID`. `rocm_ray_env.sh` maps Slurm's GPU visibility mask
to `HIP_VISIBLE_DEVICES` before importing Ray; it never broadens the allocation.

## Evidence and next gate

On 2026-09-30, preflight job `22447840` passed on a real MI250X with verl, Ray, vLLM, and
the OpenEuroLLM checkpoint config. One-update training job `22447867` was submitted but was
still **pending (Priority)** when this documentation was written. No verl optimizer result or
checkpoint is claimed yet. The TMAX Phase A job `22432251` is separate and was also queued.

Before a longer run, require a completed group of 64 rollouts, useful mixed-reward groups,
one finite nonzero-gradient update, successful learner-to-rollout weight refit, and a readable
checkpoint. Then compare against the exact same frozen parent and held-out prompts. Online
reward by itself cannot establish improvement or compare frameworks fairly.
