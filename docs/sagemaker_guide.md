# SageMaker Training Guide

*Training LaMer + language-table on AWS SageMaker*
*March 2026*

---

## 1. Prerequisites

### 1.1 AWS CLI

```bash
# Install
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Configure
aws configure
# Enter your Access Key ID, Secret Access Key, region (us-west-2), and output format (json)

# Verify
aws sts get-caller-identity
```

### 1.2 Docker

You need Docker to build the training image. On Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect
```

### 1.3 SageMaker Python SDK

Install on your local machine (where you'll launch jobs from):

```bash
pip install "sagemaker<3" boto3
```

The launch script still depends on the v2 Estimator API (`sagemaker.estimator`), so keep the SDK below 3.x when running `launch.py`.

### 1.4 IAM Role

SageMaker needs an IAM execution role with permissions for:
- ECR (pull images)
- S3 (read inputs, write checkpoints/model artifacts)
- CloudWatch (write logs)

If your AWS account already has a SageMaker execution role, use its ARN. Otherwise, create one:

```bash
# Create the role (one-time)
aws iam create-role \
    --role-name SageMakerExecutionRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach the managed SageMaker policy
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Get the ARN (put this in .env.sagemaker as SAGEMAKER_ROLE)
aws iam get-role --role-name SageMakerExecutionRole --query Role.Arn --output text
```

### 1.5 S3 Bucket

Create a bucket for checkpoints and input data:

```bash
aws s3 mb s3://your-lamer-bucket --region us-west-2
```

### 1.6 SageMaker Quota

Check your quota for the target instance type:

```bash
aws service-quotas get-service-quota \
    --service-code sagemaker \
    --quota-code L-A2C2C46C \
    --query 'Quota.Value'
```

If this returns 0, request a quota increase via the AWS console (Service Quotas → SageMaker → `ml.p4d.24xlarge for training job usage`). Quota increases for p4d instances typically take 1-3 business days.

### 1.7 Repos

The Docker build expects the language-table repo to be adjacent to LaMer:

```
parent-dir/
├── LaMer/             # this repo
└── language-table/    # language-table repo
```

If it's elsewhere, set `LANGTABLE_DIR` before running `build_and_push.sh`.

---

## 2. Configuration: `.env.sagemaker`

All scripts read from `.env.sagemaker` in the project root. Copy the example and edit it:

```bash
cp .env.sagemaker.example .env.sagemaker
```

| Variable | What it does | Example |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for ECR and SageMaker | `us-west-2` |
| `ECR_REPO_NAME` | ECR repository name for the Docker image | `lamer-langtable` |
| `SAGEMAKER_IMAGE_URI` | Full ECR image URI (set after build_and_push.sh) | `123456789.dkr.ecr.us-west-2.amazonaws.com/lamer-langtable:latest` |
| `SAGEMAKER_ROLE` | IAM execution role ARN | `arn:aws:iam::123456789:role/SageMakerExecutionRole` |
| `SAGEMAKER_INSTANCE_TYPE` | Instance type | `ml.p4d.24xlarge` |
| `VLA_S3_URI` | S3 path to VLA checkpoint directory | `s3://bucket/vla-checkpoint/` |
| `CHECKPOINT_S3_URI` | S3 path for checkpoint sync | `s3://bucket/checkpoints/` |
| `WANDB_API_KEY` | WandB API key for logging | (from wandb.ai/authorize) |
| `HF_TOKEN` | HuggingFace token for model downloads | (from huggingface.co/settings/tokens) |

`.env.sagemaker` is gitignored. Never commit secrets.

---

## 3. Running Training

1. Upload the VLA checkpoint to S3:
   ```bash
   # Download if you don't have it locally
   wget -O bc_resnet_sim_checkpoint_955000 \
       "https://storage.googleapis.com/gresearch/robotics/language_table_checkpoints/bc_resnet_sim_checkpoint_955000"
   # Upload to S3
   aws s3 cp bc_resnet_sim_checkpoint_955000 s3://your-bucket/vla-checkpoint/bc_resnet_sim_checkpoint_955000
   ```
2. Build and push the Docker image (~20-30 min first time):
   ```bash
   bash scripts/sagemaker/build_and_push.sh
   ```
3. Update `SAGEMAKER_IMAGE_URI` in `.env.sagemaker` with the URI printed by the build script
4. Launch the training job:
   ```bash
   python scripts/sagemaker/launch.py
   ```
5. Monitor logs in your terminal (streams automatically) or in the SageMaker console
6. Check WandB for training metrics

---

## 4. Script-by-Script Walkthrough

### 4.1 `scripts/sagemaker/Dockerfile` — The Training Image

A single Docker image containing both conda environments. This is the largest and most important piece.

**Layer structure:**

```
nvidia/cuda:12.4.1-devel-ubuntu22.04        (base with CUDA)
  → System packages                          (cmake, GL libs for PyBullet)
  → Miniforge                                (conda package manager)
  → Conda env "ltvenv" (Python 3.10)         (language-table + JAX + scenic)
  → Conda env "lamer" (Python 3.12)          (LaMer + PyTorch + vLLM + flash-attn)
  → COPY language-table → /opt/ml/code/      (env server code)
  → COPY LaMer → /opt/ml/code/              (training code)
  → entrypoint.sh                            (orchestrates everything)
```

The image is large (~15-20GB) because it contains two full ML stacks. Pip/conda caches are cleaned in each layer to minimize size.

**Why two conda environments?** LaMer needs Python 3.12 + PyTorch 2.6 + vLLM. Language-table needs Python 3.10 + JAX + TensorFlow. These have incompatible dependency trees — JAX and PyTorch fight over CUDA libraries, and vLLM requires Python 3.12+.

### 4.2 `scripts/sagemaker/entrypoint.sh` — The Orchestrator

Adapted from `scripts/slurm/lamer_language_table.slurm`. Runs the same 4-step flow:

1. **Start env servers** on GPU 0 (training server on port 50051, validation on 50052)
2. **Wait for ports** — polls with `nc -z` until both servers accept connections (up to 4 min)
3. **Connection test** — verifies end-to-end data flow through both servers
4. **Launch training** — preps parquet data, then calls the same `lamer_language_table_slurm.sh` training script

Key differences from the SLURM version:
- No `module load` — everything is in the Docker image
- Uses absolute paths to conda python binaries instead of `conda activate`
- Reads SageMaker env vars (`SM_CHANNEL_VLA`, `SM_NUM_GPUS`)
- Checkpoints go to `/opt/ml/checkpoints/` (auto-synced to S3 by SageMaker)
- Final model copied to `/opt/ml/model/` (uploaded to S3 as training artifact)

### 4.3 `scripts/sagemaker/launch.py` — Job Submission

Creates a SageMaker `Estimator` and calls `.fit()`. Reads config from `.env.sagemaker` and CLI args.

```bash
# Basic launch (reads everything from .env.sagemaker)
python scripts/sagemaker/launch.py

# Override via CLI
python scripts/sagemaker/launch.py \
    --instance-type ml.p4d.24xlarge \
    --run-name my_experiment_v2 \
    --vla-s3-uri s3://my-bucket/vla-checkpoint/ \
    --use-spot
```

The `--use-spot` flag enables managed spot training — SageMaker uses cheaper spot instances and automatically restarts the job if the instance is reclaimed, restoring checkpoints from S3.

### 4.4 `scripts/sagemaker/build_and_push.sh` — Docker Build + ECR Push

Builds the Docker image and pushes to Amazon ECR:

1. Creates the ECR repository if it doesn't exist
2. Authenticates Docker with ECR
3. Copies `language-table` into the build context (cleaned up after)
4. Builds the image from the Dockerfile
5. Tags and pushes to ECR

```bash
# Default (reads from .env.sagemaker)
bash scripts/sagemaker/build_and_push.sh

# Override
AWS_REGION=us-east-1 IMAGE_TAG=v2 bash scripts/sagemaker/build_and_push.sh
```

---

## 5. SageMaker Directory Mapping

When SageMaker runs your container, it mounts several special directories:

| Path | Purpose | Lifecycle |
|------|---------|-----------|
| `/opt/ml/input/data/<channel>/` | Input data from S3 (read-only) | Downloaded before training starts |
| `/opt/ml/checkpoints/` | Checkpoint directory | Auto-synced to `checkpoint_s3_uri` |
| `/opt/ml/model/` | Final model output | Uploaded to S3 after job completes |
| `/opt/ml/code/` | Your code (baked into Docker image) | Part of the image |
| `/tmp/` | Temporary storage | Instance-local, not persisted |

Our mapping:

| What | SageMaker Path | Source |
|------|---------------|--------|
| VLA checkpoint | `/opt/ml/input/data/vla/bc_resnet_sim_checkpoint_955000` | S3 input channel (`vla`) |
| HF model weights | `/tmp/hf_cache/` | Downloaded at runtime |
| Training checkpoints | `/opt/ml/checkpoints/<run_name>/` | Auto-synced to S3 |
| Parquet prompts | `/opt/ml/checkpoints/<run_name>/data/` | Generated at runtime |
| LaMer code | `/opt/ml/code/LaMer/` | Baked into image |
| language-table code | `/opt/ml/code/language-table/` | Baked into image |

---

## 6. Spot Instances and Checkpoint Resume

SageMaker managed spot training can save 60-70% on p4d instances. Here's how it works:

1. You set `--use-spot` in `launch.py` (or `SAGEMAKER_USE_SPOT=true` in `.env.sagemaker`)
2. SageMaker requests a spot instance. If none is available, it waits (up to `max_wait` seconds)
3. Training runs normally. Checkpoints are written to `/opt/ml/checkpoints/` and synced to S3
4. If AWS reclaims the spot instance, SageMaker automatically:
   - Saves a final checkpoint
   - Waits for another spot instance
   - Restores `/opt/ml/checkpoints/` from S3
   - Reruns your entrypoint from the beginning

**Important:** The verl trainer's checkpoint resume is based on finding existing checkpoints in `default_local_dir`. Since SageMaker restores `/opt/ml/checkpoints/` before rerunning the entrypoint, and our `TRAINER_LOCAL_DIR` points there, the trainer will automatically detect and resume from the latest checkpoint.

The env servers restart fresh on each spot resume (no state to restore — they're stateless simulators).

---

## 7. Monitoring

### CloudWatch Logs

SageMaker streams all stdout/stderr to CloudWatch. The `launch.py` script runs with `logs="All"`, so logs also stream to your terminal while the job is running. If you need to review logs after the fact, use the CLI or console.

**Via the SageMaker console:**
Training → Training jobs → `<job-name>` → View logs

**Via the CLI (step-by-step):**

```bash
# 1. Find your training job name
aws sagemaker list-training-jobs \
    --name-contains lamer \
    --region us-west-2 \
    --max-results 5

# This returns job names, status, and timestamps. Find the job you want.

# 2. Find the log stream for that job
aws logs describe-log-streams \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name-prefix "lamer-20260320142217" \
    --region us-west-2

# The log stream name looks like: <job-name>/algo-1-<timestamp>

# 3. Read the first 100 log events (from the start of the job)
aws logs get-log-events \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name "<job-name>/algo-1-<timestamp>" \
    --region us-west-2 \
    --start-from-head \
    --limit 100

# 4. Read the last 100 log events (end of the job — useful for finding errors)
#    Omit --start-from-head to read from the end:
aws logs get-log-events \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name "<job-name>/algo-1-<timestamp>" \
    --region us-west-2 \
    --limit 100
```

**Tip:** Pipe through `python3 -c` to extract just the log messages (the raw JSON includes timestamps and metadata):

```bash
aws logs get-log-events \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name "<job-name>/algo-1-<timestamp>" \
    --region us-west-2 \
    --limit 100 \
    --output json \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for e in data['events']:
    print(e['message'])
"
```

**Paginating through logs:** Each `get-log-events` response includes a `nextForwardToken`. Pass it as `--next-token` to get the next page of results.

### WandB

If `WANDB_API_KEY` is set, training metrics appear in your WandB dashboard under the `lamer` project. The experiment name matches `RUN_NAME`.

### SageMaker Console

The [SageMaker console](https://console.aws.amazon.com/sagemaker/) shows:
- Job status (InProgress, Completed, Failed, Stopped)
- Instance utilization (CPU, GPU, memory)
- Billable seconds and estimated cost
- Link to CloudWatch logs

---

## 8. Troubleshooting

### "ResourceLimitExceeded" when launching

Your account doesn't have quota for the requested instance type. Request a quota increase:

1. Go to AWS Console → Service Quotas → Amazon SageMaker
2. Search for `ml.p4d.24xlarge for training job usage`
3. Request an increase to at least 1

### Docker build fails on flash-attn wheel

The flash-attn prebuilt wheel is pinned to torch 2.6 + CUDA 12 + Python 3.12. If you change any of these versions, you'll need a different wheel. Check available wheels at the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases).

### Env servers fail to start (PyBullet GL errors)

The Dockerfile installs `libgl1-mesa-glx` and `libegl1-mesa` for headless rendering. If you see GL-related errors, the image may need additional mesa packages:

```dockerfile
RUN apt-get update && apt-get install -y mesa-utils libosmesa6-dev
```

### Out of GPU memory

The p4d.24xlarge has 8× A100 80GB. GPU 0 runs env servers (~3GB), GPUs 1-4 run training. If training OOMs:

- Reduce `ppo_micro_batch_size_per_gpu` from 16 to 8
- Reduce `gpu_memory_utilization` from 0.6 to 0.5
- Reduce `max_num_batched_tokens` from 32768 to 16384
- These are set in `examples/language_table/lamer_language_table_slurm.sh`

### Training job stuck at "Downloading"

SageMaker downloads S3 input channels before starting your container. If the VLA checkpoint is large and in a different region than your SageMaker job, this can be slow. Keep your S3 bucket in the same region as the SageMaker job.

### Checkpoint restore not working after spot interruption

Verify that:
1. `CHECKPOINT_S3_URI` is set in `.env.sagemaker`
2. The checkpoint files are actually in S3: `aws s3 ls <checkpoint_s3_uri> --recursive`
3. The `TRAINER_LOCAL_DIR` in `entrypoint.sh` matches `checkpoint_local_path` (`/opt/ml/checkpoints/<run_name>`)

---

## 9. Cost Estimate

| Instance | GPUs | Hourly Cost (On-Demand) | Hourly Cost (Spot) |
|----------|------|------------------------|-------------------|
| ml.p4d.24xlarge | 8× A100 80GB | ~$32/hr | ~$10-13/hr |

A 24-hour training run costs roughly **$770 on-demand** or **$240-310 on spot**.

Compared to Azure (~$59/hr for 3 VMs), SageMaker is cheaper per hour and simpler to operate. Spot instances make it significantly cheaper still.

**Cost controls:**
- Set `max_run` in `launch.py` to cap job duration (default: 24 hours)
- Use spot instances for non-critical runs
- Monitor in the SageMaker console — you can stop a job at any time

