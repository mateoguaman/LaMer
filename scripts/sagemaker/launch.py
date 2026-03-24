#!/usr/bin/env python3
"""Launch a LaMer + language-table training job on SageMaker.

Usage:
    # Using .env.sagemaker config:
    python scripts/sagemaker/launch.py

    # Override via CLI:
    python scripts/sagemaker/launch.py \
        --instance-type ml.p4d.24xlarge \
        --run-name my_experiment \
        --vla-s3-uri s3://my-bucket/vla-checkpoint/
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def load_env_file(path: str) -> None:
    """Load a .env file into os.environ (simple key=value, no shell expansion)."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch LaMer training on SageMaker")
    parser.add_argument(
        "--image-uri",
        default=os.environ.get("SAGEMAKER_IMAGE_URI"),
        help="ECR image URI (default: from SAGEMAKER_IMAGE_URI env var)",
    )
    parser.add_argument(
        "--role",
        default=os.environ.get("SAGEMAKER_ROLE"),
        help="IAM role ARN for SageMaker (default: from SAGEMAKER_ROLE env var)",
    )
    parser.add_argument(
        "--instance-type",
        default=os.environ.get("SAGEMAKER_INSTANCE_TYPE", "ml.p4d.24xlarge"),
        help="SageMaker instance type (default: ml.p4d.24xlarge)",
    )
    parser.add_argument(
        "--volume-size",
        type=int,
        default=int(os.environ.get("SAGEMAKER_VOLUME_SIZE", "200")),
        help="EBS volume size in GB (default: 200)",
    )
    parser.add_argument(
        "--max-run",
        type=int,
        default=int(os.environ.get("SAGEMAKER_MAX_RUN", "86400")),
        help="Max runtime in seconds (default: 86400 = 24h)",
    )
    parser.add_argument(
        "--run-name",
        default=os.environ.get("RUN_NAME", "language_table_lamer_qwen3_4b"),
        help="Experiment/run name",
    )
    parser.add_argument(
        "--vla-s3-uri",
        default=os.environ.get("VLA_S3_URI"),
        help="S3 URI for VLA checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-s3-uri",
        default=os.environ.get("CHECKPOINT_S3_URI"),
        help="S3 URI for checkpoint sync",
    )
    parser.add_argument(
        "--use-spot",
        action="store_true",
        default=os.environ.get("SAGEMAKER_USE_SPOT", "").lower() in ("1", "true"),
        help="Use spot instances (with automatic checkpoint restore)",
    )
    return parser.parse_args()


def main() -> None:
    # Load .env.sagemaker from repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    load_env_file(repo_root / ".env.sagemaker")

    args = parse_args()

    if not args.image_uri:
        print("ERROR: --image-uri or SAGEMAKER_IMAGE_URI is required", file=sys.stderr)
        sys.exit(1)
    if not args.role:
        print("ERROR: --role or SAGEMAKER_ROLE is required", file=sys.stderr)
        sys.exit(1)

    # Import here so CLI arg parsing works even without sagemaker installed locally
    import boto3
    from sagemaker.estimator import Estimator

    # Create a clean source tarball using git archive (excludes .git/ and
    # respects .gitignore, so logs, virtualenvs, etc. are omitted).
    region = os.environ.get("AWS_REGION", "us-west-2")
    account = boto3.client("sts").get_caller_identity()["Account"]
    bucket = os.environ.get(
        "SAGEMAKER_SOURCE_BUCKET",
        f"sagemaker-{region}-{account}",
    )
    s3_key = f"lamer/source/sourcedir.tar.gz"
    s3_source_uri = f"s3://{bucket}/{s3_key}"

    print("Uploading source code (git archive)...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        subprocess.run(
            ["git", "archive", "--format=tar.gz", "-o", tmp.name, "HEAD"],
            cwd=str(repo_root),
            check=True,
        )
        tarball_size_mb = Path(tmp.name).stat().st_size / (1024 * 1024)
        print(f"  Tarball: {tarball_size_mb:.1f} MB")
        subprocess.run(
            ["aws", "s3", "cp", tmp.name, s3_source_uri, "--region", region],
            check=True,
        )
    print(f"  Uploaded to {s3_source_uri}")

    # Training environment variables
    env = {
        "RUN_NAME": args.run_name,
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND", "FLASH_ATTN"),
        "HF_HOME": "/tmp/hf_cache",
    }

    # Pass through secrets if set
    for key in ("WANDB_API_KEY", "HF_TOKEN"):
        val = os.environ.get(key)
        if val:
            env[key] = val

    # Pass through optional overrides
    for key in ("ADV_ESTIMATOR", "SKIP_CONNECTION_TEST"):
        val = os.environ.get(key)
        if val:
            env[key] = val

    # source_dir points to the tarball we just uploaded, so you don't need
    # to rebuild the Docker image for code-only changes.
    estimator = Estimator(
        image_uri=args.image_uri,
        role=args.role,
        source_dir=s3_source_uri,
        entry_point="scripts/sagemaker/entrypoint.sh",
        instance_type=args.instance_type,
        instance_count=1,
        volume_size=args.volume_size,
        max_run=args.max_run,
        max_wait=args.max_run + 3600 if args.use_spot else None,
        use_spot_instances=args.use_spot if args.use_spot else None,
        checkpoint_s3_uri=args.checkpoint_s3_uri,
        checkpoint_local_path="/opt/ml/checkpoints" if args.checkpoint_s3_uri else None,
        base_job_name="lamer",
        environment=env,
    )

    # Input data channels
    inputs = {}
    if args.vla_s3_uri:
        inputs["vla"] = args.vla_s3_uri

    print("Launching SageMaker training job:")
    print(f"  Image:     {args.image_uri}")
    print(f"  Source:    {s3_source_uri}")
    print(f"  Instance:  {args.instance_type}")
    print(f"  Volume:    {args.volume_size} GB")
    print(f"  Run name:  {args.run_name}")
    print(f"  Spot:      {args.use_spot}")
    if args.checkpoint_s3_uri:
        print(f"  Ckpt S3:   {args.checkpoint_s3_uri}")
    if args.vla_s3_uri:
        print(f"  VLA S3:    {args.vla_s3_uri}")
    print()

    estimator.fit(inputs=inputs or None, wait=True, logs="All")

    print("\nTraining job completed.")


if __name__ == "__main__":
    main()
