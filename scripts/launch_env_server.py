#!/usr/bin/env python3
"""
CLI entry point for starting an environment server.

Examples
--------
# Serve a training Sokoban environment on port 50051
python scripts/launch_env_server.py \\
    --env_name sokoban \\
    --config verl/trainer/config/eval/sokoban.yaml \\
    --port 50051 --mode train

# Serve a validation environment on port 50052 with batch size override
python scripts/launch_env_server.py \\
    --env_name sokoban \\
    --config verl/trainer/config/eval/sokoban.yaml \\
    --port 50052 --mode val \\
    --batch_size 4
"""

import argparse
import logging
import sys
import os

# Ensure project root is on sys.path so imports work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_system.environments.remote.launcher import launch_env_server


def main():
    parser = argparse.ArgumentParser(
        description="Start a remote environment server."
    )
    parser.add_argument(
        "--env_name", required=True,
        help="Environment name (sokoban, minesweeper, maze, alfworld, webshop)",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=50051, help="Bind port")
    parser.add_argument(
        "--mode", choices=["train", "val"], default="train",
        help="Serve the training or validation environment",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override the batch size from config",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    overrides = {}
    if args.batch_size is not None:
        if args.mode == "train":
            overrides["data"] = {"train_batch_size": args.batch_size}
        else:
            overrides["data"] = {"val_batch_size": args.batch_size}

    launch_env_server(
        env_name=args.env_name,
        config_path=args.config,
        host=args.host,
        port=args.port,
        is_train=(args.mode == "train"),
        config_overrides=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()
