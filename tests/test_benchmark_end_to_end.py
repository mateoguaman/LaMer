from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from agent_system.environments.language_table.env_manager import make_envs
from benchmarks.benchmark_end_to_end import _build_trainer_cmd
from benchmarks.end_to_end_sweep import build_sweep_manifest
from verl.trainer.ppo.ray_trainer import _summarize_benchmark_records


class _FakeRemote:
    def __init__(self, address: str):
        self.address = address
        self.num_processes = 32
        self.num_attempts = 3
        self.max_turns = 4
        self.do_reflection = True

    def close(self):
        return None


def _make_args(**overrides):
    defaults = {
        "run_name": "bench",
        "adv_estimator": "gigpo",
        "train_data": "/tmp/train.parquet",
        "val_data": "/tmp/val.parquet",
        "train_num_envs": 16,
        "val_num_envs": 128,
        "model_path": "Qwen/Qwen3-4B",
        "learning_rate": 1e-6,
        "batch_size": 64,
        "micro_batch_size": 16,
        "use_kl_loss": False,
        "kl_loss_coef": 0.001,
        "kl_loss_type": "low_var_kl",
        "use_kl_in_reward": False,
        "kl_reward_coef": 0.001,
        "group_size": 8,
        "num_attempts": 3,
        "max_turns": 4,
        "warmup_iterations": 2,
        "measured_iterations": 3,
        "preset": "resolved_training_config",
        "skip_val_server": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_trainer_cmd_skip_val_server_omits_val_remote_keys():
    args = _make_args(skip_val_server=True)
    cmd = _build_trainer_cmd(
        args=args,
        output_dir=Path("/tmp/benchmark"),
        train_addresses=["127.0.0.1:50051", "127.0.0.1:50052"],
        val_addresses=[],
    )

    assert "+env.skip_val_env=True" in cmd
    assert "+env.sharded=True" in cmd
    assert any(part.startswith("+env.remote_addresses=") for part in cmd)
    assert not any("remote_val_address" in part for part in cmd)


def test_language_table_make_envs_can_skip_val_env(monkeypatch):
    monkeypatch.setattr(
        "agent_system.environments.language_table.env_manager.RemoteEnvironmentManager",
        _FakeRemote,
    )

    cfg = OmegaConf.create(
        {
            "env": {
                "remote_address": "train0",
                "skip_val_env": True,
                "reflection_type": "history_and_reflection",
            }
        }
    )

    envs, val_envs = make_envs(cfg)

    assert envs.num_processes == 32
    assert val_envs is None
    envs.close()


def test_summarize_benchmark_records_keeps_env_side_fields():
    records = [
        {
            "is_warmup": True,
            "timing_raw": {"gen": 1.0, "step": 2.0},
            "rollout_trace": {"rollout_elapsed_s": 5.0, "turns": []},
        },
        {
            "is_warmup": False,
            "timing_raw": {"gen": 12.0, "step": 14.0},
            "rollout_trace": {
                "rollout_elapsed_s": 10.0,
                "turns": [
                    {
                        "remote": {
                            "server_elapsed_s": 1.0,
                            "transport_overhead_s": 0.1,
                            "server_payloads": [
                                {
                                    "vla_total_s": 2.0,
                                    "env_step_total_s": 3.0,
                                    "ray_dispatch_s": 0.5,
                                    "ray_collect_s": 0.2,
                                    "ray_unpack_s": 0.1,
                                }
                            ],
                        }
                    },
                    {
                        "remote": {
                            "server_elapsed_s": 2.5,
                            "transport_overhead_s": 0.4,
                            "shards": [
                                {
                                    "server_elapsed_s": 2.0,
                                    "server_payloads": [
                                        {
                                            "vla_total_s": 1.0,
                                            "env_step_total_s": 1.5,
                                            "ray_dispatch_s": 0.2,
                                            "ray_collect_s": 0.1,
                                            "ray_unpack_s": 0.05,
                                        }
                                    ],
                                },
                                {
                                    "server_elapsed_s": 2.5,
                                    "server_payloads": [
                                        {
                                            "vla_total_s": 1.2,
                                            "env_step_total_s": 1.7,
                                            "ray_dispatch_s": 0.3,
                                            "ray_collect_s": 0.15,
                                            "ray_unpack_s": 0.06,
                                        }
                                    ],
                                },
                            ],
                        }
                    },
                ],
            },
        },
        {
            "is_warmup": False,
            "timing_raw": {"gen": 18.0, "step": 20.0},
            "rollout_trace": {
                "rollout_elapsed_s": 14.0,
                "turns": [
                    {
                        "remote": {
                            "server_elapsed_s": 1.5,
                            "transport_overhead_s": 0.2,
                            "server_payloads": [
                                {
                                    "vla_total_s": 2.5,
                                    "env_step_total_s": 3.5,
                                    "ray_dispatch_s": 0.4,
                                    "ray_collect_s": 0.25,
                                    "ray_unpack_s": 0.11,
                                }
                            ],
                        }
                    }
                ],
            },
        },
    ]

    summary = _summarize_benchmark_records(records)

    assert summary["iterations_measured"] == 2
    assert summary["timing_s_p50"]["gen"] == 15.0
    assert summary["rollout_elapsed_s_p50"] == 12.0
    assert summary["measured_turns"] == 3
    assert summary["server_payloads_measured"] == 4
    assert round(summary["env_server_elapsed_s_mean"], 6) == round((1.0 + 2.5 + 1.5) / 3, 6)
    assert round(summary["transport_overhead_s_mean"], 6) == round((0.1 + 0.4 + 0.2) / 3, 6)
    assert round(summary["vla_total_s_sum"], 6) == 6.7
    assert round(summary["ray_dispatch_s_sum"], 6) == 1.4
    assert summary["per_shard_server_elapsed_s"]["shard_0"]["mean"] == 2.0
    assert summary["per_shard_server_elapsed_s"]["shard_1"]["mean"] == 2.5


def test_build_sweep_manifest_picks_best_rollout(tmp_path):
    sweep_root = tmp_path / "benchmark_end_to_end"
    run1 = sweep_root / "shards_1"
    run2 = sweep_root / "shards_4"
    for run_dir, shard_count, rollout_mean, transport_mean in (
        (run1, 1, 10.0, 0.35),
        (run2, 4, 6.0, 0.22),
    ):
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (run_dir / "launcher_metadata.json").write_text(
            (
                "{\n"
                f'  "train_shard_count": {shard_count},\n'
                '  "train_envs_per_shard": 4,\n'
                '  "env_server_gpus": ["4"],\n'
                '  "skip_val_server": true,\n'
                '  "val_server_enabled": false,\n'
                '  "preprocess_mode": "jax_gpu"\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        (artifacts_dir / "benchmark_summary.json").write_text(
            (
                "{\n"
                f'  "rollout_elapsed_s_mean": {rollout_mean},\n'
                f'  "transport_overhead_s_mean": {transport_mean},\n'
                '  "ray_dispatch_s_sum": 0.2,\n'
                '  "ray_collect_s_sum": 1.0,\n'
                '  "ray_unpack_s_sum": 0.5\n'
                "}\n"
            ),
            encoding="utf-8",
        )

    manifest = build_sweep_manifest(sweep_root)

    assert manifest["run_count"] == 2
    assert manifest["best_rollout_run"]["train_shard_count"] == 4
    assert manifest["post_fused_backlog_triage"]["priority"] == "transport_reduction"
