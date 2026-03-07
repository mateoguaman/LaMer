"""
Utility to create an environment manager from config and start serving.

Can be used programmatically::

    from agent_system.environments.remote.launcher import launch_env_server
    launch_env_server("sokoban", "verl/trainer/config/eval/sokoban.yaml",
                      host="0.0.0.0", port=50051, is_train=True)

Or via the CLI wrapper ``scripts/launch_env_server.py``.
"""

import logging

from omegaconf import OmegaConf

from .server import EnvServer

logger = logging.getLogger(__name__)

# Maps lowercase env names to their make_envs import paths.
_ENV_FACTORIES = {
    "sokoban": "agent_system.environments.sokoban",
    "minesweeper": "agent_system.environments.minesweeper",
    "maze": "agent_system.environments.maze",
    "alfworld": "agent_system.environments.alfworld",
    "webshop": "agent_system.environments.webshop",
    "pybullet_vla": "agent_system.environments.pybullet_vla",
}


def launch_env_server(
    env_name: str,
    config_path: str,
    host: str = "0.0.0.0",
    port: int = 50051,
    is_train: bool = True,
    config_overrides: dict | None = None,
):
    """Create an environment manager from *config_path* and serve it.

    Parameters
    ----------
    env_name : str
        One of ``sokoban``, ``minesweeper``, ``maze``, ``alfworld``,
        ``webshop``.
    config_path : str
        Path to a YAML config (e.g.
        ``verl/trainer/config/eval/sokoban.yaml``).
    host / port : str / int
        Bind address for the TCP server.
    is_train : bool
        If *True* return the training env manager; otherwise the
        validation one.
    config_overrides : dict, optional
        Key-value overrides merged into the loaded config (e.g.
        ``{"data.train_batch_size": 4}``).
    """
    import importlib
    import ray

    # ---- Load config ----
    config = OmegaConf.load(config_path)
    if config_overrides:
        config = OmegaConf.merge(config, OmegaConf.create(config_overrides))

    # ---- Init Ray (needed for vectorised env workers) ----
    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    # ---- Import make_envs for the requested environment ----
    key = env_name.lower()
    if key not in _ENV_FACTORIES:
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Available: {list(_ENV_FACTORIES)}"
        )
    module = importlib.import_module(_ENV_FACTORIES[key])
    make_envs = module.make_envs

    # ---- Build envs ----
    train_envs, val_envs = make_envs(config)
    env_manager = train_envs if is_train else val_envs

    logger.info(
        "Serving %s env (%s, num_processes=%d) on %s:%d",
        "train" if is_train else "val",
        env_name,
        env_manager.num_processes,
        host,
        port,
    )

    # ---- Serve ----
    server = EnvServer(env_manager, host=host, port=port)
    server.serve()
