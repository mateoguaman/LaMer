"""
Thin LaMer-side wrapper for Language Table.

The actual environment runs in a separate process (language-table venv)
via the remote env server protocol. This module just returns
RemoteEnvironmentManager instances pointing at the server.
"""

from agent_system.environments.remote import RemoteEnvironmentManager


def make_envs(config):
    """Return (train_env_manager, val_env_manager) for Language Table.

    Expects config.env to have:
        - remote_address: "host:port" for training server
        - remote_val_address: "host:port" for validation server
    """
    envs = RemoteEnvironmentManager(config.env.remote_address)
    val_envs = RemoteEnvironmentManager(config.env.remote_val_address)
    return envs, val_envs
