"""
VLA (Vision-Language-Action) inference wrapper.

Loads a VLA checkpoint and exposes batched ``predict()`` for use inside
``PyBulletVLAEnvironmentManager``.  The VLA is treated as a *frozen* inner
policy: the outer LLM never calls it directly — the env manager does.

This module is intentionally abstract.  Concrete model loading, tokenisation,
and forward-pass logic depend on the specific VLA architecture (e.g. RT-2,
Octo, OpenVLA).  Subclass ``VLAPolicy`` and override ``_load_model`` /
``_forward`` for your checkpoint format.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VLAPolicy(ABC):
    """Abstract base for batched VLA inference."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 128,
    ):
        """
        Parameters
        ----------
        checkpoint_path : str
            Path to the VLA model checkpoint.
        device : str
            Torch device string (``"cuda"``, ``"cuda:0"``, etc.).
        batch_size : int
            Maximum batch size for inference.
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self._load_model()
        logger.info(
            "VLAPolicy loaded from %s on %s (batch_size=%d)",
            checkpoint_path, device, batch_size,
        )

    @abstractmethod
    def _load_model(self) -> None:
        """Load the VLA model weights into ``self.model``.

        Must be overridden by concrete subclasses.
        """

    @abstractmethod
    def _forward(
        self,
        goals: List[str],
        observations: List[Dict[str, Any]],
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """Run a single forward pass through the VLA.

        Parameters
        ----------
        goals : list[str]
            Natural-language goal for each env in the batch.
        observations : list[dict]
            Per-env observation dicts from pybullet (images, proprioception).
        active_mask : np.ndarray
            Boolean mask — ``True`` for envs that still need actions.

        Returns
        -------
        np.ndarray
            Action array of shape ``(batch, action_dim)``.  Entries where
            ``active_mask`` is False may be garbage and will be ignored.
        """

    def predict(
        self,
        goals: List[str],
        observations: List[Dict[str, Any]],
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """Public entry point — delegates to ``_forward`` with logging.

        Parameters
        ----------
        goals : list[str]
            One goal string per env.
        observations : list[dict]
            One observation dict per env (from pybullet).
        active_mask : np.ndarray
            Boolean array of shape ``(batch,)``.

        Returns
        -------
        np.ndarray
            Actions of shape ``(batch, action_dim)``.
        """
        n_active = int(active_mask.sum())
        if n_active == 0:
            # Nothing to do — return zeros (shape will be filled by caller).
            action_dim = getattr(self, "action_dim", 7)
            return np.zeros((len(goals), action_dim), dtype=np.float32)

        actions = self._forward(goals, observations, active_mask)
        return actions


class DummyVLAPolicy(VLAPolicy):
    """Placeholder VLA that returns random actions.

    Useful for integration testing before a real checkpoint is available.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        batch_size: int = 128,
        action_dim: int = 7,
    ):
        self.action_dim = action_dim
        super().__init__(checkpoint_path, device, batch_size)

    def _load_model(self) -> None:
        logger.info("DummyVLAPolicy: no model to load")
        self.model = None

    def _forward(
        self,
        goals: List[str],
        observations: List[Dict[str, Any]],
        active_mask: np.ndarray,
    ) -> np.ndarray:
        batch = len(goals)
        return np.random.uniform(-1, 1, size=(batch, self.action_dim)).astype(
            np.float32
        )
