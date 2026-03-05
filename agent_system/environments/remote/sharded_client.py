"""
ShardedRemoteEnvironmentManager — fans out to N EnvServers and merges
results, presenting the same interface as a single RemoteEnvironmentManager.

Each shard is an independent RemoteEnvironmentManager connection.  Per-env
arguments (text_actions, kwargs with per-env lists) are sliced by shard
boundaries before dispatch; results are concatenated back.

Threading is used for parallel dispatch (I/O-bound socket waits, no async
propagation needed in the synchronous training loop).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

from .client import RemoteEnvironmentManager

logger = logging.getLogger(__name__)


class ShardedRemoteEnvironmentManager:
    """Drop-in replacement that distributes work across N ``EnvServer`` s."""

    def __init__(
        self,
        server_addresses: List[str],
        timeout: float = 300.0,
    ):
        """
        Parameters
        ----------
        server_addresses : list[str]
            ``["host1:port", "host2:port", ...]`` — one per shard.
        timeout : float
            Per-shard socket timeout (passed to each
            ``RemoteEnvironmentManager``).
        """
        if not server_addresses:
            raise ValueError("server_addresses must be non-empty")

        self._shards: List[RemoteEnvironmentManager] = [
            RemoteEnvironmentManager(addr, timeout=timeout)
            for addr in server_addresses
        ]
        self._num_shards = len(self._shards)

        # Pre-compute slice boundaries from per-shard num_processes.
        shard_sizes = [s.num_processes for s in self._shards]
        self._shard_sizes = shard_sizes
        self._total_num_processes = sum(shard_sizes)

        # Cumulative boundaries: _boundaries[i] is the start index for shard i.
        self._boundaries: List[int] = []
        cumsum = 0
        for sz in shard_sizes:
            self._boundaries.append(cumsum)
            cumsum += sz

        # Validate that scalar properties agree across all shards.
        self._num_attempts = self._shards[0].num_attempts
        self._max_turns = self._shards[0].max_turns
        self._do_reflection = self._shards[0].do_reflection

        for i, shard in enumerate(self._shards[1:], 1):
            if shard.num_attempts != self._num_attempts:
                raise ValueError(
                    f"Shard {i} num_attempts={shard.num_attempts} != "
                    f"shard 0 num_attempts={self._num_attempts}"
                )
            if shard.max_turns != self._max_turns:
                raise ValueError(
                    f"Shard {i} max_turns={shard.max_turns} != "
                    f"shard 0 max_turns={self._max_turns}"
                )
            if shard.do_reflection != self._do_reflection:
                raise ValueError(
                    f"Shard {i} do_reflection={shard.do_reflection} != "
                    f"shard 0 do_reflection={self._do_reflection}"
                )

        self._executor = ThreadPoolExecutor(max_workers=self._num_shards)
        logger.info(
            "ShardedRemoteEnvironmentManager: %d shards, "
            "shard_sizes=%s, total_num_processes=%d",
            self._num_shards, shard_sizes, self._total_num_processes,
        )

    # ------------------------------------------------------------------
    # Properties (same interface as RemoteEnvironmentManager)
    # ------------------------------------------------------------------

    @property
    def num_attempts(self) -> int:
        return self._num_attempts

    @property
    def num_processes(self) -> int:
        return self._total_num_processes

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def do_reflection(self) -> bool:
        return self._do_reflection

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self):
        """Returns ``(observations_dict, infos_list)``."""
        results = self._parallel_call("reset")
        return self._merge_obs_infos(results)

    def step(self, text_actions: List[str], phase: str = "play"):
        """Returns ``(next_obs_dict, rewards, dones, infos)``."""
        sliced_actions = self._slice_list(text_actions)
        results = self._parallel_call_with_args(
            "step", sliced_actions, phase=phase,
        )
        return self._merge_step_results(results)

    def restart(self):
        """Returns ``(observations_dict, infos_list)``."""
        results = self._parallel_call("restart")
        return self._merge_obs_infos(results)

    def reflect(self):
        """Returns ``(observations_dict, infos_list)``."""
        results = self._parallel_call("reflect")
        return self._merge_obs_infos(results)

    def success_evaluator(self, **kwargs):
        """Returns ``Dict[str, np.ndarray]``."""
        # Slice the per-env kwargs: total_infos and total_batch_list.
        total_infos = kwargs.get("total_infos")
        total_batch_list = kwargs.get("total_batch_list")

        per_shard_kwargs = []
        for shard_idx in range(self._num_shards):
            start = self._boundaries[shard_idx]
            end = start + self._shard_sizes[shard_idx]
            shard_kw = dict(kwargs)  # shallow copy
            if total_infos is not None:
                shard_kw["total_infos"] = total_infos[start:end]
            if total_batch_list is not None:
                shard_kw["total_batch_list"] = total_batch_list[start:end]
            per_shard_kwargs.append(shard_kw)

        futures = {}
        for shard_idx, shard in enumerate(self._shards):
            fut = self._executor.submit(
                shard.success_evaluator, **per_shard_kwargs[shard_idx]
            )
            futures[fut] = shard_idx

        ordered_results = [None] * self._num_shards
        for fut in as_completed(futures):
            ordered_results[futures[fut]] = fut.result()  # raises on error

        # Merge: concatenate arrays by key.
        merged: Dict[str, np.ndarray] = {}
        all_keys = ordered_results[0].keys()
        for key in all_keys:
            merged[key] = np.concatenate(
                [r[key] for r in ordered_results], axis=0
            )
        return merged

    def close(self):
        """Close all shard connections."""
        for shard in self._shards:
            try:
                shard.close()
            except Exception:
                pass
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _slice_list(self, items: List) -> List[List]:
        """Slice a flat list into per-shard sublists."""
        sliced = []
        for shard_idx in range(self._num_shards):
            start = self._boundaries[shard_idx]
            end = start + self._shard_sizes[shard_idx]
            sliced.append(items[start:end])
        return sliced

    def _parallel_call(self, method: str):
        """Call *method* (no per-env args) on all shards in parallel.

        Returns a list of results in shard order.
        """
        futures = {}
        for shard_idx, shard in enumerate(self._shards):
            fut = self._executor.submit(getattr(shard, method))
            futures[fut] = shard_idx

        ordered = [None] * self._num_shards
        for fut in as_completed(futures):
            ordered[futures[fut]] = fut.result()
        return ordered

    def _parallel_call_with_args(
        self,
        method: str,
        per_shard_args: List[List],
        **kwargs,
    ):
        """Call *method* with per-shard positional args + shared kwargs.

        Returns a list of results in shard order.
        """
        futures = {}
        for shard_idx, shard in enumerate(self._shards):
            fut = self._executor.submit(
                getattr(shard, method), per_shard_args[shard_idx], **kwargs,
            )
            futures[fut] = shard_idx

        ordered = [None] * self._num_shards
        for fut in as_completed(futures):
            ordered[futures[fut]] = fut.result()
        return ordered

    # ------------------------------------------------------------------
    # Merging helpers
    # ------------------------------------------------------------------

    def _merge_obs_infos(self, results):
        """Merge ``(obs_dict, infos_list)`` tuples from all shards."""
        all_obs = [r[0] for r in results]
        all_infos = [r[1] for r in results]

        merged_obs = self._merge_obs_dicts(all_obs)
        merged_infos = self._concat_lists(all_infos)
        return merged_obs, merged_infos

    def _merge_step_results(self, results):
        """Merge ``(obs_dict, rewards, dones, infos)`` from all shards."""
        all_obs = [r[0] for r in results]
        all_rewards = [r[1] for r in results]
        all_dones = [r[2] for r in results]
        all_infos = [r[3] for r in results]

        merged_obs = self._merge_obs_dicts(all_obs)
        merged_rewards = np.concatenate(all_rewards, axis=0)
        merged_dones = np.concatenate(all_dones, axis=0)
        merged_infos = self._concat_lists(all_infos)
        return merged_obs, merged_rewards, merged_dones, merged_infos

    def _merge_obs_dicts(self, obs_list: List[Dict]) -> Dict:
        """Merge observation dicts by concatenating per-key."""
        merged = {}
        for key in obs_list[0]:
            values = [obs[key] for obs in obs_list]
            if all(v is None for v in values):
                merged[key] = None
            elif isinstance(values[0], list):
                # Text observations — concatenate lists.
                merged[key] = []
                for v in values:
                    merged[key].extend(v)
            elif isinstance(values[0], np.ndarray):
                merged[key] = np.concatenate(values, axis=0)
            elif isinstance(values[0], str):
                # Single-string obs (e.g. empty reflect obs) — keep as-is
                # if all shards return identical strings.
                merged[key] = values[0]
            else:
                # Fallback: return list of per-shard values.
                merged[key] = values
        return merged

    @staticmethod
    def _concat_lists(lists_of_lists):
        """Flatten a list of lists."""
        merged = []
        for sub in lists_of_lists:
            merged.extend(sub)
        return merged

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"ShardedRemoteEnvironmentManager("
            f"{self._num_shards} shards, "
            f"num_processes={self._total_num_processes})"
        )
