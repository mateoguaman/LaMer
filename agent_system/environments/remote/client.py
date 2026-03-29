"""
RemoteEnvironmentManager — drop-in replacement for any
EnvironmentManagerBase subclass that forwards calls to an EnvServer
over TCP.

The TrajectoryCollector and RayPPOTrainer never know the difference.
"""

import logging
import socket
import time
from collections import defaultdict
from uuid import uuid4

import numpy as np

from .protocol import EnvRequest, EnvResponse, send_message, recv_message

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_INITIAL_BACKOFF_S = 2


class RemoteEnvironmentManager:
    """Network client implementing the EnvironmentManagerBase interface."""

    def __init__(self, server_address: str, timeout: float = 300.0):
        """
        Parameters
        ----------
        server_address : str
            ``"host:port"`` of the running EnvServer.
        timeout : float
            Socket timeout in seconds (default 5 min to accommodate slow
            environments or large batches).
        """
        host, port_str = server_address.rsplit(":", 1)
        self._host = host
        self._port = int(port_str)
        self._timeout = timeout
        self._socket: socket.socket | None = None
        self._last_benchmark_stats: dict = {}

        self._connect()

        # Cache read-only properties that the rollout loop accesses.
        props = self._call("get_properties")
        self._num_attempts: int = props["num_attempts"]
        self._num_processes: int = props["num_processes"]
        self._max_turns: int = props["max_turns"]
        self._do_reflection: bool = props["do_reflection"]

    # ------------------------------------------------------------------
    # Properties read by TrajectoryCollector
    # ------------------------------------------------------------------

    @property
    def num_attempts(self) -> int:
        return self._num_attempts

    @property
    def num_processes(self) -> int:
        return self._num_processes

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def do_reflection(self) -> bool:
        return self._do_reflection

    # ------------------------------------------------------------------
    # Environment interface (mirrors EnvironmentManagerBase)
    # ------------------------------------------------------------------

    def reset(self):
        """Returns ``(observations_dict, infos_list)``."""
        call_t0 = time.perf_counter()
        observations, infos = self._call("reset")
        elapsed_s = time.perf_counter() - call_t0
        server_payloads = self._extract_benchmark_payloads(infos)
        self._last_benchmark_stats = self._build_benchmark_stats(
            method="reset",
            client_round_trip_s=elapsed_s,
            server_payloads=server_payloads,
        )
        return observations, infos

    def step(self, text_actions, phase="play"):
        """Returns ``(next_obs_dict, rewards, dones, infos)``."""
        call_t0 = time.perf_counter()
        observations, rewards, dones, infos = self._call(
            "step", text_actions, phase=phase,
        )
        elapsed_s = time.perf_counter() - call_t0
        server_payloads = self._extract_benchmark_payloads(infos)
        self._last_benchmark_stats = self._build_benchmark_stats(
            method=f"step:{phase}",
            client_round_trip_s=elapsed_s,
            server_payloads=server_payloads,
        )
        return observations, rewards, dones, infos

    def restart(self):
        """Returns ``(observations_dict, infos_list)``."""
        call_t0 = time.perf_counter()
        observations, infos = self._call("restart")
        elapsed_s = time.perf_counter() - call_t0
        server_payloads = self._extract_benchmark_payloads(infos)
        self._last_benchmark_stats = self._build_benchmark_stats(
            method="restart",
            client_round_trip_s=elapsed_s,
            server_payloads=server_payloads,
        )
        return observations, infos

    def reflect(self):
        """Returns ``(observations_dict, infos_list)``."""
        call_t0 = time.perf_counter()
        observations, infos = self._call("reflect")
        elapsed_s = time.perf_counter() - call_t0
        server_payloads = self._extract_benchmark_payloads(infos)
        self._last_benchmark_stats = self._build_benchmark_stats(
            method="reflect",
            client_round_trip_s=elapsed_s,
            server_payloads=server_payloads,
        )
        return observations, infos

    def success_evaluator(self, **kwargs):
        """Evaluate episode success locally (no TCP round-trip needed).

        This is pure data analysis over trajectory info dicts — sending the
        full trajectory data over the wire would exceed the 4 GB protocol
        limit for large batch sizes.
        """
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)

        success = defaultdict(list)
        for bs in range(batch_size):
            wons = [False for _ in range(self._num_attempts)]
            for i in reversed(range(len(total_batch_list[bs]))):
                batch_item = total_batch_list[bs][i]
                if batch_item["active_masks"]:
                    info = total_infos[bs][i]
                    traj_idx = batch_item["traj_idx"]
                    if batch_item["phase"] == "play":
                        wons[traj_idx] = wons[traj_idx] or info.get("won", False)

            _won = False
            for traj_idx, won in enumerate(wons):
                _won = _won or won
                success[f"success_rate[{traj_idx}]"].append(_won)

        return {key: np.array(value) for key, value in success.items()}

    def close(self):
        """Tell the server we are done, then close the socket."""
        if self._socket is None:
            return
        try:
            self._call("close")
        except Exception:
            pass
        try:
            self._socket.close()
        except Exception:
            pass
        self._socket = None

    def get_last_benchmark_stats(self):
        """Return benchmark timing metadata for the most recent remote call."""
        return dict(self._last_benchmark_stats)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self):
        """Establish (or re-establish) the TCP connection.

        Retries with exponential backoff if the server is not yet listening
        (e.g. PyBullet workers still initializing on SLURM).
        """
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass

        backoff = _INITIAL_BACKOFF_S
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._timeout)
                sock.connect((self._host, self._port))
                self._socket = sock
                logger.info("Connected to EnvServer at %s:%s", self._host, self._port)
                return
            except (ConnectionRefusedError, OSError) as exc:
                last_exc = exc
                logger.warning(
                    "Connect attempt %d/%d to %s:%s failed: %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, self._host, self._port, exc, backoff,
                )
                time.sleep(backoff)
                backoff *= 2

        raise ConnectionRefusedError(
            f"Could not connect to {self._host}:{self._port} "
            f"after {_MAX_RETRIES} attempts"
        ) from last_exc

    def _call(self, method: str, *args, **kwargs):
        """Send a request, receive the response, retry on connection errors."""
        request = EnvRequest(
            request_id=uuid4().hex,
            method=method,
            args=args,
            kwargs=kwargs,
        )

        last_exc = None
        backoff = _INITIAL_BACKOFF_S

        for attempt in range(_MAX_RETRIES):
            try:
                send_message(self._socket, request)
                response: EnvResponse = recv_message(self._socket)

                if response.status == "error":
                    raise RuntimeError(
                        f"Remote env error in '{method}':\n"
                        f"{response.error_message}"
                    )
                return response.result

            except (ConnectionError, OSError, BrokenPipeError) as exc:
                last_exc = exc
                logger.warning(
                    "Connection error on attempt %d/%d for '%s': %s — "
                    "retrying in %ds",
                    attempt + 1, _MAX_RETRIES, method, exc, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                try:
                    self._connect()
                except Exception as conn_exc:
                    logger.warning("Reconnect failed: %s", conn_exc)

        raise ConnectionError(
            f"Failed to call '{method}' after {_MAX_RETRIES} retries"
        ) from last_exc

    def _extract_benchmark_payloads(self, infos):
        """Remove benchmark payloads from info dicts and return them."""
        payloads = []
        if not isinstance(infos, list):
            return payloads
        for info in infos:
            if isinstance(info, dict) and "benchmark_env" in info:
                payloads.append(info.pop("benchmark_env"))
        return payloads

    def _build_benchmark_stats(self, method: str, client_round_trip_s: float, server_payloads):
        """Build a compact timing record for the last remote call."""
        stats = {
            "kind": "remote_call",
            "method": method,
            "server_address": f"{self._host}:{self._port}",
            "client_round_trip_s": float(client_round_trip_s),
            "server_payloads": server_payloads,
        }
        server_elapsed_candidates = [
            payload.get("server_elapsed_s")
            for payload in server_payloads
            if isinstance(payload, dict) and payload.get("server_elapsed_s") is not None
        ]
        if server_elapsed_candidates:
            server_elapsed_s = max(float(v) for v in server_elapsed_candidates)
            stats["server_elapsed_s"] = server_elapsed_s
            stats["transport_overhead_s"] = float(client_round_trip_s - server_elapsed_s)
        return stats

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
            f"RemoteEnvironmentManager("
            f"{self._host}:{self._port}, "
            f"num_processes={self._num_processes})"
        )
