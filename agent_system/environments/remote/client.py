"""
RemoteEnvironmentManager — drop-in replacement for any
EnvironmentManagerBase subclass that forwards calls to an EnvServer
over TCP.

The TrajectoryCollector and RayPPOTrainer never know the difference.
"""

import logging
import socket
import time
from uuid import uuid4

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
        return self._call("reset")

    def step(self, text_actions, phase="play"):
        """Returns ``(next_obs_dict, rewards, dones, infos)``."""
        return self._call("step", text_actions, phase=phase)

    def restart(self):
        """Returns ``(observations_dict, infos_list)``."""
        return self._call("restart")

    def reflect(self):
        """Returns ``(observations_dict, infos_list)``."""
        return self._call("reflect")

    def success_evaluator(self, **kwargs):
        """Returns ``Dict[str, np.ndarray]``."""
        return self._call("success_evaluator", **kwargs)

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

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self):
        """Establish (or re-establish) the TCP connection."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect((self._host, self._port))
        self._socket = sock
        logger.info("Connected to EnvServer at %s:%s", self._host, self._port)

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
