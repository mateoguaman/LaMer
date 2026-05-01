import atexit, pickle, socket, struct, subprocess, time
import os
import numpy as np

REPO = os.path.expanduser("~/projects/language-table")
CONDA_ENV = "/home/sidhraja/miniconda3/envs/lerobotenv"
use_conda = True
if use_conda:
    LEROBOT_PYTHON = f"{CONDA_ENV}/bin/python"
else:
    LEROBOT_PYTHON = f"{REPO}/lerobotenv/bin/python"
SERVER_SCRIPT  = f"{REPO}/training/eval/lerobot_policy_server.py"

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

def _send(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(_HEADER_FMT, len(payload)) + payload)

def _recv(sock):
    head = b""
    while len(head) < _HEADER_SIZE:
        chunk = sock.recv(_HEADER_SIZE - len(head))
        if not chunk:
            raise ConnectionError("socket closed mid-header")
        head += chunk
    (n,) = struct.unpack(_HEADER_FMT, head)
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-payload")
        buf += chunk
    return pickle.loads(buf)


class SmolVLAPolicy:
    def __init__(self, checkpoint_path,
                 host="127.0.0.1", port=50100,
                 server_log="/tmp/smolvla_interactive.log",
                 ready_timeout=300.0):
        self.host, self.port = host, port
        self.proc, self.sock = None, None
        self._spawn_server(checkpoint_path, server_log, ready_timeout)
        self._connect()
        atexit.register(self.close)

    def _spawn_server(self, checkpoint, log_path, timeout):
        log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [LEROBOT_PYTHON, "-u", SERVER_SCRIPT,
             "--checkpoint_path", checkpoint,
             "--host", self.host, "--port", str(self.port)],
            stdout=log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            with open(log_path) as f:
                if "Policy server listening" in f.read():
                    print(f"SmolVLA server up (pid={self.proc.pid}, log={log_path})")
                    return
            if self.proc.poll() is not None:
                with open(log_path) as f:
                    raise RuntimeError(f"server died:\n{f.read()[-2000:]}")
            time.sleep(1.0)
        self.proc.terminate()
        raise RuntimeError(f"server not ready within {timeout}s; see {log_path}")

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def reset(self, num_envs=1):
        _send(self.sock, {"method": "reset"})
        resp = _recv(self.sock)
        if resp.get("status") != "ok":
            raise RuntimeError(f"reset failed: {resp.get('error_message')}")

    def predict(self, goals, obs_list, active_mask):
        actions = []
        for goal, obs, active in zip(goals, obs_list, active_mask):
            if not active:
                actions.append(np.zeros(2, dtype=np.float32))
                continue
            rgb = obs["rgb"]
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            state = np.asarray(
                obs.get("effector_translation", np.zeros(2, dtype=np.float32)),
                dtype=np.float32,
            )
            _send(self.sock, {
                "method": "action",
                "rgb": rgb.tolist(),
                "state": state.tolist(),
                "instruction": goal,
            })
            resp = _recv(self.sock)
            # print(f"Action: {resp}")
            if resp.get("status") != "ok":
                raise RuntimeError(f"action failed: {resp.get('error_message')}")
            actions.append(np.asarray(resp["action"], dtype=np.float32))
        return actions

    def close(self):
        if self.sock is not None:
            try:
                _send(self.sock, {"method": "close"})
                _recv(self.sock)
            except Exception:
                pass
            try: self.sock.close()
            except Exception: pass
            self.sock = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired: self.proc.kill()
            self.proc = None

    def __del__(self):
        self.close()