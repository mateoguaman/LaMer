"""Lightweight recorder for interactive LAVA attempts.

One Recorder per high-level task. Wraps run_command so logging is automatic;
auto-saves to disk after every call. Manual success/failure labeling. Frames
returned by `rec.run` are annotated with both the high-level task and the
current low-level command, so `show_rollout(frames)` immediately displays
the full context.

Usage in the notebook:

    from recorder import Recorder
    env.reset()
    rec = Recorder(task="draw a 6", env=env, run_fn=run_command)
    rec.print_state()          # current EE/blocks, with high-level task

    frames, reward, info = rec.run("move behind the red moon")
    show_rollout(frames)       # animation labeled "draw a 6" + the command
    frames, reward, info = rec.run("push the red moon to the top-left")

    rec.label(success=False, notes="moon overshot")
    rec.reset()                # env.reset() + new attempt dir

    rec.summary()              # aggregate success rate for this task

On-disk layout (frames are stored RAW, no overlay, so they remain useful
for downstream ML; overlays are applied only when `rec.run` returns):

    ~/recordings/lamer/draw_a_6/
      attempt_0001/
        meta.json    # task, init_state, commands[], cmd_boundaries, label
        frames.npy   # stacked RGB frames from all commands in this attempt
      attempt_0002/
        ...
"""

import contextlib
import io
import json
import textwrap
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from language_table.lamer.state_to_text import state_to_text


def _load_mono_font(size: int) -> ImageFont.ImageFont:
    """Try to load a monospace TTF; fall back to PIL's bitmap font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _annotate_frame(
    frame: np.ndarray,
    *,
    task: str,
    instruction: str,
    trial_idx: int,
    turn_idx: int,
    subtitle: Optional[str] = None,
    target_width: int = 640,
    font_size: int = 16,
    padding: int = 6,
) -> np.ndarray:
    """Draw fully opaque, full-width banners on top of *frame*.

    Top banner: 'Trial N | Step K' + the low-level instruction (wrapped).
    Bottom banner: the high-level task, optionally followed by *subtitle*
    (used for things like "Attempt 3 -- success").
    The frame is upscaled to *target_width* first so the text is legible.
    """
    h, w = frame.shape[:2]
    if w != target_width:
        new_h = int(round(h * target_width / w))
        img = Image.fromarray(frame).resize(
            (target_width, new_h), Image.BILINEAR
        )
    else:
        img = Image.fromarray(frame)

    width, height = img.size
    font = _load_mono_font(font_size)

    # Estimate how many characters fit on a line.
    if hasattr(font, "getlength"):
        char_w = max(1.0, font.getlength("M"))
    else:
        char_w = font_size * 0.6
    max_chars = max(10, int((width - 2 * padding) / char_w))

    header = f"Trial {trial_idx} | Step {turn_idx}"
    instr_lines = textwrap.wrap(instruction, width=max_chars) or [""]
    top_lines = [header] + instr_lines
    bot_lines = textwrap.wrap(task, width=max_chars) or [""]
    if subtitle:
        bot_lines += textwrap.wrap(subtitle, width=max_chars)

    # Measure line height with the font's actual ascent/descent so wrapped
    # lines don't overlap each other.
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
    else:
        line_h = font_size + 2

    top_h = line_h * len(top_lines) + 2 * padding
    bot_h = line_h * len(bot_lines) + 2 * padding

    # Build a new canvas: top banner | original image | bottom banner.
    out = Image.new("RGB", (width, height + top_h + bot_h), (0, 0, 0))
    out.paste(img, (0, top_h))
    draw = ImageDraw.Draw(out)

    # Top banner text.
    y = padding
    for line in top_lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    # Bottom banner text.
    y = height + top_h + padding
    for line in bot_lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    return np.array(out)


def _annotate_frames(
    frames: List[np.ndarray],
    *,
    task: str,
    instruction: str,
    trial_idx: int,
    turn_idx: int,
) -> List[np.ndarray]:
    return [
        _annotate_frame(
            f,
            task=task,
            instruction=instruction,
            trial_idx=trial_idx,
            turn_idx=turn_idx,
        )
        for f in frames
    ]


def _make_title_card(
    width: int,
    height: int,
    text: str,
    font_size: int = 28,
) -> np.ndarray:
    """Render centered white text on a black canvas."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_mono_font(font_size)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = max(0, (width - tw) // 2 - bbox[0])
    y = max(0, (height - th) // 2 - bbox[1])
    draw.multiline_text(
        (x, y), text, fill=(255, 255, 255), font=font, align="center"
    )
    return np.array(img)


def _pad_to(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pad an HxWx3 uint8 frame with black to (height, width). No-op if larger."""
    h, w = frame.shape[:2]
    if h == height and w == width:
        return frame
    pad_h = max(0, height - h)
    pad_w = max(0, width - w)
    return np.pad(
        frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
    )


def _safe_task_name(task: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in task)


def render_task_video(
    task: str,
    out_path: Optional[str] = None,
    *,
    fps: int = 10,
    only: Optional[str] = None,
    include_unlabeled: bool = True,
    title_card_seconds: float = 1.5,
    root: str = "~/recordings/lamer",
    crf: int = 23,
    preset: str = "fast",
) -> Path:
    """Concatenate every attempt for *task* into one annotated H.264 mp4.

    Walks ``<root>/<safe_task_name>/attempt_*/``, loads each attempt's raw
    frames, draws the same task/instruction banners as ``rec.run`` (plus a
    bottom-banner subtitle showing ``Attempt N -- <label>``), and streams
    the result to ``out_path`` via ``ffmpeg`` (libx264, yuv420p) so it
    plays in any modern browser / video player. A short title card is
    inserted before each attempt so transitions are obvious.

    Args:
        task: high-level task string used when the recordings were made.
        out_path: where to write the mp4. Defaults to
            ``<root>/<safe_task_name>.mp4`` (next to the recordings).
        fps: playback framerate.
        only: ``"success"``, ``"failure"``, or ``None`` to keep all labels.
        include_unlabeled: if False, attempts with ``label is None`` are skipped.
        title_card_seconds: how long each per-attempt title card lingers.
        root: directory containing per-task subdirs.
        crf: x264 quality (lower = better, 18-28 is the typical range).
        preset: x264 speed preset (ultrafast/fast/medium/slow/...).

    Returns the resolved output path.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "render_task_video requires `ffmpeg` on PATH. Install it via "
            "`sudo apt install ffmpeg` (Debian/Ubuntu) and retry."
        )

    safe = _safe_task_name(task)
    root_path = Path(root).expanduser()
    task_dir = root_path / safe
    if not task_dir.exists():
        raise FileNotFoundError(f"no task directory at {task_dir}")

    attempt_dirs = sorted(task_dir.glob("attempt_*"))
    if not attempt_dirs:
        raise RuntimeError(f"no attempts found under {task_dir}")

    if out_path is None:
        out_path = root_path / f"{safe}.mp4"
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: select attempts and probe the worst-case canvas size. ----
    selected = []  # list of (att_dir, meta, subtitle)
    max_h, max_w = 0, 0
    dummy = np.zeros((180, 320, 3), dtype=np.uint8)

    for att_dir in attempt_dirs:
        meta_path = att_dir / "meta.json"
        frames_path = att_dir / "frames.npy"
        if not meta_path.exists() or not frames_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        commands = meta.get("commands") or []
        if not commands:
            continue
        label = meta.get("label")
        if label is None and not include_unlabeled:
            continue
        if only is not None and label != only:
            continue

        attempt_id = meta["attempt_id"]
        notes = meta.get("notes", "")
        label_str = label if label is not None else "unlabeled"
        subtitle = f"Attempt {attempt_id} -- {label_str}"
        if notes:
            subtitle += f" ({notes})"

        for cmd_idx, cmd in enumerate(commands):
            probe = _annotate_frame(
                dummy,
                task=task,
                instruction=cmd["command"],
                trial_idx=attempt_id,
                turn_idx=cmd_idx,
                subtitle=subtitle,
            )
            ph, pw = probe.shape[:2]
            if ph > max_h:
                max_h = ph
            if pw > max_w:
                max_w = pw

        selected.append((att_dir, meta, subtitle))

    if not selected:
        raise RuntimeError(
            f"no attempts matched filters for task {task!r} "
            f"(only={only}, include_unlabeled={include_unlabeled})"
        )

    # H.264 / yuv420p requires even dimensions.
    if max_w % 2:
        max_w += 1
    if max_h % 2:
        max_h += 1

    # ---- Spin up ffmpeg as a subprocess writing libx264 + yuv420p mp4. ----
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{max_w}x{max_h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    n_written = 0
    title_card_frames = max(1, int(round(title_card_seconds * fps)))

    try:
        for att_dir, meta, subtitle in selected:
            commands = meta["commands"]
            boundaries = meta["cmd_boundaries"]
            attempt_id = meta["attempt_id"]
            label_str = (meta.get("label") or "unlabeled")
            notes = meta.get("notes", "")
            frames = np.load(att_dir / "frames.npy")

            # Title card before each attempt.
            card_text = f"Attempt {attempt_id}\n{label_str}"
            if notes:
                card_text += f"\n{notes}"
            card = _make_title_card(max_w, max_h, card_text)
            card_bytes = _pad_to(card, max_h, max_w).tobytes()
            for _ in range(title_card_frames):
                proc.stdin.write(card_bytes)
                n_written += 1

            for cmd_idx, cmd in enumerate(commands):
                instruction = cmd["command"]
                start = boundaries[cmd_idx]
                end = boundaries[cmd_idx + 1]
                for raw in frames[start:end]:
                    annotated = _annotate_frame(
                        raw,
                        task=task,
                        instruction=instruction,
                        trial_idx=attempt_id,
                        turn_idx=cmd_idx,
                        subtitle=subtitle,
                    )
                    padded = _pad_to(annotated, max_h, max_w)
                    proc.stdin.write(padded.tobytes())
                    n_written += 1
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", "replace")
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            f"ffmpeg exited with code {rc}\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr:\n{stderr}"
        )

    print(
        f"[render] task={task!r}  attempts={len(selected)}  "
        f"frames={n_written}  fps={fps}  size={max_w}x{max_h}  -> {out_path}"
    )
    return out_path


class Recorder:
    def __init__(
        self,
        task: str,
        env,
        run_fn: Callable,
        root: str = "~/recordings/lamer",
    ):
        self.task = task
        self.env = env
        self.run_fn = run_fn
        # Disable the env's baked-in 'instruction:' overlay (the white strip
        # at the top of every rendered frame). It only affects env.render();
        # _compute_observation still uses raw camera frames so the policy
        # is unaffected.
        self.env._render_text_in_image = False
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task)
        self.root = Path(root).expanduser() / safe
        self.root.mkdir(parents=True, exist_ok=True)
        self._start_new_attempt()

    def _start_new_attempt(self):
        existing = sorted(self.root.glob("attempt_*"))
        self.attempt_id = len(existing) + 1
        self.attempt_dir = self.root / f"attempt_{self.attempt_id:04d}"
        self.attempt_dir.mkdir()
        self.commands = []
        self.frames = [self.env.render(mode="rgb_array")]
        self.cmd_boundaries = [0]
        self.init_state = state_to_text(self.env._last_state)
        self._label: Optional[str] = None
        self._notes = ""
        self._save()
        print(f"[rec] started {self.attempt_dir.name}  (task={self.task!r})")

    def run(self, command: str, **kwargs):
        """Execute command via run_fn, log raw frames, return annotated frames.

        The returned frames have a top banner with `Trial N | Step K | <command>`
        and a bottom banner with the high-level task, so `show_rollout(frames)`
        directly shows both. Raw frames are still stored on disk for later use.
        """
        # Suppress run_fn's prints (including the env's misleading preset
        # `Task:` line) -- we print our own clean status below.
        with contextlib.redirect_stdout(io.StringIO()):
            frames, reward, info = self.run_fn(command, **kwargs)

        turn_idx = len(self.commands)
        self.commands.append({
            "command": command,
            "reward": float(reward),
            "won": bool(info.get("won", False)),
            "num_frames": int(len(frames)),
            "final_state": state_to_text(self.env._last_state),
            "timestamp": time.time(),
        })
        # Drop first frame -- duplicates last frame of previous command.
        self.frames.extend(frames[1:])
        self.cmd_boundaries.append(len(self.frames))
        self._save()

        steps = max(0, len(frames) - 1)
        print(f"[rec] {self.task!r}  turn {turn_idx}: {command!r}  ({steps} steps)")

        annotated = _annotate_frames(
            list(frames),
            task=self.task,
            instruction=command,
            trial_idx=self.attempt_id,
            turn_idx=turn_idx,
        )
        return annotated, reward, info

    def print_state(self):
        """Print the current env state with the high-level task in place of
        the env's misleading preset `Task:` instruction."""
        text = state_to_text(self.env._last_state)
        out = [f"Task: {self.task}"] + [
            line for line in text.split("\n") if not line.startswith("Task:")
        ]
        print("\n".join(out))

    def label(self, success: bool, notes: str = ""):
        """Mark the current attempt as success or failure."""
        self._label = "success" if success else "failure"
        self._notes = notes
        self._save()
        tag = f"[rec] {self.attempt_dir.name}: {self._label}"
        print(tag + (f" -- {notes}" if notes else ""))

    def reset(self):
        """Finalize the current attempt, reset env, start a new attempt."""
        if self.commands and self._label is None:
            print(f"[rec] warning: {self.attempt_dir.name} has no label")
        self.env.reset()
        self._start_new_attempt()

    def _save(self):
        meta = {
            "task": self.task,
            "attempt_id": self.attempt_id,
            "init_state": self.init_state,
            "commands": self.commands,
            "cmd_boundaries": self.cmd_boundaries,
            "label": self._label,
            "notes": self._notes,
        }
        (self.attempt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        np.save(self.attempt_dir / "frames.npy", np.stack(self.frames))

    def summary(self):
        """Aggregate labeled attempts for this task."""
        attempts = sorted(self.root.glob("attempt_*/meta.json"))
        labels = []
        for p in attempts:
            m = json.loads(p.read_text())
            if m.get("label") in ("success", "failure"):
                labels.append(m["label"] == "success")
        n_attempts = len(attempts)
        n_labeled = len(labels)
        n_wins = sum(labels)
        rate = f"  |  rate: {n_wins / n_labeled:.1%}" if n_labeled else ""
        print(
            f"Task: {self.task}  |  attempts: {n_attempts}  "
            f"|  labeled: {n_labeled}  |  successes: {n_wins}{rate}"
        )
