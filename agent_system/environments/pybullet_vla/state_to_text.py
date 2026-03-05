"""
Convert pybullet state dictionaries to natural-language descriptions
that the outer LLM can consume.

Each state dict is expected to have (at minimum):
    - ``"joint_positions"``: list/array of joint angles
    - ``"object_poses"``: dict mapping object name → (position, orientation)
    - ``"gripper_state"``: float (0 = closed, 1 = open)
    - ``"ee_position"``: (x, y, z) end-effector position

Additional keys are silently ignored, making this forward-compatible as
the pybullet wrapper evolves.
"""

from typing import Any, Dict, List


def state_to_text(state: Dict[str, Any]) -> str:
    """Convert a single pybullet state dict to a text description.

    Parameters
    ----------
    state : dict
        Pybullet state dictionary (see module docstring for expected keys).

    Returns
    -------
    str
        Human-readable description of the robot/scene state.
    """
    parts: List[str] = []

    # End-effector position
    ee = state.get("ee_position")
    if ee is not None:
        parts.append(
            f"End-effector position: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})"
        )

    # Gripper
    gripper = state.get("gripper_state")
    if gripper is not None:
        status = "open" if gripper > 0.5 else "closed"
        parts.append(f"Gripper: {status} ({gripper:.2f})")

    # Joint positions
    joints = state.get("joint_positions")
    if joints is not None:
        joint_strs = ", ".join(f"{j:.2f}" for j in joints)
        parts.append(f"Joint positions: [{joint_strs}]")

    # Object poses
    objects = state.get("object_poses")
    if objects:
        obj_lines = []
        for name, (pos, orn) in objects.items():
            pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            obj_lines.append(f"  {name}: position={pos_str}")
        parts.append("Objects:\n" + "\n".join(obj_lines))

    # Task info (optional)
    task_info = state.get("task_info")
    if task_info:
        parts.append(f"Task: {task_info}")

    return "\n".join(parts) if parts else "No state information available."


def batch_state_to_text(states: List[Dict[str, Any]]) -> List[str]:
    """Convert a batch of pybullet states to text descriptions.

    Parameters
    ----------
    states : list[dict]
        List of pybullet state dictionaries.

    Returns
    -------
    list[str]
        One text description per environment.
    """
    return [state_to_text(s) for s in states]
