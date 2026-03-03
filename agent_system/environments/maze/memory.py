from typing import List, Dict, Any


class SimpleMemoryMaze:
    """
    Memory manager: stores and fetches per-environment trajectory history.

    Each stored record is one "turn" (one episodic action string + resulting
    observation).  Because the maze uses an episodic wrapper, a single turn
    covers the entire action sequence for one attempt.
    """

    def __init__(self, num_processes: int = 0):
        self._data: List[List[Dict]] = [[] for _ in range(num_processes)]
        self.keys = None
        self.num_processes = num_processes

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, num_processes: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(num_processes)]
        self.num_processes = num_processes
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store one step of history for every environment.

        Parameters
        ----------
        record : Dict[str, List[Any]]
            Keys map to per-environment lists, e.g.
            ``{'text_obs': [...], 'thought': [...], 'action': [...],
               'reward': [...], 'dones': [...], 'won': [...]}``.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.num_processes):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int = 7,
        obs_key: str = "text_obs",
        action_key: str = "action",
        obs_length: int = 2,
    ):
        """
        Return formatted trajectory strings and their valid lengths.

        Parameters
        ----------
        history_length : int
            Maximum number of past steps to include per environment.
        obs_key : str
            Key used to retrieve observation text from stored records.
        action_key : str
            Key used to retrieve the action string from stored records.
        obs_length : int
            Show full observations only for the most recent ``obs_length``
            steps; earlier steps show ``...`` to save context length.

        Returns
        -------
        memory_contexts : List[str]
        valid_lengths : List[int]
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.num_processes):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]

                if len(recent) - j > obs_length:
                    lines.append(
                        f"Action {step_num}: {act}\nObservation {step_num}: ..."
                    )
                else:
                    lines.append(
                        f"Action {step_num}: {act}\nObservation {step_num}:\n{obs}"
                    )
                if 'dones' in rec and rec['dones']:
                    valid_len = step_num
                    break

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
