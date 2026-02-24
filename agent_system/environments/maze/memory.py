from typing import List, Tuple, Dict, Union, Any


class SimpleMemoryMaze:
    """
    Memory manager: responsible for storing & fetching per-environment history records.
    Tracks planned vs executed actions to support reflection on stochastic effects.
    """

    def __init__(self, num_processes=0):
        self._data = [{} for _ in range(num_processes)]
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
        Store a new record (one turn of history) for each environment instance.

        Args:
            record: Dictionary where each key maps to a list of length num_processes.
                Expected keys: 'text_obs', 'planned_actions', 'executed_actions',
                               'reward', 'dones', 'won'
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
        obs_length: int = 2,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.

        Shows planned vs executed actions to help the agent reflect on stochastic effects.

        Args:
            history_length: Maximum number of past turns to retrieve per environment.
            obs_key: Key name for observations in stored records.
            obs_length: How many recent turns should include full observations.

        Returns:
            memory_contexts: List of formatted strings, one per environment.
            valid_lengths: Number of valid turns per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.num_processes):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                turn_num = start_idx + j + 1
                planned = rec.get('planned_actions', '')
                executed = rec.get('executed_actions', '')
                obs = rec[obs_key]

                # Format planned vs executed
                action_line = f"Turn {turn_num}: Planned: {planned} | Executed: {executed}"

                if len(recent) - j > obs_length:
                    lines.append(f"{action_line}\nObservation {turn_num}: ...")
                else:
                    lines.append(f"{action_line}\nObservation {turn_num}:\n{obs}")

                if 'dones' in rec and rec['dones']:
                    valid_len = turn_num
                    break

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
