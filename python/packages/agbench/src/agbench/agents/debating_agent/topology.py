"""
Topology configurations for multi-agent debate.
"""

from dataclasses import dataclass
from enum import Enum


class TopologyType(Enum):
    """Types of debater connection topologies."""
    CIRCULAR = "circular"        # Ring: each debater sees 2 neighbors
    GRID_2D = "grid_2d"          # 2D grid connectivity
    FULLY_CONNECTED = "fully"    # Everyone sees everyone
    PAIRS = "pairs"              # Debaters work in pairs


@dataclass
class TopologyConfig:
    """Configuration for debater connection topology."""

    topology_type: TopologyType = TopologyType.CIRCULAR
    num_debaters: int = 4
    debater_prefix: str = "Debater"

    def get_debater_ids(self) -> list[str]:
        """Get list of debater IDs."""
        ids = []
        for i in range(self.num_debaters):
            suffix = chr(ord('A') + i) if i < 26 else f"{i}"
            ids.append(f"{self.debater_prefix}{suffix}")
        return ids

    def get_connections(self) -> dict[str, list[str]]:
        """Get the connection mapping for this topology."""
        debater_ids = self.get_debater_ids()
        n = len(debater_ids)

        if self.topology_type == TopologyType.CIRCULAR:
            return self._build_circular(debater_ids)
        elif self.topology_type == TopologyType.GRID_2D:
            return self._build_grid_2d(debater_ids)
        elif self.topology_type == TopologyType.FULLY_CONNECTED:
            return self._build_fully_connected(debater_ids)
        elif self.topology_type == TopologyType.PAIRS:
            return self._build_pairs(debater_ids)
        else:
            raise ValueError(f"Unknown topology: {self.topology_type}")

    def _build_circular(self, debater_ids: list[str]) -> dict[str, list[str]]:
        """Ring topology: each debater sees left and right neighbors."""
        n = len(debater_ids)
        connections = {}
        for i, did in enumerate(debater_ids):
            left = debater_ids[(i - 1) % n]
            right = debater_ids[(i + 1) % n]
            connections[did] = [left, right] if n > 2 else ([left] if n == 2 else [])
        return connections

    def _build_grid_2d(self, debater_ids: list[str]) -> dict[str, list[str]]:
        """2D grid: each debater sees up/down/left/right neighbors."""
        n = len(debater_ids)
        rows = int(n ** 0.5)
        while n % rows != 0 and rows > 1:
            rows -= 1
        cols = n // rows

        connections = {}
        for i, did in enumerate(debater_ids):
            row, col = i // cols, i % cols
            neighbors = []
            if row > 0:
                neighbors.append(debater_ids[(row - 1) * cols + col])
            if row < rows - 1:
                neighbors.append(debater_ids[(row + 1) * cols + col])
            if col > 0:
                neighbors.append(debater_ids[row * cols + (col - 1)])
            if col < cols - 1:
                neighbors.append(debater_ids[row * cols + (col + 1)])
            connections[did] = neighbors
        return connections

    def _build_fully_connected(self, debater_ids: list[str]) -> dict[str, list[str]]:
        """Fully connected: every debater sees all others."""
        return {did: [d for d in debater_ids if d != did] for did in debater_ids}

    def _build_pairs(self, debater_ids: list[str]) -> dict[str, list[str]]:
        """Pairs: debaters work in pairs."""
        n = len(debater_ids)
        connections = {}
        for i in range(0, n - 1, 2):
            connections[debater_ids[i]] = [debater_ids[i + 1]]
            connections[debater_ids[i + 1]] = [debater_ids[i]]
        if n % 2 == 1:
            connections[debater_ids[-1]] = [debater_ids[0]]
            connections[debater_ids[0]].append(debater_ids[-1])
        return connections

    def get_num_neighbors(self, debater_id: str) -> int:
        """Get number of neighbors for a debater."""
        return len(self.get_connections().get(debater_id, []))

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if self.num_debaters < 2:
            errors.append("num_debaters must be at least 2")
        return errors
