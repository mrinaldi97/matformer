
from abc import ABC, abstractmethod

class MatformerDataSaver(ABC):
    """Rank-aware result serialiser. All ranks must call save(); only rank 0 writes."""

    @abstractmethod
    def save_batch(self, records: list[dict], rank: int = 0) -> None:
        """Gather records across ranks (if distributed) and persist on rank 0.

        Args:
            records: this rank's batch records.
            rank: current process rank.
        """
        ...
    @abstractmethod
    def save_summary(self, summary, rank: int = 0) -> None:
        """Gather records across ranks (if distributed) and persist on rank 0.

        Args:
            records: this rank's batch records.
            rank: current process rank.
        """
        ...
    def end(self) -> None:
        """Close any open resources."""
        pass

