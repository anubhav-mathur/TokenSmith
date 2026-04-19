from typing import List, Dict, Optional, Any
from tests.metrics.base import MetricBase


class ChunkRetrievalMetric(MetricBase):
    """
    Chunk retrieval evaluation metric.

    Primary score is F1 (normalized [0, 1]).
    Precision, recall, and raw count are stored separately via calculate_detailed().
    Weight=0 so this does not affect final_score — it is a supplementary metric.
    """

    @property
    def name(self) -> str:
        return "chunk_retrieval"

    @property
    def weight(self) -> float:
        return 0.0

    def calculate(self,
                  ideal_retrieved_chunks: Optional[List[int]],
                  retrieved_chunks: Optional[List[Dict[str, Any]]]) -> float:
        """Return F1 score in [0, 1]. Returns 0.0 if either input is missing."""
        return self.calculate_detailed(ideal_retrieved_chunks, retrieved_chunks)["f1"]

    def calculate_detailed(self,
                           ideal_retrieved_chunks: Optional[List[int]],
                           retrieved_chunks: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Returns precision, recall, F1, and raw hit count.

        Args:
            ideal_retrieved_chunks: Ground-truth chunk IDs from benchmarks.yaml.
            retrieved_chunks: List of dicts from get_answer(), each with a 'chunk_id' key.
        """
        if not ideal_retrieved_chunks or not retrieved_chunks:
            return {"count": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        ideal_set = set(ideal_retrieved_chunks)
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        retrieved_set = set(retrieved_ids)

        true_positives = len(ideal_set & retrieved_set)

        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall    = true_positives / len(ideal_set)     if ideal_set    else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            "count":     true_positives,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }
