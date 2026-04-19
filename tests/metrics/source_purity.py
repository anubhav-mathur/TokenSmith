import sqlite3
import pathlib
from typing import Dict, List, Optional, Any
from tests.metrics.base import MetricBase


class SourcePurityMetric(MetricBase):
    """
    Measures what fraction of retrieved chunks came from the expected document type.

    Only meaningful for benchmarks that set ground_truth_doc_type.
    Weight=0 so this does not affect final_score — it is a supplementary metric.

    Loads the document registry from index/sections/document_registry.db at
    construction time to build a chunk_id -> doc_type lookup table.
    is_available() returns False if the registry file is not present.
    """

    _REGISTRY_PATH = pathlib.Path("index/sections/document_registry.db")

    def __init__(self):
        self._chunk_to_doctype: Dict[int, str] = {}
        self._registry_loaded = False
        self._load_registry()

    def _load_registry(self) -> None:
        if not self._REGISTRY_PATH.exists():
            return
        try:
            conn = sqlite3.connect(str(self._REGISTRY_PATH))
            rows = conn.execute(
                "SELECT chunk_start, chunk_end, doc_type FROM documents"
            ).fetchall()
            conn.close()
            for chunk_start, chunk_end, doc_type in rows:
                for chunk_id in range(chunk_start, chunk_end + 1):
                    self._chunk_to_doctype[chunk_id] = doc_type
            self._registry_loaded = True
        except Exception as exc:
            print(f"[SourcePurityMetric] Could not load registry: {exc}")

    def is_available(self) -> bool:
        return self._registry_loaded

    @property
    def name(self) -> str:
        return "source_purity"

    @property
    def weight(self) -> float:
        return 0.0

    def calculate(self,
                  retrieved_chunks: Optional[List[Dict[str, Any]]],
                  ground_truth_doc_type: Optional[str]) -> float:
        """Return purity score in [0, 1]."""
        return self.calculate_detailed(retrieved_chunks, ground_truth_doc_type)["purity"]

    def calculate_detailed(self,
                           retrieved_chunks: Optional[List[Dict[str, Any]]],
                           ground_truth_doc_type: Optional[str]) -> Dict[str, Any]:
        """
        Returns purity, correct_count, and total_count.

        Args:
            retrieved_chunks: List of dicts from get_answer(), each with a 'chunk_id' key.
            ground_truth_doc_type: Expected doc_type string ('document', 'slides', 'paper').
        """
        empty = {"purity": 0.0, "correct_count": 0, "total_count": 0}
        if not retrieved_chunks or not ground_truth_doc_type or not self._registry_loaded:
            return empty

        total = len(retrieved_chunks)
        correct = sum(
            1 for chunk in retrieved_chunks
            if self._chunk_to_doctype.get(chunk["chunk_id"]) == ground_truth_doc_type
        )
        purity = correct / total if total > 0 else 0.0

        return {
            "purity":        round(purity, 4),
            "correct_count": correct,
            "total_count":   total,
        }
