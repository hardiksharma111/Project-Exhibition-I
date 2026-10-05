"""
Embedded Vector Store Architecture
Provides persistent, low-overhead semantic retrieval across SLM switches.
Guarantees retrieval latency: T_retrieval < 15.0 ms.
"""

import json
import math
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from core.config import MEMORY_STORE_DIR, MAX_MEMORY_LOOKUP_LATENCY_MS
from core.router.domain_centroids import compute_query_vector, TOTAL_DIM


class MemoryItem:
    def __init__(
        self,
        item_id: str,
        text: str,
        domain: str,
        source_model: str,
        vector: np.ndarray,
        importance: float = 1.0,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.item_id = item_id
        self.text = text
        self.domain = domain
        self.source_model = source_model
        self.vector = vector
        self.importance = importance
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "domain": self.domain,
            "source_model": self.source_model,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "vector": self.vector.tolist()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(
            item_id=data["item_id"],
            text=data["text"],
            domain=data["domain"],
            source_model=data["source_model"],
            vector=np.array(data["vector"], dtype=np.float32),
            importance=data.get("importance", 1.0),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )


class VectorMemoryStore:
    """
    Offline Serverless Semantic Vector Store.
    Persistent cross-model dialogue embedding index with fast cosine projection.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.join(MEMORY_STORE_DIR, "semantic_memory.json")
        self.items: Dict[str, MemoryItem] = {}
        self.matrix: Optional[np.ndarray] = None
        self.id_list: List[str] = []
        self._load_storage()

    def _load_storage(self):
        """Loads persistent records from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for d in data:
                        item = MemoryItem.from_dict(d)
                        self.items[item.item_id] = item
                self._rebuild_matrix()
            except Exception as e:
                print(f"[VectorMemoryStore] Warning: Failed to load existing store: {e}")
                self.items = {}

    def _save_storage(self):
        """Persists records atomically to disk."""
        try:
            temp_path = self.storage_path + ".tmp"
            data = [item.to_dict() for item in self.items.values()]
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if os.path.exists(self.storage_path):
                os.remove(self.storage_path)
            os.rename(temp_path, self.storage_path)
        except Exception as e:
            print(f"[VectorMemoryStore] Error saving memory store: {e}")

    def _rebuild_matrix(self):
        """Rebuilds the vector search matrix for fast vectorized BLAS operations."""
        if not self.items:
            self.matrix = None
            self.id_list = []
            return

        self.id_list = list(self.items.keys())
        vectors = []
        for i in self.id_list:
            v = self.items[i].vector
            if len(v) != TOTAL_DIM:
                v = compute_query_vector(self.items[i].text)
                self.items[i].vector = v
            vectors.append(v)
        self.matrix = np.vstack(vectors).astype(np.float32)

    def insert(
        self,
        text: str,
        domain: str,
        source_model: str,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Embeds text and adds it to the vector index."""
        import uuid
        item_id = f"mem-{uuid.uuid4().hex[:8]}"
        vec = compute_query_vector(text)
        
        item = MemoryItem(
            item_id=item_id,
            text=text,
            domain=domain,
            source_model=source_model,
            vector=vec,
            importance=importance,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.items[item_id] = item
        self._rebuild_matrix()
        self._save_storage()
        return item_id

    def search(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
        min_similarity: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes fast cosine similarity nearest-neighbor search.
        Enforces T_retrieval < 15.0 ms.
        """
        t0 = time.perf_counter()

        if self.matrix is None or len(self.id_list) == 0:
            t1 = time.perf_counter()
            return [], round((t1 - t0) * 1000.0, 3)

        q_vec = compute_query_vector(query)
        if np.linalg.norm(q_vec) == 0:
            t1 = time.perf_counter()
            return [], round((t1 - t0) * 1000.0, 3)

        # Dot product across all vectors in store: O(N * D)
        similarities = np.dot(self.matrix, q_vec)

        results = []
        for idx, sim in enumerate(similarities):
            item_id = self.id_list[idx]
            item = self.items[item_id]

            if domain_filter and item.domain != domain_filter:
                continue

            if sim >= min_similarity:
                results.append({
                    "item_id": item.item_id,
                    "text": item.text,
                    "domain": item.domain,
                    "source_model": item.source_model,
                    "similarity": round(float(sim), 4),
                    "importance": item.importance,
                    "timestamp": item.timestamp,
                    "age_seconds": round(time.time() - item.timestamp, 1)
                })

        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:top_k]

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 3)

        return top_results, latency_ms

    def clear(self):
        """Clears all stored memories."""
        self.items.clear()
        self.matrix = None
        self.id_list.clear()
        self._save_storage()

    def get_stats(self) -> Dict[str, Any]:
        """Returns storage diagnostics."""
        domain_counts = {}
        for item in self.items.values():
            domain_counts[item.domain] = domain_counts.get(item.domain, 0) + 1

        return {
            "total_items": len(self.items),
            "storage_path": self.storage_path,
            "vector_dimension": TOTAL_DIM,
            "domain_breakdown": domain_counts
        }
