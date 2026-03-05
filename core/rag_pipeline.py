#!/usr/bin/env python3
"""
Synthetic RAG Pipeline - Synthesus 2.0
AIVM LLC

Implements the Synthetic RAG Reasoning System from DeepSeek design:
- FAISS vector index for semantic retrieval
- Character-aware context injection
- Batched embedding with sleep intervals (CPU-safe)
- Checkpoint-based migration (34% @ 290K patterns resume point)
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Synthetic RAG Pipeline for Synthesus.
    Retrieves relevant patterns and knowledge nodes from FAISS index.
    Injects context into right hemisphere queries.
    """

    def __init__(
        self,
        index_path: str = "./data/faiss.index",
        metadata_path: str = "./data/faiss_metadata.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        score_threshold: float = 0.6,
        batch_size: int = 256,
        batch_sleep_s: float = 0.5,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.batch_sleep_s = batch_sleep_s

        self._index: Optional[faiss.Index] = None
        self._metadata: List[Dict] = []
        self._embedder: Optional[SentenceTransformer] = None

        self._load()

    def _load(self):
        """Load FAISS index and metadata from disk."""
        logger.info("Loading RAG embedding model...")
        try:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return

        if self.index_path.exists():
            logger.info(f"Loading FAISS index from {self.index_path}...")
            self._index = faiss.read_index(str(self.index_path))
            logger.info(f"FAISS index loaded: {self._index.ntotal} vectors")
        else:
            logger.warning(f"FAISS index not found at {self.index_path}. Starting empty.")
            self._index = faiss.IndexFlatIP(384)  # Inner product for cosine sim

        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self._metadata = json.load(f)
            logger.info(f"Metadata loaded: {len(self._metadata)} entries")
        else:
            logger.warning("Metadata file not found. Starting empty.")
            self._metadata = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        embeddings = self._embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    async def retrieve(
        self,
        query: str,
        character_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.
        Returns dict with 'context' string and 'sources' list.
        """
        if self._index is None or self._index.ntotal == 0:
            return {"context": "", "sources": []}

        k = top_k or self.top_k

        # Run blocking embedding + search in executor
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._search(query, character_id, k)
        )

        if not results:
            return {"context": "", "sources": []}

        context_parts = []
        sources = []
        for score, meta in results:
            pattern = meta.get("pattern", "")
            response = meta.get("response", "")
            char = meta.get("character_id", "global")
            context_parts.append(f"Q: {pattern}\nA: {response}")
            sources.append({"pattern": pattern, "score": round(score, 4), "character": char})

        context = "\n\n".join(context_parts)
        return {"context": context, "sources": sources}

    def _search(self, query: str, character_id: Optional[str], k: int) -> List[Tuple[float, Dict]]:
        """Synchronous FAISS search."""
        try:
            query_emb = self._embed([query])
            scores, indices = self._index.search(query_emb, k * 2)  # over-fetch for filtering

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._metadata):
                    continue
                if score < self.score_threshold:
                    continue
                meta = self._metadata[idx]
                # Filter by character if specified
                if character_id and meta.get("character_id") not in (character_id, "global", None):
                    continue
                results.append((float(score), meta))
                if len(results) >= k:
                    break

            return results
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []

    def add_patterns(
        self,
        patterns: List[Dict],
        character_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ) -> int:
        """
        Add patterns to the FAISS index in CPU-safe batches.
        Supports checkpoint resume for large migrations.
        Returns number of patterns added.
        """
        total = len(patterns)
        added = 0
        checkpoint_file = Path(checkpoint_path) if checkpoint_path else None

        # Load checkpoint if exists
        start_idx = 0
        if checkpoint_file and checkpoint_file.exists():
            with open(checkpoint_file) as f:
                cp = json.load(f)
                start_idx = cp.get("last_batch_end", 0)
                logger.info(f"Resuming from checkpoint: {start_idx}/{total}")

        for batch_start in range(start_idx, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch = patterns[batch_start:batch_end]

            texts = [p.get("pattern", "") for p in batch]
            embeddings = self._embed(texts)

            self._index.add(embeddings)
            for p in batch:
                meta = dict(p)
                if character_id:
                    meta["character_id"] = character_id
                self._metadata.append(meta)

            added += len(batch)

            # Save checkpoint
            if checkpoint_file:
                with open(checkpoint_file, "w") as f:
                    json.dump({"last_batch_end": batch_end, "total": total}, f)

            logger.info(f"RAG migration: {batch_end}/{total} ({batch_end/total*100:.1f}%)")

            # CPU-safe sleep between batches
            time.sleep(self.batch_sleep_s)

        return added

    def save_index(self):
        """Save FAISS index and metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self._metadata, f)
        logger.info(f"FAISS index saved: {self._index.ntotal} vectors")

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": self.total_vectors,
            "metadata_entries": len(self._metadata),
            "index_path": str(self.index_path),
            "top_k": self.top_k,
            "score_threshold": self.score_threshold
        }
