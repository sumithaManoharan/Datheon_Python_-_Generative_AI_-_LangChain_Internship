"""Persistent vector store with idempotent, hash-based ingestion."""

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb

from docqa.chunking import Chunk
from docqa.embeddings import Embedder
from docqa.errors import DocumentReadError, StoreError

logger = logging.getLogger(__name__)

COLLECTION_NAME = "docqa"
UPSERT_BATCH = 256
HASH_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk with its citation fields and cosine distance (0 = identical)."""

    text: str
    source: str
    distance: float
    page: int | None = None
    section: str | None = None
    start_line: int | None = None
    end_line: int | None = None


def file_sha256(path: Path) -> str:
    """Hash a file's bytes, reading in blocks so large files stay cheap on memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(HASH_BLOCK_BYTES), b""):
                digest.update(block)
    except OSError as exc:
        raise DocumentReadError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _metadata(chunk: Chunk, file_hash: str) -> dict[str, str | int]:
    """Chroma metadata for a chunk. Chroma rejects None, so unset fields are omitted."""
    meta: dict[str, str | int] = {"source": str(chunk.source), "file_hash": file_hash}
    if chunk.page is not None:
        meta["page"] = chunk.page
    if chunk.section is not None:
        meta["section"] = chunk.section
    if chunk.start_line is not None:
        meta["start_line"] = chunk.start_line
    if chunk.end_line is not None:
        meta["end_line"] = chunk.end_line
    return meta


class VectorStore:
    """A Chroma collection on disk. The embedder is injected.

    A file's chunks are keyed by `str(source)` and tagged with the file's content
    hash. Ingesting an unchanged file does nothing; a changed file is replaced whole.
    """

    def __init__(
        self,
        path: Path,
        embedder: Embedder,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._embedder = embedder
        try:
            self._client = chromadb.PersistentClient(path=str(path))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "embedder": embedder.name},
            )
        except Exception as exc:
            raise StoreError(f"cannot open vector store at {path}: {exc}") from exc
        stored = (self._collection.metadata or {}).get("embedder")
        if stored != embedder.name:
            raise StoreError(
                f"store at {path} was built with embedder {stored!r}, not {embedder.name!r}; "
                "delete the store or use the original model"
            )

    def count(self) -> int:
        """Number of chunks in the store."""
        return self._collection.count()

    def known_hash(self, source: Path) -> str | None:
        """The stored content hash for a file, or None if the file is not ingested."""
        found = self._collection.get(where={"source": str(source)}, limit=1, include=["metadatas"])
        metadatas = found["metadatas"] or []
        if not metadatas:
            return None
        value = metadatas[0].get("file_hash")
        return str(value) if value is not None else None

    def ingest_file(self, source: Path, file_hash: str, chunks: Sequence[Chunk]) -> bool:
        """Store a file's chunks. Returns False when the file was unchanged or empty.

        Embedding happens before the old chunks are deleted, so a failed embedding
        call leaves the previous version of the file in place.
        """
        key = str(source)
        if self.known_hash(source) == file_hash:
            logger.info("skipping unchanged file %s", key)
            return False
        vectors = self._embedder.embed_documents([chunk.embed_text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise StoreError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
        self._delete_source(key)
        try:
            for start in range(0, len(chunks), UPSERT_BATCH):
                batch = chunks[start : start + UPSERT_BATCH]
                self._collection.upsert(
                    ids=[f"{key}::{start + i}" for i in range(len(batch))],
                    embeddings=vectors[start : start + len(batch)],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[_metadata(chunk, file_hash) for chunk in batch],
                )
        except Exception as exc:
            # A half-written file would carry the new hash and be skipped next time.
            self._delete_source(key)
            raise StoreError(f"could not store {key}: {exc}") from exc
        logger.info("stored %d chunks from %s", len(chunks), key)
        return bool(chunks)

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Return the k nearest chunks, closest first."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        hits: list[SearchHit] = []
        for text, meta, distance in zip(documents, metadatas, distances, strict=True):
            hits.append(
                SearchHit(
                    text=text,
                    source=str(meta["source"]),
                    distance=float(distance),
                    page=int(meta["page"]) if "page" in meta else None,
                    section=str(meta["section"]) if "section" in meta else None,
                    start_line=int(meta["start_line"]) if "start_line" in meta else None,
                    end_line=int(meta["end_line"]) if "end_line" in meta else None,
                )
            )
        return hits

    def _delete_source(self, key: str) -> None:
        """Remove every chunk that came from one file."""
        self._collection.delete(where={"source": key})
