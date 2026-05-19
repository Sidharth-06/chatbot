from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int = 280) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


@dataclass
class MemoryHit:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: Optional[float] = None


def _build_embedding_function(embedding_model: str):
    import chromadb
    from chromadb.utils import embedding_functions as embedding_functions

    model = embedding_model.strip()
    if model in {"chroma-default", "default", "onnx"}:
        return embedding_functions.DefaultEmbeddingFunction()

    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for custom embedding models. "
            "Set MEMORY_EMBEDDING_MODEL=chroma-default or install sentence-transformers."
        ) from exc

    return SentenceTransformerEmbeddingFunction(model_name=model)


def _collection_name(embedding_model: str) -> str:
    model = embedding_model.strip()
    if model in {"chroma-default", "default", "onnx"}:
        return "chatbot_memory_light"
    safe = model.replace("/", "_").replace(":", "_")
    return f"chatbot_memory_{safe}"[:63]


class PersistentMemoryStore:
    def __init__(self, persist_dir: str | Path, embedding_model: str = "chroma-default") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - surfaced in the UI
            raise RuntimeError(
                "ChromaDB is required for memory storage. Install the project dependencies first."
            ) from exc

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._embedder = _build_embedding_function(embedding_model)
        self._collection = self._client.get_or_create_collection(
            name=_collection_name(embedding_model),
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model.strip()},
        )

    def _store(self, item_id: str, document: str, metadata: dict[str, Any]) -> None:
        payload = {
            "ids": [item_id],
            "documents": [document],
            "metadatas": [metadata],
        }

        if hasattr(self._collection, "upsert"):
            self._collection.upsert(**payload)
            return

        self._collection.delete(ids=[item_id])
        self._collection.add(**payload)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn_index: int,
        kind: str = "message",
    ) -> str:
        item_id = f"{session_id}:{turn_index}:{role}:{uuid4().hex}"
        metadata = {
            "session_id": session_id,
            "role": role,
            "kind": kind,
            "turn_index": turn_index,
            "created_at": _utc_now(),
        }
        self._store(item_id, content, metadata)
        return item_id

    def store_summary(self, session_id: str, summary: str) -> str:
        item_id = self._summary_id(session_id)
        metadata = {
            "session_id": session_id,
            "role": "system",
            "kind": "summary",
            "created_at": _utc_now(),
        }
        self._store(item_id, summary, metadata)
        return item_id

    def load_summary(self, session_id: str) -> str:
        result = self._collection.get(ids=[self._summary_id(session_id)], include=["documents"])
        documents = result.get("documents") or []
        if documents and documents[0]:
            return str(documents[0][0])
        return ""

    def search(
        self,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
    ) -> list[MemoryHit]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        where = {"session_id": session_id} if session_id else None
        result = self._collection.query(
            query_texts=[cleaned_query],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[MemoryHit] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            hit_id = ids[index] if index < len(ids) else f"hit-{index}"
            hits.append(MemoryHit(id=str(hit_id), document=str(document), metadata=dict(metadata), distance=distance))
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    @staticmethod
    def _summary_id(session_id: str) -> str:
        return f"{session_id}:summary"


def format_memory_hits(hits: list[MemoryHit]) -> str:
    if not hits:
        return "No relevant long-term memories were retrieved."

    lines = []
    for index, hit in enumerate(hits, 1):
        role = str(hit.metadata.get("role", "memory"))
        kind = str(hit.metadata.get("kind", "message"))
        created_at = str(hit.metadata.get("created_at", ""))
        prefix = f"{index}. [{kind}/{role}]"
        suffix = f" ({created_at})" if created_at else ""
        lines.append(f"{prefix} {_truncate(hit.document)}{suffix}")
    return "\n".join(lines)


def format_recent_messages(messages: list[dict[str, str]]) -> str:
    if not messages:
        return ""
    lines = []
    for message in messages:
        role = str(message.get("role", "user")).title()
        content = _truncate(message.get("content", ""), 260)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def refresh_summary(
    client: Any,
    model: str,
    existing_summary: str,
    recent_messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 220,
) -> str:
    if not recent_messages and not existing_summary:
        return ""

    prompt = [
        {
            "role": "system",
            "content": (
                "You maintain durable memory for a chatbot. Update the memory summary with only stable facts, "
                "user preferences, open tasks, important context, and unresolved decisions. Remove chatter, "
                "small talk, and one-off details. Return only the updated summary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Existing memory summary:\n{existing_summary or '(none)'}\n\n"
                f"Recent conversation:\n{format_recent_messages(recent_messages)}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return str(response.choices[0].message.content or "").strip()
