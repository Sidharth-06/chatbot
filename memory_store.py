from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


_WORD_RE = re.compile(r"[a-z0-9]+")
_STORE_FILENAME = "memory_store.json"


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


def _tokenize(text: str) -> set[str]:
    return {match.group(0) for match in _WORD_RE.finditer(text.lower())}


def _score_document(query_tokens: set[str], document: str, metadata: dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0

    document_tokens = _tokenize(document)
    overlap = len(query_tokens & document_tokens)
    if not overlap:
        return 0.0

    score = overlap / len(query_tokens)
    kind = str(metadata.get("kind", "message"))
    if kind == "summary":
        score *= 0.85

    turn_index = metadata.get("turn_index")
    if isinstance(turn_index, int) and turn_index >= 0:
        score += min(turn_index, 1000) / 10000.0

    return score


def _read_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "messages": [], "summaries": {}}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "messages": [], "summaries": {}}

    if not isinstance(data, dict):
        return {"version": 1, "messages": [], "summaries": {}}

    data.setdefault("version", 1)
    data.setdefault("messages", [])
    data.setdefault("summaries", {})
    if not isinstance(data["messages"], list):
        data["messages"] = []
    if not isinstance(data["summaries"], dict):
        data["summaries"] = {}
    return data


def _write_store(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


class PersistentMemoryStore:
    def __init__(self, persist_dir: str | Path, embedding_model: str = "chroma-default") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._store_path = self.persist_dir / _STORE_FILENAME
        self._data = _read_store(self._store_path)
        self._embedding_model = embedding_model.strip()

    def _store(self, item_id: str, document: str, metadata: dict[str, Any]) -> None:
        record = {
            "id": item_id,
            "document": document,
            "metadata": metadata,
        }

        messages = self._data.setdefault("messages", [])
        messages = [existing for existing in messages if existing.get("id") != item_id]
        messages.append(record)
        self._data["messages"] = messages
        _write_store(self._store_path, self._data)

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
        summaries = self._data.setdefault("summaries", {})
        summaries[session_id] = {
            "id": item_id,
            "document": summary,
            "metadata": metadata,
        }
        self._data["summaries"] = summaries
        _write_store(self._store_path, self._data)
        return item_id

    def load_summary(self, session_id: str) -> str:
        summaries = self._data.get("summaries", {})
        summary = summaries.get(session_id, {})
        document = summary.get("document")
        if document:
            return str(document)
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

        query_tokens = _tokenize(cleaned_query)
        hits: list[MemoryHit] = []
        for record in self._data.get("messages", []):
            if not isinstance(record, dict):
                continue

            metadata = dict(record.get("metadata", {}))
            if session_id and metadata.get("session_id") != session_id:
                continue

            document = str(record.get("document", ""))
            score = _score_document(query_tokens, document, metadata)
            if score <= 0:
                continue

            hits.append(
                MemoryHit(
                    id=str(record.get("id", f"hit-{len(hits)}")),
                    document=document,
                    metadata=metadata,
                    distance=1.0 - min(score, 1.0),
                )
            )

        hits.sort(key=lambda hit: (hit.distance if hit.distance is not None else 1.0, hit.metadata.get("turn_index", 0)))
        return hits

    def count(self) -> int:
        return len(self._data.get("messages", [])) + len(self._data.get("summaries", {}))

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
