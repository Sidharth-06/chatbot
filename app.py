from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from memory_store import MemoryHit, PersistentMemoryStore, format_memory_hits, refresh_summary


APP_DIR = Path(__file__).resolve().parent


def load_app_env() -> None:
    """Load .env from the app directory so Streamlit cwd does not matter."""
    load_dotenv(APP_DIR / ".env")


load_app_env()

APP_TITLE = "OpenRouter Memory Chatbot"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MEMORY_DIR = os.getenv("MEMORY_PERSIST_DIR", ".chat_memory")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful chatbot. Use relevant long-term memory when it helps answer the user. "
    "Prefer concise answers, ask clarifying questions when needed, and do not invent facts from memory."
)


st.set_page_config(page_title=APP_TITLE, page_icon="", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu, footer, header { visibility: hidden; }
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 28%),
          radial-gradient(circle at top right, rgba(168, 85, 247, 0.10), transparent 24%),
          linear-gradient(180deg, #07111d 0%, #0b1726 50%, #07111d 100%);
        color: #e8eef7;
      }
      .block-container {
        padding-top: 1.2rem !important;
        max-width: 1240px;
      }
      div[data-testid="stChatMessage"] {
        border-radius: 18px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def secret_or_env(section: str, key: str, env_key: str, default: str = "") -> str:
    try:
        section_data = st.secrets[section]
    except Exception:
        section_data = {}

    if hasattr(section_data, "get"):
        value = section_data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    try:
        flat_value = st.secrets[env_key]
        if flat_value is not None and str(flat_value).strip():
            return str(flat_value).strip()
    except Exception:
        pass

    env_value = os.getenv(env_key)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()

    return default


def init_state() -> None:
    st.session_state.setdefault("session_id", uuid4().hex)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("memory_summary", "")
    st.session_state.setdefault("summary_refreshes", 0)
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)


@st.cache_resource(show_spinner="Loading memory store...")
def get_memory_store(persist_dir: str) -> PersistentMemoryStore:
    return PersistentMemoryStore(persist_dir=persist_dir)


def api_key_is_configured(api_key: str) -> bool:
    cleaned = api_key.strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered.startswith("your_") or lowered in {"changeme", "replace_me", "xxx"}:
        return False
    return True


@st.cache_resource(show_spinner=False)
def get_openrouter_client(api_key: str, base_url: str, app_name: str) -> OpenAI:
    referer = secret_or_env(
        "openrouter",
        "http_referer",
        "OPENROUTER_HTTP_REFERER",
        "https://share.streamlit.io",
    )
    default_headers = {
        "HTTP-Referer": referer,
        "X-Title": app_name,
    }
    return OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), default_headers=default_headers)


def build_context_messages(
    system_prompt: str,
    memory_summary: str,
    memory_hits: list[MemoryHit],
    recent_messages: list[ChatCompletionMessageParam],
    user_prompt: str,
) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt.strip()},
    ]

    memory_parts: list[str] = []
    if memory_summary.strip():
        memory_parts.append(f"Long-term memory summary:\n{memory_summary.strip()}")
    if memory_hits:
        memory_parts.append(f"Relevant retrieved memories:\n{format_memory_hits(memory_hits)}")
    if memory_parts:
        messages.append({"role": "system", "content": "\n\n".join(memory_parts)})

    messages.extend(recent_messages)
    messages.append({"role": "user", "content": user_prompt.strip()})
    return messages


def stream_openrouter_response(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def render_chat_history(messages: list[dict[str, str]]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    init_state()

    api_key = secret_or_env("openrouter", "api_key", "OPENROUTER_API_KEY")
    base_url = secret_or_env("openrouter", "base_url", "OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    model = secret_or_env("openrouter", "model", "OPENROUTER_MODEL", DEFAULT_MODEL)
    memory_dir = secret_or_env("memory", "persist_dir", "MEMORY_PERSIST_DIR", DEFAULT_MEMORY_DIR)

    st.title(APP_TITLE)
    st.caption("Chat with OpenRouter models and keep durable memory in a local persistent store.")
    st.caption("Configuration is loaded from your environment variables or Streamlit secrets.")

    top_controls = st.columns([1, 1, 1, 1])
    with top_controls[0]:
        memory_limit = st.slider("Memory hits", min_value=1, max_value=10, value=5, step=1)
    with top_controls[1]:
        recent_turns = st.slider("Recent turns", min_value=1, max_value=12, value=4, step=1)
    with top_controls[2]:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.5, step=0.1)
    with top_controls[3]:
        top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

    bottom_controls = st.columns([1, 1, 2])
    with bottom_controls[0]:
        max_tokens = st.slider("Max tokens", min_value=128, max_value=4096, value=1024, step=64)
    with bottom_controls[1]:
        session_only_memory = st.checkbox("Limit memory lookup to this session", value=False)
    with bottom_controls[2]:
        if st.button("Clear chat"):
            st.session_state["messages"] = []
            st.session_state["memory_summary"] = ""
            st.session_state["summary_refreshes"] = 0
            st.rerun()

    with st.expander("System prompt", expanded=False):
        st.session_state["system_prompt"] = st.text_area(
            "Assistant instructions",
            value=st.session_state["system_prompt"],
            height=180,
            label_visibility="collapsed",
        )

    env_path = APP_DIR / ".env"
    if not api_key_is_configured(api_key):
        st.error(
            "OpenRouter API key is missing or still a placeholder. Add a real `OPENROUTER_API_KEY=sk-or-...` to "
            f"`{env_path}` (copy from `.env.example`) or set it in `.streamlit/secrets.toml`, then restart Streamlit."
        )
        st.stop()

    try:
        client = get_openrouter_client(api_key.strip(), base_url.strip(), APP_TITLE)
    except Exception as exc:
        st.error(f"Could not initialize OpenRouter client: {exc}")
        st.stop()

    memory_store: PersistentMemoryStore | None = None
    try:
        memory_store = get_memory_store(memory_dir)
    except Exception as exc:
        st.warning(
            "Long-term memory is disabled for this session. "
            f"Chat still works, but past messages will not be stored or retrieved. ({exc})"
        )

    if memory_store is not None:
        st.caption(f"Memory items stored: {memory_store.count():,}")

    if not st.session_state["messages"]:
        st.info("Ask a question below to start chatting.")

    render_chat_history(st.session_state["messages"])

    user_prompt = st.chat_input("Ask something...")
    if user_prompt:
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        user_turn_count = sum(1 for message in st.session_state["messages"] if message["role"] == "user")
        recent_message_count = max(recent_turns * 2, 1)
        recent_history = st.session_state["messages"][:-1][-recent_message_count:]

        memory_hits: list[MemoryHit] = []
        memory_summary = st.session_state["memory_summary"]
        if memory_store is not None:
            memory_hits = memory_store.search(
                user_prompt,
                limit=memory_limit,
                session_id=st.session_state["session_id"] if session_only_memory else None,
            )
            memory_summary = memory_summary or memory_store.load_summary(st.session_state["session_id"])
        context_messages = build_context_messages(
            st.session_state["system_prompt"],
            memory_summary,
            memory_hits,
            recent_history,
            user_prompt,
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            try:
                assistant_response = st.write_stream(
                    stream_openrouter_response(
                        client=client,
                        messages=context_messages,
                        model=model.strip(),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                )
                if isinstance(assistant_response, str):
                    assistant_text = assistant_response
                elif assistant_response is None:
                    assistant_text = ""
                else:
                    assistant_text = "".join(str(part) for part in assistant_response)
            except OpenAIError as exc:
                st.error(f"OpenRouter request failed: {exc}")
                st.stop()

        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
        turn_index = len(st.session_state["messages"]) - 1
        if memory_store is not None:
            memory_store.add_message(st.session_state["session_id"], "user", user_prompt, turn_index - 1)
            memory_store.add_message(st.session_state["session_id"], "assistant", assistant_text, turn_index)

            if user_turn_count == 1 or user_turn_count % 3 == 0:
                try:
                    refreshed_summary = refresh_summary(
                        client=client,
                        model=model.strip(),
                        existing_summary=memory_summary,
                        recent_messages=st.session_state["messages"][-8:],
                    )
                except OpenAIError as exc:
                    st.warning(f"Memory summary update skipped: {exc}")
                else:
                    if refreshed_summary:
                        st.session_state["memory_summary"] = refreshed_summary
                        memory_store.store_summary(st.session_state["session_id"], refreshed_summary)
                        st.session_state["summary_refreshes"] += 1

            with st.expander("Retrieved memory", expanded=False):
                st.markdown(format_memory_hits(memory_hits))


if __name__ == "__main__":
    main()
