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
DEFAULT_MEMORY_LIMIT = int(os.getenv("MEMORY_LIMIT", "5"))
DEFAULT_RECENT_TURNS = int(os.getenv("RECENT_TURNS", "4"))
DEFAULT_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.5"))
DEFAULT_TOP_P = float(os.getenv("OPENROUTER_TOP_P", "1.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "1024"))
DEFAULT_SESSION_ONLY_MEMORY = os.getenv("MEMORY_SESSION_ONLY", "false").lower() in {"1", "true", "yes"}
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful chatbot. Use relevant long-term memory when it helps answer the user. "
    "Prefer concise answers, ask clarifying questions when needed, and do not invent facts from memory."
)
DEFAULT_SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


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
                padding-top: 1.4rem !important;
                max-width: 1040px;
            }
            [data-testid="stChatMessage"] {
                margin-bottom: 0.35rem;
            }
            [data-testid="stChatMessage"] > div {
                border-radius: 22px;
                background: rgba(15, 23, 42, 0.68);
                border: 1px solid rgba(148, 163, 184, 0.12);
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
            }
            [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
                font-size: 1.02rem;
                line-height: 1.65;
            }
            div[data-testid="stChatInput"] {
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 24px;
                padding: 0.15rem;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.22);
            }
            div[data-testid="stChatInput"] textarea {
                color: #eef4ff !important;
            }
            .stButton button {
                border-radius: 999px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                background: rgba(15, 23, 42, 0.72);
                color: #eef4ff;
            }
            .stButton button:hover {
                border-color: rgba(56, 189, 248, 0.45);
                color: #ffffff;
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


def is_streamlit_cloud() -> bool:
    return os.getenv("STREAMLIT_CLOUD", "").lower() in {"1", "true", "yes"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    memory_limit = env_int("MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT)
    recent_turns = env_int("RECENT_TURNS", DEFAULT_RECENT_TURNS)
    temperature = env_float("OPENROUTER_TEMPERATURE", DEFAULT_TEMPERATURE)
    top_p = env_float("OPENROUTER_TOP_P", DEFAULT_TOP_P)
    max_tokens = env_int("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    session_only_memory = env_bool("MEMORY_SESSION_ONLY", DEFAULT_SESSION_ONLY_MEMORY)

    st.title(APP_TITLE)
    st.caption("A clean OpenRouter chat interface with durable memory, configured from environment or Streamlit Secrets.")

    header_cols = st.columns([7, 1])
    with header_cols[1]:
        if st.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["memory_summary"] = ""
            st.session_state["summary_refreshes"] = 0
            st.rerun()

    env_path = APP_DIR / ".env"
    if not api_key_is_configured(api_key):
        if is_streamlit_cloud():
            st.error(
                "OpenRouter API key is missing or still a placeholder. Add it in Streamlit Cloud "
                "Settings → Secrets as either `[openrouter].api_key` or `OPENROUTER_API_KEY`, then reboot the app."
            )
            st.code(
                """[openrouter]
api_key = "sk-or-..."
base_url = "https://openrouter.ai/api/v1"
model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free""",
                language="toml",
            )
        else:
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


if __name__ == "__main__":
    main()
