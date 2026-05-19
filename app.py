from __future__ import annotations

import os
import sys
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

APP_TITLE = "AI Chatbot UI"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MEMORY_DIR = os.getenv("MEMORY_PERSIST_DIR", ".chat_memory")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Be concise, friendly, and helpful. "
    "Use relevant memory when available to provide better answers."
)


st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")

# Sleek minimal UI with purple/lavender accents
st.markdown(
    """
    <style>
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }

      html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        height: 100%;
        background: linear-gradient(135deg, #f3e7ff 0%, #ede9fe 50%, #faf5ff 100%) !important;
      }

      .stApp {
        background: linear-gradient(135deg, #f3e7ff 0%, #ede9fe 50%, #faf5ff 100%) !important;
        color: #1a1a1a;
      }

      .block-container {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex;
        flex-direction: column;
        height: 100vh;
      }

      /* Header Section */
      [data-testid="stVerticalBlockBorderWrapper"]:first-child {
        background: transparent !important;
        border: none !important;
        padding: 3rem 2rem 1.5rem !important;
        flex-shrink: 0;
      }

      [data-testid="stVerticalBlockBorderWrapper"]:first-child h1 {
        font-size: 48px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        color: #000 !important;
        letter-spacing: -2px;
      }

      [data-testid="stVerticalBlockBorderWrapper"]:first-child p {
        color: #666 !important;
        font-size: 14px !important;
        margin: 8px 0 0 0 !important;
      }

      /* Chat Messages */
      [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0 !important;
        margin: 0.5rem 0 !important;
      }

      [data-testid="stChatMessage"] div:first-child {
        display: none;
      }

      [data-testid="stChatMessageContent"] {
        padding: 0 !important;
        background: transparent !important;
      }

      .stChatMessage--user [data-testid="stChatMessageContent"] {
        background: transparent !important;
        color: #333 !important;
        padding: 0 !important;
        max-width: 100%;
        margin-left: auto;
        font-size: 14px;
        line-height: 1.5;
      }

      .stChatMessage--assistant [data-testid="stChatMessageContent"] {
        background: rgba(168, 85, 247, 0.08) !important;
        color: #333 !important;
        border-left: 3px solid #a855f7 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        max-width: 100%;
        font-size: 14px;
        line-height: 1.6;
      }

      /* Chat Container */
      .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 2rem;
      }

      /* Input Area (hide Streamlit default chat input; we provide a custom input) */
      [data-testid="stChatInputContainer"] {
        display: none !important;
      }

      [data-testid="stChatInputContainer"] textarea {
        background: white !important;
        border: 1px solid #e9d5ff !important;
        color: #333 !important;
        border-radius: 24px !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        transition: all 0.2s;
      }

      [data-testid="stChatInputContainer"] textarea::placeholder {
        color: #999 !important;
      }

      [data-testid="stChatInputContainer"] textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.1) !important;
      }

      [data-testid="stChatInputContainer"] button {
        background: transparent !important;
        border: none !important;
        color: #a855f7 !important;
        font-size: 20px !important;
        padding: 0 12px !important;
        transition: all 0.2s;
      }

      [data-testid="stChatInputContainer"] button:hover {
        transform: scale(1.1);
        color: #9333ea !important;
      }

      /* Scrollbar */
      ::-webkit-scrollbar {
        width: 6px;
      }

      ::-webkit-scrollbar-track {
        background: transparent;
      }

      ::-webkit-scrollbar-thumb {
        background: rgba(168, 85, 247, 0.2);
        border-radius: 3px;
      }

      ::-webkit-scrollbar-thumb:hover {
        background: rgba(168, 85, 247, 0.4);
      }

      /* Info/Warning Messages */
      .stInfo, .stWarning, .stError {
        background: rgba(168, 85, 247, 0.08) !important;
        border: 1px solid #e9d5ff !important;
        border-radius: 8px !important;
        color: #333 !important;
      }

      /* Remove extra padding */
      .stMarkdown {
        padding: 0 !important;
      }

      @media (max-width: 768px) {
        [data-testid="stVerticalBlockBorderWrapper"]:first-child h1 {
          font-size: 32px !important;
        }
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


def init_state() -> None:
    st.session_state.setdefault("session_id", uuid4().hex)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("memory_summary", "")
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
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0,
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

    # Header
    st.title(APP_TITLE)
    st.caption("with loading animation")

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
model = "openai/gpt-4o-mini""",
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
            f"Chat still works, but past messages will not be stored or retrieved."
        )

    # Chat Container
    render_chat_history(st.session_state["messages"])

    # Replace the standard `st.chat_input` with a styled text area + suggestion buttons
    if "ui_input" not in st.session_state:
      st.session_state["ui_input"] = ""

    left_col, right_col = st.columns([3, 1], gap="small")

    with left_col:
      st.session_state["ui_input"] = st.text_area(
        "",
        value=st.session_state.get("ui_input", ""),
        placeholder="Ask our AI anything about projects, plans, or code...",
        height=96,
        key="ui_text",
      )

      send = st.button("Send", key="send", use_container_width=True)
      clear = st.button("Clear chat", key="clear", use_container_width=True)

    with right_col:
      if st.button("What can I ask you?", key="s1_small"):
        st.session_state["ui_input"] = "What can I ask you to do?"
      if st.button("Plan a trip", key="s2_small"):
        st.session_state["ui_input"] = "Plan a 2 day trip to Thailand"
      if st.button("Project plan", key="s3_small"):
        st.session_state["ui_input"] = "Give me a 3-step project plan"

    if clear:
      st.session_state["messages"] = []
      st.session_state["memory_summary"] = ""
      st.session_state["summary_refreshes"] = 0
      st.session_state["ui_input"] = ""
      st.experimental_rerun()

    user_prompt = st.session_state.get("ui_input", "").strip() if send else ""
    if user_prompt:
      # Clear the input immediately for a snappy UX
      st.session_state["ui_input"] = ""

      st.session_state["messages"].append({"role": "user", "content": user_prompt})

      memory_hits: list[MemoryHit] = []
      memory_summary = st.session_state["memory_summary"]
      if memory_store is not None:
        memory_hits = memory_store.search(
          user_prompt,
          limit=5,
          session_id=st.session_state["session_id"],
        )
        memory_summary = memory_summary or memory_store.load_summary(st.session_state["session_id"])

      context_messages = build_context_messages(
        st.session_state["system_prompt"],
        memory_summary,
        memory_hits,
        st.session_state["messages"][:-1][-8:],
        user_prompt,
      )

      # Render the user message (already appended to history) and stream the assistant response
      with st.chat_message("user"):
        st.markdown(user_prompt)

      with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_text = ""
        try:
          for delta in stream_openrouter_response(
            client=client, messages=context_messages, model=model.strip(), temperature=temperature, max_tokens=max_tokens, top_p=top_p
          ):
            assistant_text += str(delta)
            # progressively update assistant content
            placeholder.markdown(assistant_text)
        except OpenAIError as exc:
          st.error(f"OpenRouter request failed: {exc}")
          st.stop()

      # Save the assistant's final text into the session history
      st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

      if memory_store is not None:
        user_turn_count = sum(1 for msg in st.session_state["messages"] if msg["role"] == "user")
        memory_store.add_message(st.session_state["session_id"], "user", user_prompt, user_turn_count - 1)
        memory_store.add_message(st.session_state["session_id"], "assistant", assistant_text, user_turn_count)

        if user_turn_count % 3 == 0:
          try:
            refreshed_summary = refresh_summary(
              client=client,
              model=model.strip(),
              existing_summary=memory_summary,
              recent_messages=st.session_state["messages"][-8:],
            )
            if refreshed_summary:
              st.session_state["memory_summary"] = refreshed_summary
              memory_store.store_summary(st.session_state["session_id"], refreshed_summary)
          except OpenAIError:
            pass


if __name__ == "__main__":
    main()