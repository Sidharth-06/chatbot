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

APP_TITLE = "Allen - ChatBot with Memory"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MEMORY_DIR = os.getenv("MEMORY_PERSIST_DIR", ".chat_memory")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful chatbot. Use relevant long-term memory when it helps answer the user. "
    "Prefer concise answers, ask clarifying questions when needed, and do not invent facts from memory."
)


st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")

# Sleek React-style CSS
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
        background: linear-gradient(135deg, #0f172a 0%, #1a2a4a 50%, #0f1729 100%) !important;
      }

      .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1a2a4a 50%, #0f1729 100%) !important;
        color: #e2e8f0;
      }

      .block-container {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex;
        flex-direction: column;
        height: 100vh;
      }

      /* Typography */
      h1, h2, h3 {
        color: #e2e8f0 !important;
      }

      p, span, label {
        color: #cbd5e1 !important;
      }

      /* Header Section */
      [data-testid="stVerticalBlockBorderWrapper"]:first-child {
        background: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important;
        padding: 1.5rem 2rem !important;
        flex-shrink: 0;
        position: relative;
      }

      [data-testid="stVerticalBlockBorderWrapper"]:first-child h1 {
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 24px !important;
        font-weight: 700 !important;
        margin: 0 0 4px 0 !important;
        letter-spacing: -0.5px;
      }

      [data-testid="stVerticalBlockBorderWrapper"]:first-child p {
        color: #94a3b8 !important;
        font-size: 13px !important;
        margin: 0 !important;
      }

      /* Control Sliders and Inputs */
      .stSlider {
        background: transparent !important;
      }

      .stSlider [data-testid="stNumberInput"] {
        display: none;
      }

      .stSlider label {
        font-size: 13px !important;
        color: #94a3b8 !important;
      }

      .stSlider > div > div > div:nth-child(2) > div {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
      }

      input[type="range"] {
        width: 100% !important;
        accent-color: #60a5fa !important;
      }

      .stCheckbox label {
        font-size: 13px !important;
        color: #94a3b8 !important;
      }

      input[type="checkbox"] {
        accent-color: #60a5fa !important;
      }

      /* Columns for controls */
      [data-testid="column"] {
        background: transparent !important;
      }

      /* Chat Messages */
      [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
      }

      [data-testid="stChatMessage"] div:first-child {
        display: none;
      }

      [data-testid="stChatMessageContent"] {
        padding: 0 !important;
        background: transparent !important;
      }

      .stChatMessage--user [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.8), rgba(167, 139, 250, 0.8)) !important;
        color: #f8fafc !important;
        padding: 0.875rem 1.125rem !important;
        border-radius: 12px 4px 12px 12px !important;
        max-width: 70%;
        margin-left: auto;
        font-size: 14px;
        line-height: 1.6;
      }

      .stChatMessage--assistant [data-testid="stChatMessageContent"] {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        padding: 0.875rem 1.125rem !important;
        border-radius: 12px 12px 12px 4px !important;
        max-width: 70%;
        font-size: 14px;
        line-height: 1.6;
      }

      /* Input Area */
      [data-testid="stChatInputContainer"] {
        background: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(12px);
        border-top: 1px solid rgba(148, 163, 184, 0.1) !important;
        padding: 1rem 2rem 1.5rem !important;
        flex-shrink: 0;
      }

      [data-testid="stChatInputContainer"] textarea {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        transition: all 0.2s;
      }

      [data-testid="stChatInputContainer"] textarea:focus {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(96, 165, 250, 0.3) !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.1) !important;
      }

      [data-testid="stChatInputContainer"] button {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.9), rgba(167, 139, 250, 0.9)) !important;
        border: none !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        transition: all 0.2s;
        font-weight: 600;
      }

      [data-testid="stChatInputContainer"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(96, 165, 250, 0.2) !important;
      }

      [data-testid="stChatInputContainer"] button:disabled {
        opacity: 0.4 !important;
        cursor: not-allowed !important;
      }

      /* Buttons */
      .stButton > button {
        background: rgba(148, 163, 184, 0.1) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        color: #cbd5e1 !important;
        border-radius: 8px !important;
        transition: all 0.2s;
        width: 100% !important;
      }

      .stButton > button:hover {
        background: rgba(148, 163, 184, 0.15) !important;
        border-color: rgba(148, 163, 184, 0.25) !important;
      }

      /* Expander */
      .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(148, 163, 184, 0.08) !important;
        border-radius: 8px !important;
        color: #cbd5e1 !important;
      }

      .streamlit-expanderHeader:hover {
        background: rgba(30, 41, 59, 0.6) !important;
      }

      .streamlit-expanderContent {
        background: transparent !important;
        border: none !important;
      }

      /* Text Area */
      .stTextArea textarea {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
      }

      .stTextArea textarea:focus {
        border-color: rgba(96, 165, 250, 0.3) !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.1) !important;
      }

      /* Info/Warning Messages */
      .stInfo, .stWarning, .stError {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(96, 165, 250, 0.15) !important;
        border-radius: 8px !important;
        color: #cbd5e1 !important;
      }

      /* Scrollbar */
      ::-webkit-scrollbar {
        width: 6px;
      }

      ::-webkit-scrollbar-track {
        background: transparent;
      }

      ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.2);
        border-radius: 3px;
      }

      ::-webkit-scrollbar-thumb:hover {
        background: rgba(148, 163, 184, 0.4);
      }

      /* Animations */
      @keyframes slideIn {
        from {
          opacity: 0;
          transform: translateY(8px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      [data-testid="stChatMessage"] {
        animation: slideIn 0.3s ease-out;
      }

      /* Responsive */
      @media (max-width: 768px) {
        .stChatMessage--user [data-testid="stChatMessageContent"],
        .stChatMessage--assistant [data-testid="stChatMessageContent"] {
          max-width: 90%;
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

    # Header
    st.title(APP_TITLE)
    st.caption("Chat with OpenRouter models and keep durable memory in a local persistent store.")

    # Sleek Hero layout: large title at left, chat card at right
    left, right = st.columns([1.6, 2.4], gap="large")

    with left:
      st.markdown("""
      <div style="padding:28px 12px 8px 12px">
        <h1 style="font-size:48px; margin-bottom:6px;">
        AI Chatbot UI
        </h1>
        <div style="display:inline-block;padding:8px 14px;border-radius:999px;background:#fff;color:#0f172a;font-weight:600;box-shadow:0 8px 24px rgba(16,24,40,0.45)">
        with loading animation
        </div>
        <p style="color:rgba(226,232,240,0.9); margin-top:18px; max-width:520px">Ask our AI anything. Durable memory and thoughtful replies, configured from environment or Streamlit Secrets.</p>
      </div>
      """, unsafe_allow_html=True)

    # Defaults (kept hidden from UI; configure via env/secrets)
    memory_limit = 5
    recent_turns = 4
    temperature = 0.5
    top_p = 1.0
    max_tokens = 1024
    session_only_memory = False

    # Chat card on the right
    with right:
      st.markdown(
        """
        <div style='background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));padding:22px;border-radius:18px;box-shadow:0 24px 60px rgba(2,6,23,0.6);'>
          <div style='min-height:260px; display:flex; flex-direction:column; justify-content:flex-start;'>
          <div style='color:#94a3b8; font-size:13px; margin-bottom:12px;'>Ask our AI anything</div>
          <div style='flex:1; border-radius:12px; padding:12px; background:rgba(15,23,42,0.35); display:flex; align-items:center; justify-content:center; color:#94a3b8;'>
            <div style='text-align:center; opacity:0.6'>Suggestions and quick actions live here</div>
          </div>
          <div style='margin-top:12px; display:flex; gap:8px;'>
            <button data-suggest='What can I ask you to do?' style='padding:10px 12px;border-radius:10px;border:none;background:#0ea5e9;color:#fff;cursor:pointer'>What can I ask you to do?</button>
            <button data-suggest='Plan a 2 day trip to Thailand' style='padding:10px 12px;border-radius:10px;border:none;background:#a78bfa;color:#fff;cursor:pointer'>Plan a Thailand trip</button>
            <button data-suggest='Give me a 3-step project plan' style='padding:10px 12px;border-radius:10px;border:none;background:#60a5fa;color:#fff;cursor:pointer'>Project plan</button>
          </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
      )

      # Provide a simple text area and send button that feeds the chat logic
      if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

      # Suggestion buttons use JS-less fallback: render as Streamlit buttons beneath card
      sug1, sug2, sug3 = st.columns([1,1,1])
      with sug1:
        if st.button("What can I ask you to do?", key="s1"):
          st.session_state["user_input"] = "What can I ask you to do?"
      with sug2:
        if st.button("Plan a Thailand trip", key="s2"):
          st.session_state["user_input"] = "Plan a Thailand trip for 2 days"
      with sug3:
        if st.button("Project plan", key="s3"):
          st.session_state["user_input"] = "Give me a 3-step project plan"

      st.session_state["user_input"] = st.text_area("", value=st.session_state.get("user_input", ""), placeholder="Ask me anything about your projects or plans...", key="ui_input", height=90)
      if st.button("Send", key="send", use_container_width=True):
        user_prompt = st.session_state.get("user_input", "").strip()
        st.session_state["user_input"] = ""
      else:
        user_prompt = ""

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

    if memory_store is not None:
        st.caption(f"Memory items stored: {memory_store.count():,}")

    if not st.session_state["messages"]:
        st.info("Ask a question below to start chatting.")

    # Chat Container
    render_chat_history(st.session_state["messages"])

    user_prompt = st.chat_input("Ask something... (Shift+Enter for new line)")
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