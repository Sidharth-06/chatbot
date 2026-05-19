import streamlit as st
from uuid import uuid4
from openai import OpenAIError

from chatbot.config import (
    APP_TITLE,
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_MEMORY_DIR,
    DEFAULT_SYSTEM_PROMPT,
    APP_DIR,
    secret_or_env,
    is_streamlit_cloud,
    api_key_is_configured
)
from chatbot.ui import apply_custom_css, render_chat_history
from chatbot.api import get_openrouter_client, build_context_messages, generate_openrouter_response
from chatbot.memory import PersistentMemoryStore, refresh_summary

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")
apply_custom_css()


def init_state() -> None:
    st.session_state.setdefault("session_id", uuid4().hex)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("memory_summary", "")
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    # Use "pending_prompt" — never tied to a widget key, avoids Streamlit conflict
    st.session_state.setdefault("pending_prompt", "")


@st.cache_resource(show_spinner="Loading memory store...")
def get_memory_store(persist_dir: str) -> PersistentMemoryStore:
    return PersistentMemoryStore(persist_dir=persist_dir)


def main() -> None:
    init_state()

    api_key = secret_or_env("openrouter", "api_key", "OPENROUTER_API_KEY")
    base_url = secret_or_env("openrouter", "base_url", "OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    model = secret_or_env("openrouter", "model", "OPENROUTER_MODEL", DEFAULT_MODEL)
    memory_dir = secret_or_env("memory", "persist_dir", "MEMORY_PERSIST_DIR", DEFAULT_MEMORY_DIR)

    temperature = 0.7
    max_tokens = 1024
    top_p = 1.0

    st.title(APP_TITLE)

    env_path = APP_DIR / ".env"
    if not api_key_is_configured(api_key):
        if is_streamlit_cloud():
            st.error(
                "OpenRouter API key is missing. Add it in Streamlit Cloud "
                "Settings → Secrets as `[openrouter].api_key` or `OPENROUTER_API_KEY`, then reboot."
            )
            st.code(
                '[openrouter]\napi_key = "sk-or-..."\nbase_url = "https://openrouter.ai/api/v1"\nmodel = "openai/gpt-4o-mini"',
                language="toml",
            )
        else:
            st.error(
                f"OpenRouter API key is missing. Add `OPENROUTER_API_KEY=sk-or-...` to `{env_path}` and restart."
            )
        st.stop()

    try:
        client = get_openrouter_client(api_key.strip(), base_url.strip(), APP_TITLE)
    except Exception as exc:
        st.error(f"Could not initialise OpenRouter client: {exc}")
        st.stop()

    memory_store = None
    try:
        memory_store = get_memory_store(memory_dir)
    except Exception:
        st.warning("Long-term memory disabled for this session — chat still works.")

    # ── Callbacks ────────────────────────────────────────────────────────────
    def set_pending(value: str) -> None:
        """Called by preset buttons (on_click). No widget key involved."""
        st.session_state["pending_prompt"] = value

    def clear_chat() -> None:
        st.session_state["messages"] = []
        st.session_state["memory_summary"] = ""
        st.session_state["pending_prompt"] = ""

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.button("🗑️ Clear chat", use_container_width=True, on_click=clear_chat)

    # ── Chat history ──────────────────────────────────────────────────────────
    render_chat_history(st.session_state["messages"])

    # ── Preset quick-action buttons (only on empty chat) ─────────────────────
    if not st.session_state["messages"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("💡 What can I do?", use_container_width=True,
                      on_click=set_pending, args=("What can I ask you to do?",))
        with c2:
            st.button("✈️ Plan a trip", use_container_width=True,
                      on_click=set_pending, args=("Plan a 2 day trip to Thailand",))
        with c3:
            st.button("📋 Project plan", use_container_width=True,
                      on_click=set_pending, args=("Give me a 3-step project plan",))

    # ── Input resolution: native input wins; preset fills if nothing typed ────
    typed_prompt = st.chat_input("Ask anything about projects, plans, or code...")

    if typed_prompt:
        user_prompt = typed_prompt
        st.session_state["pending_prompt"] = ""          # clear any stale preset
    elif st.session_state["pending_prompt"]:
        user_prompt = st.session_state["pending_prompt"]
        st.session_state["pending_prompt"] = ""
    else:
        user_prompt = ""

    if not user_prompt:
        return

    # ── Append user turn ──────────────────────────────────────────────────────
    st.session_state["messages"].append({"role": "user", "content": user_prompt})

    # ── Memory retrieval ──────────────────────────────────────────────────────
    memory_hits = []
    memory_summary = st.session_state["memory_summary"]
    if memory_store is not None:
        memory_hits = memory_store.search(
            user_prompt, limit=5, session_id=st.session_state["session_id"]
        )
        memory_summary = memory_summary or memory_store.load_summary(st.session_state["session_id"])

    context_messages = build_context_messages(
        st.session_state["system_prompt"],
        memory_summary,
        memory_hits,
        st.session_state["messages"][:-1][-8:],
        user_prompt,
    )

    # ── Render user bubble ────────────────────────────────────────────────────
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # ── Call model & render assistant bubble ──────────────────────────────────
    assistant_text = ""
    reasoning_details = None

    with st.chat_message("assistant", avatar="✨"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                message_obj = generate_openrouter_response(
                    client=client,
                    messages=context_messages,
                    model=model.strip(),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                assistant_text = message_obj.content or ""
                if hasattr(message_obj, "reasoning_details"):
                    reasoning_details = message_obj.reasoning_details
            except OpenAIError as exc:
                placeholder.error(f"OpenRouter request failed: {exc}")
                return
            except Exception as exc:
                placeholder.error(f"Unexpected error: {exc}")
                return

        placeholder.markdown(assistant_text)

        if reasoning_details:
            with st.expander("🧠 Reasoning", expanded=False):
                st.json(reasoning_details)

    # ── Persist assistant turn ────────────────────────────────────────────────
    assistant_message: dict = {"role": "assistant", "content": assistant_text}
    if reasoning_details:
        assistant_message["reasoning_details"] = reasoning_details
    st.session_state["messages"].append(assistant_message)

    # ── Memory storage & periodic summarisation ───────────────────────────────
    if memory_store is not None:
        user_turn_count = sum(1 for m in st.session_state["messages"] if m["role"] == "user")
        memory_store.add_message(st.session_state["session_id"], "user", user_prompt, user_turn_count - 1)
        memory_store.add_message(st.session_state["session_id"], "assistant", assistant_text, user_turn_count)

        if user_turn_count % 3 == 0:
            try:
                refreshed = refresh_summary(
                    client=client,
                    model=model.strip(),
                    existing_summary=memory_summary,
                    recent_messages=st.session_state["messages"][-8:],
                )
                if refreshed:
                    st.session_state["memory_summary"] = refreshed
                    memory_store.store_summary(st.session_state["session_id"], refreshed)
            except OpenAIError:
                pass


if __name__ == "__main__":
    main()
