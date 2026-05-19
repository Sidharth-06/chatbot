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
from chatbot.memory import PersistentMemoryStore, MemoryHit

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")
apply_custom_css()

def init_state() -> None:
    st.session_state.setdefault("session_id", uuid4().hex)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("memory_summary", "")
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    st.session_state.setdefault("ui_input", "")

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
    st.caption("with loading animation")

    env_path = APP_DIR / ".env"
    if not api_key_is_configured(api_key):
        if is_streamlit_cloud():
            st.error(
                "OpenRouter API key is missing or still a placeholder. Add it in Streamlit Cloud "
                "Settings → Secrets as either `[openrouter].api_key` or `OPENROUTER_API_KEY`, then reboot the app."
            )
            st.code(
                """[openrouter]\napi_key = "sk-or-..."\nbase_url = "https://openrouter.ai/api/v1"\nmodel = "openai/gpt-4o-mini""",
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

    memory_store = None
    try:
        memory_store = get_memory_store(memory_dir)
    except Exception as exc:
        st.warning(
            "Long-term memory is disabled for this session. "
            f"Chat still works, but past messages will not be stored or retrieved."
        )

    def set_ui_input(value: str) -> None:
        st.session_state["ui_input"] = value

    def clear_chat_state() -> None:
        st.session_state["messages"] = []
        st.session_state["memory_summary"] = ""
        st.session_state["summary_refreshes"] = 0
        st.session_state["ui_input"] = ""

    with st.sidebar:
        st.button("🗑️ Clear chat", use_container_width=True, on_click=clear_chat_state)

    render_chat_history(st.session_state["messages"])

    if not st.session_state["messages"]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button(
                "💡 What can I do?",
                use_container_width=True,
                on_click=set_ui_input,
                args=("What can I ask you to do?",),
            )
        with col2:
            st.button(
                "✈️ Plan a trip",
                use_container_width=True,
                on_click=set_ui_input,
                args=("Plan a 2 day trip to Thailand",),
            )
        with col3:
            st.button(
                "📋 Project plan",
                use_container_width=True,
                on_click=set_ui_input,
                args=("Give me a 3-step project plan",),
            )

    user_prompt = st.chat_input("Ask anything about projects, plans, or code...")
    
    if st.session_state.get("ui_input"):
        user_prompt = st.session_state["ui_input"]
        st.session_state["ui_input"] = ""

    if user_prompt:
      st.session_state["messages"].append({"role": "user", "content": user_prompt})

      memory_hits = []
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

      with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

      with st.chat_message("assistant", avatar="✨"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
              message_obj = generate_openrouter_response(
                client=client, messages=context_messages, model=model.strip(), temperature=temperature, max_tokens=max_tokens, top_p=top_p
              )
              assistant_text = message_obj.content or ""
              placeholder.markdown(assistant_text)
            except OpenAIError as exc:
              st.error(f"OpenRouter request failed: {exc}")
              st.stop()

      assistant_message = {"role": "assistant", "content": assistant_text}
      if 'message_obj' in locals() and hasattr(message_obj, "reasoning_details") and message_obj.reasoning_details:
          assistant_message["reasoning_details"] = message_obj.reasoning_details
          with st.expander("Reasoning Details"):
              st.json(message_obj.reasoning_details)
          
      st.session_state["messages"].append(assistant_message)

      if memory_store is not None:
        user_turn_count = sum(1 for msg in st.session_state["messages"] if msg["role"] == "user")
        memory_store.add_message(st.session_state["session_id"], "user", user_prompt, user_turn_count - 1)
        memory_store.add_message(st.session_state["session_id"], "assistant", assistant_text, user_turn_count)

        if user_turn_count % 3 == 0:
          from chatbot.memory import refresh_summary
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
