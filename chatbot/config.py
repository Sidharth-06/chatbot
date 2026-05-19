import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# App root directory is one level up from this file's directory
APP_DIR = Path(__file__).resolve().parent.parent

def load_app_env() -> None:
    """Load .env from the app directory so Streamlit cwd does not matter."""
    load_dotenv(APP_DIR / ".env")

load_app_env()

APP_TITLE = "ChatBot"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MEMORY_DIR = os.getenv("MEMORY_PERSIST_DIR", ".chat_memory")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Be concise, friendly, and helpful. "
    "Use relevant memory when available to provide better answers."
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

def api_key_is_configured(api_key: str) -> bool:
    cleaned = api_key.strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered.startswith("your_") or lowered in {"changeme", "replace_me", "xxx"}:
        return False
    return True
