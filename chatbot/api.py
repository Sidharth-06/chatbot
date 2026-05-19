import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from chatbot.config import secret_or_env
from chatbot.memory import MemoryHit, format_memory_hits

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
    recent_messages: list[dict],
    user_prompt: str,
) -> list[dict]:
    messages: list[dict] = [
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

def generate_openrouter_response(
    client: OpenAI,
    messages: list[dict],
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
        extra_body={"reasoning": {"enabled": True}}
    )
    return response.choices[0].message
