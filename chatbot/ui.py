import streamlit as st


def apply_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Base & typography ── */
        html, body, [data-testid="stApp"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            background: #f5f0ff !important;
            color: #1a1a1a !important;
        }

        /* ── Main container ── */
        .block-container {
            max-width: 860px !important;
            padding: 1.5rem 2rem 6rem 2rem !important;
            margin: 0 auto !important;
        }

        /* ── Page title ── */
        h1 {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #1a1a1a !important;
            letter-spacing: -0.5px !important;
            margin-bottom: 0.25rem !important;
        }

        /* Caption / subtitle */
        [data-testid="stCaptionContainer"] p,
        .stCaption p {
            color: #6b7280 !important;
            font-size: 0.875rem !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #ede9fe !important;
        }
        [data-testid="stSidebar"] * {
            color: #1a1a1a !important;
        }

        /* ── Chat message wrappers ── */
        [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0.5rem 0 !important;
            margin: 0.5rem 0 !important;
        }

        /* ── User bubble ── */
        [data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"],
        .stChatMessage--user [data-testid="stChatMessageContent"] {
            background: #ffffff !important;
            border: 1px solid #e9d5ff !important;
            border-radius: 14px 14px 4px 14px !important;
            padding: 0.75rem 1rem !important;
            box-shadow: 0 2px 8px rgba(168, 85, 247, 0.08) !important;
        }

        /* ── Assistant bubble ── */
        .stChatMessage--assistant [data-testid="stChatMessageContent"] {
            background: #ffffff !important;
            border: 1px solid #ede9fe !important;
            border-left: 4px solid #a855f7 !important;
            border-radius: 4px 14px 14px 14px !important;
            padding: 0.75rem 1rem !important;
            box-shadow: 0 2px 8px rgba(168, 85, 247, 0.05) !important;
        }

        /* ── Force ALL text inside chat bubbles dark ── */
        [data-testid="stChatMessageContent"],
        [data-testid="stChatMessageContent"] *,
        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] span,
        [data-testid="stChatMessageContent"] li,
        [data-testid="stChatMessageContent"] code,
        [data-testid="stChatMessageContent"] pre,
        [data-testid="stChatMessageContent"] a,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] *,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span {
            color: #1a1a1a !important;
            font-size: 15px !important;
            line-height: 1.65 !important;
        }

        /* ── Code blocks inside chat ── */
        [data-testid="stChatMessageContent"] pre,
        [data-testid="stChatMessageContent"] code {
            background: #f3f0ff !important;
            border: 1px solid #e9d5ff !important;
            border-radius: 6px !important;
            color: #6d28d9 !important;
            font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
            font-size: 13px !important;
        }

        /* ── Avatar icons ── */
        [data-testid="stChatMessageAvatar"] {
            background: transparent !important;
            border: none !important;
        }

        /* ── Chat input bar (bottom) ── */
        [data-testid="stChatInput"] {
            border-top: 1px solid #e9d5ff !important;
            background: #ffffff !important;
            padding: 0.75rem 1rem !important;
        }

        [data-testid="stChatInput"] textarea {
            background: #fafafa !important;
            border: 1.5px solid #e9d5ff !important;
            border-radius: 12px !important;
            color: #1a1a1a !important;
            font-size: 15px !important;
            padding: 10px 14px !important;
            font-family: 'Inter', sans-serif !important;
            box-shadow: none !important;
            transition: border-color 0.2s ease !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #9ca3af !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            border-color: #a855f7 !important;
            box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.1) !important;
            outline: none !important;
        }

        /* ── Send button ── */
        [data-testid="stChatInput"] button {
            background: linear-gradient(135deg, #a855f7, #7c3aed) !important;
            border: none !important;
            border-radius: 8px !important;
            color: white !important;
        }

        /* ── Preset / action buttons ── */
        .stButton button {
            background: #ffffff !important;
            border: 1.5px solid #e9d5ff !important;
            border-radius: 10px !important;
            color: #7c3aed !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 4px rgba(168, 85, 247, 0.08) !important;
        }

        .stButton button:hover {
            background: #f3e8ff !important;
            border-color: #a855f7 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(168, 85, 247, 0.15) !important;
        }

        /* ── Sidebar clear-chat button ── */
        [data-testid="stSidebar"] .stButton button {
            background: #f3e8ff !important;
            color: #7c3aed !important;
            border-color: #d8b4fe !important;
        }

        [data-testid="stSidebar"] .stButton button:hover {
            background: #ede9fe !important;
            border-color: #a855f7 !important;
        }

        /* ── Spinner ── */
        [data-testid="stSpinner"] p {
            color: #7c3aed !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #d8b4fe; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #a855f7; }

        /* ── Horizontal rule ── */
        hr {
            border-color: #e9d5ff !important;
            margin: 1rem 0 !important;
        }

        /* ── Alerts / info boxes ── */
        [data-testid="stAlert"] {
            background: #fdf4ff !important;
            border: 1px solid #e9d5ff !important;
            border-radius: 10px !important;
            color: #1a1a1a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history(messages: list[dict]):
    for message in messages:
        avatar = "👤" if message.get("role") == "user" else "✨"
        with st.chat_message(message.get("role", "assistant"), avatar=avatar):
            st.markdown(message.get("content", ""))
            if message.get("reasoning_details"):
                with st.expander("🧠 Reasoning", expanded=False):
                    st.json(message["reasoning_details"])
