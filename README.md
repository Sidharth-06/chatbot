# Chatbot

A Streamlit chatbot that uses the OpenRouter API for chat completions and a persistent local store for long-term memory.

## Features

- Streamlit chat UI with sidebar controls
- OpenRouter OpenAI-compatible API integration
- Persistent vector memory for past messages and summaries
- Session chat history in Streamlit state
- Optional session-only memory lookup




## Setup

```powershell
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set your real `OPENROUTER_API_KEY`. The app loads `.env` from this project folder (next to `app.py`), not from whatever directory you run Streamlit in. You can also use `.streamlit/secrets.toml` instead.

## Run

```powershell
streamlit run app.py
```

## Memory storage

The app stores chat turns and summaries in `.chat_memory/` by default. It keeps raw turns plus a rolling summary so the model can retrieve relevant context later.

## Optional secrets.toml (local)

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your key.

## Deploy to Streamlit Community Cloud

### 1. Push to GitHub

Repository: [github.com/Sidharth-06/chatbot](https://github.com/Sidharth-06/chatbot)

```powershell
cd E:\Souresys\chat_bot
git add .
git commit -m "OpenRouter memory chatbot for Streamlit Cloud"
git push -u origin main
```

Do **not** commit `.env` or `.streamlit/secrets.toml` (they are gitignored). Never commit real API keys.

### 2. Create the app on Streamlit Cloud

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** → repository **`Sidharth-06/chatbot`**, branch **`main`**.
3. Set **Main file path** to:

   `app.py`

4. Open **Advanced settings** → **Secrets** and paste the contents of `chat_bot/.streamlit/secrets.toml.example`, then replace `api_key` with your real OpenRouter key.

5. Click **Deploy**.

### 3. Secrets format (required on Cloud)

```toml
[openrouter]
api_key = "sk-or-v1-your-key-here"
base_url = "https://openrouter.ai/api/v1"
model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

[memory]
persist_dir = ".chat_memory"
```

### Notes for Cloud

- **API key**: Set only in Streamlit Cloud **Secrets**, not in `.env`.
- **Memory**: Vector data on Cloud is **ephemeral** (resets when the app restarts or redeploys). Chat in the current session still works.
- After changing secrets, use **Manage app → Reboot app** in the Cloud dashboard.
