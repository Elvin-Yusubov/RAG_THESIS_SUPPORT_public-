# Security policy

## Secrets

- Never commit `.env`. It is git-ignored; use `.env.example` as the template.
- **If a key was ever committed, treat it as compromised and rotate it**, even after the file is deleted: Git history keeps it. This applies to the DeepSeek, OpenRouter, Pinecone, Tavily, Telegram, LangSmith and any other provider keys.
- Remove secrets from history (for example with `git filter-repo`) only in addition to rotation, never instead of it.

## Access control

- Web app: configure per-user accounts in `APP_USERS` (salted PBKDF2 hashes). Do not expose the app without it.
- Telegram bot: set `TELEGRAM_ALLOWED_USER_IDS`. An empty list means open access.
- Both interfaces are rate limited per user (`RATE_LIMIT_*`).

## Data handling

Uploaded student documents are processed in memory or in temporary files that are deleted immediately, and are sent to the configured LLM and vision providers. They are never added to the shared index. Telegram conversation text is stored in the local SQLite database until `/reset`, limited to the latest 10 exchanges per chat.

## Reporting a vulnerability

Open a private security advisory on GitHub or contact the maintainer directly. Please do not file public issues for vulnerabilities.
