"""
Telegram Bot API: send a message with the token and chat ID from .env.

Messages use parse_mode HTML; escape any dynamic text with escape(). Error
text from requests contains the bot URL (and so the token): every detail
returned from here goes through security.redact().
"""
import html

import requests

from server import clock, config_store
from server.security import redact

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TIMEOUT_S = 8
NOT_CONFIGURED = "Telegram is not configured: add the bot token and chat ID in Settings."


def configured() -> bool:
    return config_store.secrets_status()["telegram"]


def escape(text) -> str:
    """Escape text for parse_mode HTML."""
    return html.escape(str(text), quote=False)


def send(text: str, parse_mode: str | None = "HTML") -> tuple:
    """Send one message. Returns (ok, detail); detail is redacted."""
    token = config_store.secret("telegram_token")
    chat_id = config_store.secret("telegram_chat_id")
    if not (token and chat_id):
        return False, NOT_CONFIGURED
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        r = requests.post(API_URL.format(token=token), json=payload, timeout=TIMEOUT_S)
        try:
            body = r.json()
        except ValueError:
            body = {}
        ok = bool(r.ok and body.get("ok", True))
        detail = "Message sent." if ok else (body.get("description") or f"Telegram answered HTTP {r.status_code}.")
    except requests.RequestException as e:
        ok, detail = False, f"Could not reach Telegram: {type(e).__name__}: {e}"
    return ok, redact(detail)


def send_test() -> tuple:
    now = clock.now_et().strftime("%Y-%m-%d %H:%M ET")
    return send(f"🏀 <b>NBA Quant</b>: connection test ✅\n<i>{escape(now)}</i>")
