"""Thin wrapper around ollama.chat: caching, JSON retry, friendly errors."""
import hashlib
import json
import sys

import ollama

import config


def _die(msg):
    sys.exit(f"ERROR: {msg}")


def check_ollama(need_vision=False):
    try:
        names = [m.model for m in ollama.list().models]
    except Exception:
        _die("Ollama is not running. Start it (`ollama serve`) and retry.")
    if not any(n == config.LLM_MODEL or n.split(":")[0] == config.LLM_MODEL for n in names):
        _die(f"Model '{config.LLM_MODEL}' not found. Run `ollama pull {config.LLM_MODEL}` or fix LLM_MODEL in config.py.")
    if need_vision:
        caps = getattr(ollama.show(config.LLM_MODEL), "capabilities", None) or []
        if "vision" not in caps:
            _die(f"Model '{config.LLM_MODEL}' does not support images (capabilities: {caps}).")


def chat(messages, json_mode=False, num_ctx=None):
    r = ollama.chat(
        model=config.LLM_MODEL, messages=messages, think=False,
        format="json" if json_mode else None, keep_alive=config.KEEP_ALIVE,
        options={"temperature": 0, "num_ctx": num_ctx or config.NUM_CTX},
    )
    return r.message.content or ""  # ignore any separate thinking field


def key(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"\0")
    return h.hexdigest()


def cached(kind, k, fn):
    """Return JSON-cached result of fn() stored at data/cache/<kind>/<k>.json."""
    path = config.CACHE_DIR / kind / f"{k}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    value = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value, False


def chat_json(messages, retries=1):
    """Call the model expecting JSON; retry once on invalid JSON, else return None."""
    for _ in range(retries + 1):
        try:
            return json.loads(chat(messages, json_mode=True))
        except json.JSONDecodeError:
            continue
    return None
