"""
Converts detected ASL words into natural English sentences.

Tries (in order):
  1. Claude API (Anthropic, model claude-haiku-4-5 by default — cheap & fast)
  2. Offline rule-based fallback (no internet/API needed)

The API key can be supplied via:
  * environment variable  ANTHROPIC_API_KEY 
  * .env file in the project root  (loaded automatically if python-dotenv is
    installed and the file exists)

If no key is configured the offline rules are used — the app still works
end-to-end, just with simpler English.
"""

from __future__ import annotations

import os
from pathlib import Path


try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except Exception:
    pass


def _get_key(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _anthropic_key() -> str:
    return _get_key("ANTHROPIC_API_KEY")


def _openai_key() -> str:
    return _get_key("OPENAI_API_KEY")


# Anthropic models (April 2026 pricing):
#   claude-haiku-4-5      — $1 in / $5 out per MTok      (default — cheap)
#   claude-sonnet-4-5     — $3 in / $15 out per MTok     (smarter, for the bot)
ANTHROPIC_MODEL_FAST  = os.environ.get("ANTHROPIC_MODEL_FAST",  "claude-haiku-4-5")
ANTHROPIC_MODEL_SMART = os.environ.get("ANTHROPIC_MODEL_SMART", "claude-sonnet-4-5")


ASL_PROMPT = (
    "You are an ASL (American Sign Language) to English translator. "
    "ASL drops articles, copulas, reorders words. "
    "Output ONLY the natural English sentence, nothing else."
)


def api_status() -> dict:
    """
    Return what the frontend should show in the API-status indicator.

      {"available": True/False, "provider": "anthropic" | "openai" | None,
       "model": str|None}

    Used by /api/api_status — lets the UI light up green when the key is in
    place (so Alina knows the moment her newly-bought key starts working).
    """
    if _anthropic_key():
        return {
            "available": True,
            "provider": "anthropic",
            "model": ANTHROPIC_MODEL_FAST,
        }
    if _openai_key():
        return {"available": True, "provider": "openai", "model": "gpt-4o-mini"}
    return {"available": False, "provider": None, "model": None}


# --- Offline fallback ------------------------------------------------------

_NEEDS_ARTICLE = {
    "doctor", "nurse", "teacher", "student", "boy", "girl", "man", "woman",
    "bird", "fish", "book", "table", "computer", "pencil", "paper",
    "sister", "brother", "mother", "father", "cousin", "friend",
    "grandmother", "grandfather", "family", "bathroom", "school", "name",
    "day", "night",
}
_FEELINGS = {
    "happy", "sad", "tired", "bored", "hungry", "sick", "hurt", "sorry",
    "fine", "beautiful", "lost",
}
_VERBS = {
    "want", "need", "like", "eat", "drink", "read", "write",
    "draw", "walk", "dance", "play", "sign", "go", "work",
    "learn", "understand", "know", "live", "forget", "help",
    "sit", "finish", "hurt",
}
_VERB_TO_INF = {
    "eat": "to eat", "drink": "to drink", "read": "to read",
    "write": "to write", "draw": "to draw", "walk": "to walk",
    "dance": "to dance", "play": "to play", "sign": "to sign",
    "go": "to go", "work": "to work", "learn": "to learn",
    "sit": "to sit", "help": "to help", "live": "to live",
}


def _offline(words):
    if not words:
        return ""
    words = [w.lower() for w in words]

    if len(words) == 1:
        w = words[0]
        if w in _FEELINGS:
            return f"I am {w}."
        if w == "hello":
            return "Hello!"
        if w == "thanks":
            return "Thank you!"
        if w in ("yes", "no", "please"):
            return w.capitalize() + "."
        return w.capitalize() + "."

    result = []
    has_subject = False
    i = 0

    while i < len(words):
        w = words[i]

        if w == "hello" and i == 0:
            result.append("Hello,")
            i += 1
            continue

        if w in ("what", "where", "when", "who", "how") and not has_subject:
            result.append(w)
            if i + 1 < len(words) and words[i + 1] in _NEEDS_ARTICLE:
                result.append("is the")
                result.append(words[i + 1])
                has_subject = True
                i += 2
                continue
            i += 1
            continue

        if w in _VERBS:
            if not has_subject:
                result.append("I")
                has_subject = True
            if w in ("want", "need") and i + 1 < len(words) and words[i + 1] in _VERB_TO_INF:
                result.append(w)
                result.append(_VERB_TO_INF[words[i + 1]])
                i += 2
                continue
            result.append(w)
            i += 1
            continue

        if not has_subject and w in _FEELINGS:
            result.append("I am")
            has_subject = True
            result.append(w)
            i += 1
            continue

        if w in _FEELINGS and result and not any(
            r.lower() in ("am", "is", "are") for r in result[-2:]
        ):
            if result and result[-1].lower() == "you":
                result.append("are")
            else:
                result.append("is")
            result.append(w)
            i += 1
            continue

        if w in _NEEDS_ARTICLE:
            if not has_subject:
                result.append(f"The {w}")
                has_subject = True
            else:
                result.append(w)
        elif w == "thanks":
            result.append("thank you")
        else:
            result.append(w)

        if w in ("i", "you") or w in _NEEDS_ARTICLE:
            has_subject = True
        i += 1

    sentence = " ".join(result)
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "!", "?")):
        if any(w in words for w in ("what", "where", "when", "who", "how")):
            sentence += "?"
        else:
            sentence += "."
    return sentence


# --- Public API ------------------------------------------------------------

def build_sentence(words):
    """
    Build an English sentence from a list of detected ASL words.
    Returns a dict so the caller can also know which provider answered:
        {"sentence": str, "provider": "anthropic"|"openai"|"offline"}
    """
    if not words:
        return {"sentence": "", "provider": "offline"}

    key = _anthropic_key()
    if key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model=ANTHROPIC_MODEL_FAST,
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{ASL_PROMPT}\n\n"
                            f"ASL signs: {' '.join(words).upper()}"
                        ),
                    }
                ],
            )
            text = msg.content[0].text.strip()
            if text:
                return {"sentence": text, "provider": "anthropic"}
        except Exception as e:
            print(f"[sentence_builder] Anthropic call failed: {e}")

    key = _openai_key()
    if key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                messages=[
                    {"role": "system", "content": ASL_PROMPT},
                    {
                        "role": "user",
                        "content": f"ASL signs: {' '.join(words).upper()}",
                    },
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return {"sentence": text, "provider": "openai"}
        except Exception as e:
            print(f"[sentence_builder] OpenAI call failed: {e}")

    return {"sentence": _offline(words), "provider": "offline"}


# --- Bot (conversational, ASL-aware) --------------------------------------

BOT_SYSTEM_PROMPT = (
    "You are an ASL (American Sign Language) chatbot. The user signs to a "
    "webcam and a recognizer transcribes their signs into individual English "
    "words (provided to you in the user message). "
    "Reply with a SHORT English sentence (max 8 words) using only common "
    "everyday vocabulary, because the response will be turned back into "
    "ASL videos by concatenating clips from a 2000-word dictionary. "
    "Do NOT use fancy or rare words. Avoid punctuation other than . ? ! "
    "Always answer something — do not refuse."
)


def bot_reply(user_words: list[str], history: list[dict] | None = None) -> dict:
    """
    Generate a short English reply suitable for being signed back via
    text_to_sign. Returns:
        {"ok": bool, "reply": str, "provider": str, "error": str|None}
    """
    if not user_words:
        return {
            "ok": False,
            "reply": "",
            "provider": "none",
            "error": "no input",
        }

    key = _anthropic_key()
    if not key:
        return {
            "ok": False,
            "reply": "",
            "provider": "none",
            "error": "Pentru bot ai nevoie de o cheie ANTHROPIC_API_KEY in fisierul .env",
        }

    msgs = []
    for h in (history or [])[-6:]:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    msgs.append(
        {
            "role": "user",
            "content": (
                "Signed words: "
                + " ".join(w.lower() for w in user_words)
            ),
        }
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=ANTHROPIC_MODEL_SMART,
            max_tokens=80,
            system=BOT_SYSTEM_PROMPT,
            messages=msgs,
        )
        text = msg.content[0].text.strip()
        return {"ok": True, "reply": text, "provider": "anthropic", "error": None}
    except Exception as e:
        return {"ok": False, "reply": "", "provider": "anthropic", "error": str(e)}
