"""
Open: http://localhost:8000
"""

import os
import sys
# Make local imports (config, recognizers, text_to_sign) work regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64
import json
import time
import traceback
import numpy as np
import cv2
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# --- App setup ---
from config import WLASL_VIDEOS_DIR

app = FastAPI(title="ASL Recognition")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve raw WLASL reference clips (read-only) so the teach panel can show
# the user how a sign is supposed to be performed before they record their
# own version.
if WLASL_VIDEOS_DIR.exists():
    app.mount(
        "/wlasl_videos",
        StaticFiles(directory=str(WLASL_VIDEOS_DIR)),
        name="wlasl_videos",
    )

# --- Recognizer (loaded on first use) ---
_recognizer = None


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from recognizers.smart import SmartRecognizer
        _recognizer = SmartRecognizer()
    return _recognizer


# --- REST ---

@app.get("/")
async def index():
    # Disable HTML caching so the user always gets the latest version of the
    # page (the page references its own JS/CSS with versioned URLs anyway).
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/signs")
async def get_signs():
    rec = get_recognizer()
    return {
        "model_signs": rec.get_model_signs(),
        "learned_signs": rec.get_learned_signs(),
        "learned_detailed": rec.get_learned_signs_detailed(),
        "blacklist": rec.get_blacklist_info(),
    }


@app.post("/api/blacklist/clear")
async def clear_blacklist():
    rec = get_recognizer()
    rec.clear_blacklist()
    return {"ok": True}


@app.get("/api/export")
async def export_dataset():
    """
    Full JSON dump of train + test datasets (including feature vectors).
    Trigger from UI to download a portable copy of the data, e.g. for
    attaching as appendix to the thesis or sharing with the advisor.
    """
    rec = get_recognizer()
    payload = rec.export_full_json()
    return JSONResponse(
        payload,
        headers={
            "Content-Disposition": 'attachment; filename="asl_dataset.json"',
        },
    )


@app.post("/api/sentence")
async def make_sentence(body: dict):
    """
    Reformulate detected ASL signs into a natural English sentence.
    Returns the sentence and which provider answered (anthropic / openai /
    offline) so the UI can show a small badge.
    """
    words = body.get("words", [])
    if not words:
        return {"sentence": "", "provider": "offline"}
    from sentence_builder import build_sentence
    return build_sentence(words)


@app.get("/api/api_status")
async def api_status():
    """
    Tell the frontend whether a Claude/OpenAI key is configured.
    The UI uses this to light up the "Claude connected" indicator and to
    enable the chatbot tab.
    """
    from sentence_builder import api_status as _status
    return _status()


@app.post("/api/bot/reply")
async def bot_reply(body: dict):
    """
    ASL chatbot: takes the user's signed words (and optional history) and
    returns a short English sentence. The frontend then runs that reply
    through /api/text_to_sign so the user sees the bot answering in ASL.
    Requires ANTHROPIC_API_KEY.
    """
    from sentence_builder import bot_reply as _bot
    user_words = body.get("words", [])
    history = body.get("history", [])
    return _bot(user_words, history)


@app.get("/api/quiz/random")
async def quiz_random(source: str = "learned"):
    """
    Pick a random word for the Quiz mode and return its WLASL reference clip.
    The pool is always the set of signs the user has already taught, because
    those are the only ones the recognizer can predict. The `source`
    parameter is kept for backwards compatibility with older frontends but
    is otherwise ignored.
    """
    import random
    from text_to_sign import resolve_word

    rec = get_recognizer()
    learned = sorted(set(rec.get_learned_signs()))

    pool = sorted({w for w in learned if w})
    if not pool:
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    "Modelul nu cunoaste inca niciun semn. Mergi la tab-ul "
                    "Recunoastere si preda cateva cuvinte intai."
                ),
            },
            status_code=404,
        )

    word = random.choice(pool)
    video = resolve_word(word)
    return {
        "ok": True,
        "word": word.lower(),
        "video_url": f"/wlasl_videos/{video}" if video else None,
        "in_learned": True,
    }


# --- Text -> Sign-language video ---

@app.post("/api/text_to_sign")
async def text_to_sign(body: dict):
    """
    Build a concatenated WLASL video for the given English sentence.

    Request:  {"sentence": "hello how are you"}
    Response: {"ok": bool, "video_url": str|None, "found": [...], "missing": [...]}
    """
    sentence = (body.get("sentence") or "").strip()
    if not sentence:
        return JSONResponse({"ok": False, "error": "empty sentence"}, status_code=400)
    try:
        from text_to_sign import build_video
        result = build_video(sentence)
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/text_to_sign/words")
async def text_to_sign_words():
    """Return all words available in the WLASL local corpus."""
    try:
        from text_to_sign import all_words
        return {"words": all_words()}
    except Exception as e:
        return JSONResponse({"words": [], "error": str(e)}, status_code=500)


@app.get("/api/sign_preview")
async def sign_preview(word: str):
    """
    Return a WLASL reference clip URL for a single word, so the teach panel
    can show the user how the sign is performed before they record their own.
    """
    try:
        from text_to_sign import resolve_word
        video = resolve_word(word)
        if not video:
            return {"found": False, "word": word.lower()}
        return {
            "found": True,
            "word": word.lower(),
            "video_url": f"/wlasl_videos/{video}",
        }
    except Exception as e:
        return JSONResponse({"found": False, "error": str(e)}, status_code=500)


# --- Glove (ESP32 hardware) endpoints ---
#
# ESP32-S3 cu 5 senzori flex + MPU6050 trimite cadre JSON aici. Cadrele sunt
# forward-uite catre SmartRecognizer.feed_glove_frame(), care le foloseste
# atat in Teach Mode (captura paralela de vectori 11D), cat si la clasificare
# (fuziune cu vectorul 126D de la camera). Pastram si _glove_latest pentru
# debug UI / /api/glove/latest.

_glove_latest = {"connected": False, "frame": None}


@app.websocket("/ws/glove")
async def ws_glove(ws: WebSocket):
    await ws.accept()
    _glove_latest["connected"] = True
    recognizer = get_recognizer()
    print("[WS-Glove] Glove connected from", ws.client.host if ws.client else "?")
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")
            if mtype == "hello":
                print(f"[WS-Glove] Hello: {msg}")
                continue
            if mtype == "glove_frame":
                _glove_latest["frame"] = msg
                # Forward catre recognizer pentru predictie / capturare Teach.
                # Apel ieftin (~50 us): extract_glove_vector + memcpy.
                recognizer.feed_glove_frame(msg)

                # Log in terminal cu predictia DOAR-din-manusa —
                # Afisam ora curenta si DOAR cand se schimba semnul. Clasificam la ~5 Hz.
                _t_now = time.time()
                if _t_now - globals().get("_glove_log_t", 0.0) >= 0.2:
                    globals()["_glove_log_t"] = _t_now
                    gp = recognizer.predict_glove()
                    label_now = gp["label"] if gp else None
                    if label_now != globals().get("_glove_log_last"):
                        globals()["_glove_log_last"] = label_now
                        ts = time.strftime("%H:%M:%S")
                        if gp:
                            print(f"[Glove {ts}] {gp['label']:<14} {gp['confidence']*100:5.1f}%")
                        else:
                            print(f"[Glove {ts}] (niciun semn potrivit)")
    except WebSocketDisconnect:
        print("[WS-Glove] Glove disconnected")
    except (OSError, ConnectionError) as e:
        # Cadere de retea (ex. WinError 121 - timeout WiFi pe hotspot). Nu e
        # un bug: ESP32 se reconecteaza automat. Afisam scurt, fara traceback.
        print(f"[WS-Glove] Conexiune intrerupta ({type(e).__name__}) — astept reconectarea ESP32")
    except Exception as e:
        print(f"[WS-Glove] Error: {e}")
        traceback.print_exc()
    finally:
        _glove_latest["connected"] = False
        recognizer.glove_disconnect()


@app.get("/api/glove/latest")
async def glove_latest():
    """Ultima măsurătoare primită de la glove (pentru UI live debug)."""
    return _glove_latest


@app.get("/api/glove/state")
async def glove_state():
    """Stare manusa + statistici despre semnele invatate cu glove."""
    return get_recognizer().get_glove_state()


# --- WebSocket (camera-based recognizer) ---

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    rec = get_recognizer()
    threshold = 0.35
    last_word = None
    running = False
    print("[WS] Client connected")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            t = msg.get("type")

            if t == "config":
                threshold = msg.get("threshold", threshold)

            elif t == "start":
                running = True
                last_word = None
                rec._frame_buf.clear()
                rec._person_box = None
                await ws.send_json({"type": "status", "running": True})

            elif t == "stop":
                running = False
                await ws.send_json({"type": "status", "running": False})

            elif t == "teach":
                sign = msg.get("sign", "").strip()
                mode = msg.get("mode", "train")
                if mode not in ("train", "test"):
                    mode = "train"
                # use_glove: tri-state (None = auto pe baza configului si a
                # starii manusii, True/False = override explicit din UI).
                use_glove_raw = msg.get("use_glove", None)
                if use_glove_raw is None:
                    use_glove = None
                else:
                    use_glove = bool(use_glove_raw)
                if sign and running:
                    rec.start_teach(sign, mode=mode, use_glove=use_glove)
                    glove_active = rec.teach_status.get("use_glove", False)
                    glove_msg = " + manusa" if glove_active else ""
                    await ws.send_json({
                        "type": "teach",
                        "phase": "countdown",
                        "mode": mode,
                        "use_glove": glove_active,
                        "message": (
                            f"Pregateste-te... (TEST{glove_msg})" if mode == "test"
                            else f"Pregateste-te...{glove_msg}"
                        ),
                    })
                elif not running:
                    await ws.send_json({"type": "teach", "phase": "idle", "message": "Porneste camera intai!"})

            elif t == "undo":
                removed = rec.undo_last()
                await ws.send_json({"type": "undo", "removed": removed})

            elif t == "word_deleted":
                # Clientul sterge un cuvant din lista -> crestem contor si posibil blacklist
                word = msg.get("word", "").strip()
                if word:
                    count, limit, banned = rec.register_delete(word)
                    await ws.send_json({
                        "type": "word_deleted",
                        "word": word,
                        "count": count,
                        "limit": limit,
                        "banned": banned,
                    })

            elif t == "clear_blacklist":
                rec.clear_blacklist()
                await ws.send_json({"type": "blacklist_cleared"})

            elif t == "delete_learned":
                # User explicitly asked to forget a learned sign.
                word = msg.get("word", "").strip()
                if word:
                    removed = rec.delete_learned(word)
                    await ws.send_json({
                        "type": "learned_deleted",
                        "word": word,
                        "removed": removed,
                    })

            elif t == "frame" and running:
                try:
                    # Decode base64 JPEG
                    data = msg.get("data", "")
                    img_bytes = base64.b64decode(data)
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_h, frame_w = frame_rgb.shape[:2]

                    # Process
                    result = rec.process_frame(frame_rgb) or {}

                    # Trimitem intotdeauna statusul mainilor + bbox normalizat
                    hand_detected = bool(result.get("hand_detected"))
                    hand_box = result.get("hand_box")
                    hand_box_norm = None
                    if hand_box:
                        x1, y1, x2, y2 = hand_box
                        hand_box_norm = {
                            "x": x1 / frame_w,
                            "y": y1 / frame_h,
                            "w": (x2 - x1) / frame_w,
                            "h": (y2 - y1) / frame_h,
                        }
                    await ws.send_json({
                        "type": "hand_status",
                        "detected": hand_detected,
                        "box": hand_box_norm,
                    })

                    # Teach status (daca e activ)
                    if rec.is_teaching:
                        status = rec.teach_status
                        await ws.send_json({"type": "teach", **status})
                        continue

                    # Daca tocmai s-a terminat teach (phase=done) -> trimitem mesajul
                    # final de teach (sa se inchida overlay-ul REC) si resetam UI predictie.
                    if rec.teach_status.get("phase") == "done":
                        last_word = None
                        status = rec.teach_status
                        await ws.send_json({"type": "teach", **status})
                        await ws.send_json({"type": "prediction_reset"})
                        # Schimbam phase ca sa nu retrimitem la fiecare frame
                        rec._teach_phase = "idle"

                    # Prediction din orice canal: camera, manusa sau fuziune.
                    # ATENTIE: cu manusa neagra camera NU vede mana, dar
                    # process_frame poate intoarce o predictie doar din senzori
                    # (result["channel"] == "glove"). De aceea NU mai iesim doar
                    # pentru ca hand_detected e False — verificam intai daca
                    # exista o predictie valida din vreun canal.
                    # Daca nu sunt maini nu afisam nimic nou (resetam last_word)
                    if not hand_detected:
                        last_word = None
                        continue

                    # Prediction
                    pred = result.get("prediction")
                    if pred and pred.get("confidence", 0) > 0:
                        label = pred["label"]
                        conf = pred["confidence"]
                        source = pred.get("source", "model")

                        # Ask the recognizer whether the user has held this
                        # sign long enough (and the cooldown has elapsed).
                        accept = rec.try_accept_word(label, conf, threshold)

                        await ws.send_json({
                            "type": "prediction",
                            "label": label,
                            "confidence": conf,
                            "source": source,
                            "hold_progress": accept.get("progress", 0.0),
                            "hold_reason": accept.get("reason", ""),
                        })

                        if accept.get("accepted") and label != last_word:
                            last_word = label
                            await ws.send_json({
                                "type": "word_accepted",
                                "label": label,
                                "confidence": conf,
                            })

                except Exception as e:
                    print(f"[Frame Error] {e}")
                    traceback.print_exc()
                    continue

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS Error] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    print("Starting ASL Recognition Server...")
    print(f"Open http://localhost:{PORT} in your browser")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
