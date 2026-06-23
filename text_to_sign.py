"""
Text -> ASL video.

Takes an English sentence, looks up each word in the WLASL dataset
and concatenates the corresponding clips into a single MP4.

Uses ffmpeg's `concat` FILTER (not the concat demuxer) so that clips with
different resolutions, framerates, or codecs are all correctly normalized
and joined into one output. The older demuxer approach silently drops
clips whose parameters don't match the first input.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from config import (
    GENERATED_VIDEOS_DIR,
    WLASL_JSON,
    WLASL_VIDEOS_DIR,
)

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = "ffmpeg"


# --- Load WLASL index (once, on first use) --------------------------

_WORD_TO_VIDEOS: dict[str, list[str]] = {}
_ALL_WORDS: list[str] = []


def _load_index() -> None:
    global _WORD_TO_VIDEOS, _ALL_WORDS
    if _WORD_TO_VIDEOS:
        return
    if not WLASL_JSON.exists():
        print(f"[text_to_sign] WLASL json not found at {WLASL_JSON}")
        return
    if not WLASL_VIDEOS_DIR.exists():
        print(f"[text_to_sign] videos dir not found at {WLASL_VIDEOS_DIR}")
        return

    available = set(os.listdir(WLASL_VIDEOS_DIR))
    with open(WLASL_JSON, encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        gloss = entry.get("gloss", "").strip().lower()
        if not gloss:
            continue
        local = [
            f"{inst.get('video_id', '')}.mp4"
            for inst in entry.get("instances", [])
            if f"{inst.get('video_id', '')}.mp4" in available
        ]
        if local:
            _WORD_TO_VIDEOS[gloss] = local

    _ALL_WORDS = sorted(_WORD_TO_VIDEOS.keys())
    print(f"[text_to_sign] Loaded {len(_WORD_TO_VIDEOS)} signed words with local video.")


# --- Tokenization ---------------------------------------------------

# ASL generally omits articles, copulas, and some prepositions; skip them.
_SKIP = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "do", "does", "did", "to", "of", "at", "in", "on", "for", "with",
    "and", "but", "or", "so", "this", "that", "these", "those",
}


def _candidates(word: str) -> list[str]:
    """Light stemming: try plural/tense variants to improve dictionary hit rate."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return []
    cand = [w]
    if w.endswith("ies") and len(w) > 4:
        cand.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        cand.append(w[:-2])
    if w.endswith("s") and len(w) > 2:
        cand.append(w[:-1])
    if w.endswith("ed") and len(w) > 3:
        cand.append(w[:-2])
        cand.append(w[:-1])
    if w.endswith("ing") and len(w) > 4:
        cand.append(w[:-3])
        cand.append(w[:-3] + "e")
    seen = set()
    out = []
    for c in cand:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def tokenize(sentence: str) -> list[str]:
    words = re.findall(r"[A-Za-z]+", sentence)
    return [w for w in words if w.lower() not in _SKIP]


def resolve_word(word: str) -> Optional[str]:
    _load_index()
    for cand in _candidates(word):
        if cand in _WORD_TO_VIDEOS:
            return _WORD_TO_VIDEOS[cand][0]
    return None


def plan_sentence(sentence: str) -> dict:
    _load_index()
    found: list[dict] = []
    missing: list[str] = []
    tokens = tokenize(sentence)
    for t in tokens:
        vid = resolve_word(t)
        if vid is not None:
            found.append({"word": t.lower(), "video": vid})
        else:
            missing.append(t.lower())
    return {"found": found, "missing": missing, "tokens": [t.lower() for t in tokens]}


# --- Video concatenation -------------------------------------------

def _sentence_hash(videos: list[str]) -> str:
    raw = "|".join(videos).encode()
    return hashlib.md5(raw).hexdigest()[:12]


def _ffmpeg_concat(input_videos: list[Path], output: Path) -> bool:
    """
    Concatenate videos using ffmpeg's concat FILTER.

    Each input is individually:
      - scaled to fit inside 480x480 preserving aspect ratio
      - padded to exactly 480x480 with black bars
      - framerate forced to 25 fps, SAR forced to 1:1

    Then all normalized streams are concatenated. This is robust against
    source clips with different resolutions/framerates/codecs, unlike the
    concat demuxer which silently drops mismatched clips.
    """
    if not input_videos:
        return False

    # Build -i flags for each input
    input_args: list[str] = []
    for v in input_videos:
        input_args += ["-i", str(v)]

    # Build filter_complex: per-input normalize, then concat
    w, h, fps = 480, 480, 25
    per_input = []
    for i in range(len(input_videos)):
        per_input.append(
            f"[{i}:v:0]"
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p[v{i}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(len(input_videos)))
    filter_complex = (
        ";".join(per_input)
        + f";{concat_inputs}concat=n={len(input_videos)}:v=1:a=0[out]"
    )

    cmd = [
        FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output.exists() and output.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        print(f"[text_to_sign] ffmpeg error: {e.stderr.decode(errors='ignore')[:500]}")
        return False


def build_video(sentence: str) -> dict:
    """Build a concatenated video for `sentence`. Cached by content hash."""
    _load_index()
    plan = plan_sentence(sentence)
    found = plan["found"]

    if not found:
        return {"ok": False, "video_url": None, **plan}

    videos = [item["video"] for item in found]
    sig = _sentence_hash(videos)
    output = GENERATED_VIDEOS_DIR / f"{sig}.mp4"

    if output.exists() and output.stat().st_size > 0:
        return {
            "ok": True,
            "video_url": f"/static/generated_signs/{output.name}",
            **plan,
        }

    input_paths = [WLASL_VIDEOS_DIR / v for v in videos]
    ok = _ffmpeg_concat(input_paths, output)
    if not ok:
        return {"ok": False, "video_url": None, **plan}

    return {
        "ok": True,
        "video_url": f"/static/generated_signs/{output.name}",
        **plan,
    }


def all_words() -> list[str]:
    _load_index()
    return list(_ALL_WORDS)


if __name__ == "__main__":
    import time
    _load_index()
    print(f"Loaded {len(_ALL_WORDS)} words.")
    test = "hello how are you"
    t = time.time()
    r = build_video(test)
    print(f"Time: {time.time()-t:.1f}s")
    print(f"OK: {r['ok']}, URL: {r['video_url']}")
    print(f"Found: {[x['word'] for x in r['found']]}")
