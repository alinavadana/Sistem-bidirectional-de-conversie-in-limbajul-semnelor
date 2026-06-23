"""
SmartRecognizer - teach-only ASL recognizer.

- MediaPipe Hands detects 1-2 hands, extracts 21 landmarks each.
- Each sign is stored as a set of normalized landmark vectors (wrist-origin,
  scale-invariant). Recognition = KNN with cosine distance.
- Teach flow: 2s countdown -> 3s capture, ~40-50 feature samples saved per sign.
- Blacklist: if the user deletes a word 5 times it is never predicted again.
"""

from __future__ import annotations

import json
import pickle
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise RuntimeError("mediapipe is required. pip install mediapipe")

# --- Paths & tunables come from the project's config.py ------------

from config import (
    LEARNED_SIGNS_PATH as LEARNED_PATH,
    TEST_SIGNS_PATH as TEST_PATH,
    BLACKLIST_PATH,
    DATA_DIR,
    HAND_FEAT_DIM,
    FEAT_DIM,
    GLOVE_FEAT_DIM,
    GLOVE_SMOOTH_FRAMES,
    KNN_K,
    KNN_MAX_COSINE,
    KNN_MAX_COSINE_GLOVE,
    KNN_MIN_EXAMPLES,
    SMOOTH_N,
    SMOOTH_MATCH,
    BLACKLIST_LIMIT,
    TEACH_COUNTDOWN_SEC,
    TEACH_CAPTURE_SEC,
    WORD_HOLD_FRAMES,
    WORD_HOLD_RATIO,
    WORD_COOLDOWN_SEC,
    HAND_REQUIRED_FRAMES,
    FUSION_WEIGHT_CAMERA,
    FUSION_WEIGHT_GLOVE,
    TEACH_DEFAULT_USE_GLOVE,
)
from .glove_features import extract_glove_vector


# Caile catre fisierele de stocare a vectorilor manusii. 
LEARNED_GLOVE_PATH = DATA_DIR / "learned_signs_glove.pkl"
TEST_GLOVE_PATH = DATA_DIR / "test_signs_glove.pkl"


# --- Helpers --------------------------------------------------------

def _normalize_hand(landmarks) -> np.ndarray:
    """
    Convert 21 MediaPipe hand landmarks -> 63D feature vector.
    Translated to wrist origin, scaled by max distance from wrist.
    Returns zeros if landmarks is None.
    """
    if landmarks is None:
        return np.zeros(HAND_FEAT_DIM, dtype=np.float32)
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
        dtype=np.float32,
    )
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# --- Recognizer -----------------------------------------------------

class SmartRecognizer:
    def __init__(self):
        # MediaPipe Hands detector.
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0,
        )

        # Learned signs (TRAIN). Each entry is {"s": session_id, "f": feat}.
        # session_id starts at 0 for the first teach of a label and grows by 1
        # every time the user re-teaches the same label, so we can do
        # cross-session evaluation later.
        self._learned: dict[str, list[dict]] = {}
        # Held-out TEST signs collected via "Modul test". Same structure.
        # Never used by classification, only by evaluate.py.
        self._test: dict[str, list[dict]] = {}

        #glove
        self._learned_glove: dict[str, list[dict]] = {}
        self._test_glove: dict[str, list[dict]] = {}

        # --- Vectorised KNN index (built from _learned, rebuilt on change) ---
        # Stacking all train features once + pre-normalising rows turns each
        # frame's classification into a single matrix-vector multiplication
        # instead of a Python-level loop. This is what makes the recognizer
        # scale comfortably to 100+ signs.
        self._train_X: Optional[np.ndarray] = None        # (N, FEAT_DIM)
        self._train_X_norm: Optional[np.ndarray] = None   # (N, FEAT_DIM) L2-normalised
        self._train_labels: list[str] = []                # length N

        # glove — index KNN separat pe 11D.
        self._train_glove_X: Optional[np.ndarray] = None
        self._train_glove_X_norm: Optional[np.ndarray] = None
        self._train_glove_labels: list[str] = []

        # Cel mai recent vector de manusa, primit de la /ws/glove. Folosit
        # atat in Teach Mode (sample pentru semnul curent), cat si la
        # clasificare (fuziune cu vectorul camerei). Nullable: cand manusa
        # nu e conectata, sistemul cade pe doar-camera.
        self._glove_current: Optional[np.ndarray] = None
        # Buffer pentru netezirea temporala a vectorului manusii. Pastram
        # ultimele GLOVE_SMOOTH_FRAMES cadre brute; la predictie folosim media
        # lor (reduce zgomotul -> eticheta nu mai "sare"). Teach-ul foloseste
        # in continuare vectorul BRUT (_glove_current), ca sa pastram variatia
        # naturala in exemplele de antrenare.
        self._glove_buffer: deque = deque(maxlen=max(1, GLOVE_SMOOTH_FRAMES))
        # Wall-clock timestamp pentru ultima actualizare a _glove_current.
        # Daca trec >0.5 s fara cadre noi de la manusa, consideram ca
        # manusa s-a deconectat si oprim fuziunea.
        self._glove_last_update_t: float = 0.0
        self._glove_connected: bool = False

        # Delete counters -> blacklist
        self._delete_counts: dict[str, int] = {}
        self._blacklist: set[str] = set()

        # Compatibility shims with server.py (which clears these on start).
        self._frame_buf: deque = deque(maxlen=30)
        self._person_box = None

        # Smoothing
        self._recent: deque = deque(maxlen=SMOOTH_N)

        # Stability buffers for word acceptance.
        # _hold_buf records the last N predicted labels (or None when no
        # confident match) and is used to decide whether the user has held
        # a sign long enough to commit it as a word.
        self._hold_buf: deque = deque(maxlen=WORD_HOLD_FRAMES)
        # Frames in a row where a hand was visible. Resets to 0 when the
        # hand leaves the frame, so a brief glitch doesn't trigger a word.
        self._hand_streak: int = 0
        # Wall-clock of the last accepted word (for cooldown).
        self._last_accept_t: float = 0.0
        self._last_accept_label: Optional[str] = None

        # Teach state
        self._teach_sign: Optional[str] = None
        self._teach_mode: str = "train"   # "train" -> _learned, "test" -> _test
        self._teach_phase: str = "idle"   # idle | countdown | capture | done
        self._teach_t0: float = 0.0
        self._teach_samples: list[np.ndarray] = []
        # Buffer paralel pentru vectorii de manusa captati in timpul Teach
        # Mode. La fiecare cadru de camera in faza "capture", luam si vectorul
        # curent al manusii (daca exista) si il punem aici. La _finalize_teach,
        # daca lista nu e goala, salvam vectorii in _learned_glove[label].
        self._teach_glove_samples: list[np.ndarray] = []
        # Daca utilizatorul a cerut explicit sa se foloseasca si manusa in sesiynea curenta
        # Captura efectiva ramane conditionata de manusa fiind conectata.
        self._teach_use_glove: bool = False
        self._teach_last_sign: Optional[str] = None
        self._teach_last_count: int = 0
        self._teach_last_glove_count: int = 0
        self._teach_last_session: int = 0
        self._teach_last_mode: str = "train"

        self._load_blacklist()
        self._load_learned()
        self._load_test()
        self._load_glove()
        self._load_test_glove()

        n_train_sessions = sum(
            len({e["s"] for e in samples}) for samples in self._learned.values()
        )
        n_test_sessions = sum(
            len({e["s"] for e in samples}) for samples in self._test.values()
        )
        n_glove_signs = len(self._learned_glove)
        n_glove_samples = sum(len(v) for v in self._learned_glove.values())
        print(
            f"[SmartRecognizer] Ready (multimodal). "
            f"{len(self._learned)} train signs ({n_train_sessions} sessions) | "
            f"{n_glove_signs} cu glove ({n_glove_samples} mostre) | "
            f"{len(self._test)} test signs ({n_test_sessions} sessions) | "
            f"{len(self._blacklist)} blacklisted."
        )

    # --- Persistence ---

    @staticmethod
    def _load_pickle_signs(path, expected_dim: int = FEAT_DIM) -> dict[str, list[dict]]:
        """
        Load a sign pickle, normalising legacy entries on the way.

        Legacy format (pre-session tracking):  dict[label, list[np.ndarray]]
        New format:                            dict[label, list[{"s": int, "f": np.ndarray}]]
        Old entries get session_id = 0 so they still count as "session 1"
        for cross-session evaluation.

        expected_dim diferentiaza fisierele camerei (FEAT_DIM = 126) de cele
        ale manusii (GLOVE_FEAT_DIM = 11). Vectorii cu alta dimensiune sunt
        ignorati (sanity check pentru a evita amestecul de fisiere).
        """
        out: dict[str, list[dict]] = {}
        if not path.exists():
            return out
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            for label, entries in data.items():
                converted: list[dict] = []
                for e in entries:
                    if isinstance(e, dict) and "f" in e:
                        feat = np.asarray(e["f"], dtype=np.float32)
                        sid = int(e.get("s", 0))
                    else:
                        feat = np.asarray(e, dtype=np.float32)
                        sid = 0
                    if feat.shape == (expected_dim,):
                        converted.append({"s": sid, "f": feat})
                if converted:
                    out[label] = converted
        except Exception as e:
            print(f"[SmartRecognizer] Could not load {path.name}: {e}")
        return out

    @staticmethod
    def _save_pickle_signs(path, data: dict) -> None:
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"[SmartRecognizer] Could not save {path.name}: {e}")

    def _load_learned(self) -> None:
        self._learned = self._load_pickle_signs(LEARNED_PATH)
        self._rebuild_train_index()
        self._save_metadata_sidecar()

    def _save_learned(self) -> None:
        self._save_pickle_signs(LEARNED_PATH, self._learned)
        self._rebuild_train_index()
        self._save_metadata_sidecar()

    def _load_test(self) -> None:
        self._test = self._load_pickle_signs(TEST_PATH)
        self._save_metadata_sidecar()

    def _save_test(self) -> None:
        self._save_pickle_signs(TEST_PATH, self._test)
        self._save_metadata_sidecar()

    def _load_glove(self) -> None:
        """Incarca vectorii 11D de manusa pentru semnele invatate."""
        self._learned_glove = self._load_pickle_signs(
            LEARNED_GLOVE_PATH, expected_dim=GLOVE_FEAT_DIM
        )
        self._rebuild_glove_index()

    def _save_glove(self) -> None:
        self._save_pickle_signs(LEARNED_GLOVE_PATH, self._learned_glove)
        self._rebuild_glove_index()

    def _load_test_glove(self) -> None:
        self._test_glove = self._load_pickle_signs(
            TEST_GLOVE_PATH, expected_dim=GLOVE_FEAT_DIM
        )

    def _save_test_glove(self) -> None:
        self._save_pickle_signs(TEST_GLOVE_PATH, self._test_glove)

    def _rebuild_train_index(self) -> None:
        """
        Stack every train feature into one (N, FEAT_DIM) matrix and pre-
        normalise the rows. Classification then becomes one matmul per frame
        instead of N python iterations.
        Called every time _learned changes (load, finalize teach, delete).
        """
        feats = []
        labels = []
        for label, entries in self._learned.items():
            if label in self._blacklist:
                continue
            if len(entries) < KNN_MIN_EXAMPLES:
                continue
            for e in entries:
                feats.append(e["f"])
                labels.append(label)
        if not feats:
            self._train_X = None
            self._train_X_norm = None
            self._train_labels = []
            return
        X = np.stack(feats).astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        self._train_X = X
        self._train_X_norm = (X / norms).astype(np.float32)
        self._train_labels = labels

    def _rebuild_glove_index(self) -> None:
        """
        Identic cu _rebuild_train_index, dar pe vectorii 11D ai manusii.
        Acelasi pattern: o singura matrice pre-normalizata, clasificare =
        un matmul. La 100-150 semne x ~30 mostre = ~4500 randuri, 4500x11 e
        nimic pentru numpy (sub 1 ms per frame chiar pe laptop modest).
        """
        feats = []
        labels = []
        for label, entries in self._learned_glove.items():
            if label in self._blacklist:
                continue
            if len(entries) < KNN_MIN_EXAMPLES:
                continue
            for e in entries:
                feats.append(e["f"])
                labels.append(label)
        if not feats:
            self._train_glove_X = None
            self._train_glove_X_norm = None
            self._train_glove_labels = []
            return
        X = np.stack(feats).astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        self._train_glove_X = X
        self._train_glove_X_norm = (X / norms).astype(np.float32)
        self._train_glove_labels = labels

    def _save_metadata_sidecar(self) -> None:
        """
        Write a small, human-readable JSON next to the pickle files showing
        what's stored (label -> sessions/samples). Useful for inspecting the
        dataset in any text editor and as documentation in the thesis.
        Features themselves stay in pickle (binary, fast).
        """
        try:
            def summarise(bucket):
                out = {}
                for label, entries in sorted(bucket.items()):
                    sessions = sorted({int(e["s"]) for e in entries})
                    out[label] = {
                        "samples": len(entries),
                        "sessions": len(sessions),
                        "session_ids": sessions,
                    }
                return out
            meta = {
                "feat_dim": FEAT_DIM,
                "train": summarise(self._learned),
                "test": summarise(self._test),
                "totals": {
                    "train_labels": len(self._learned),
                    "train_samples": sum(len(v) for v in self._learned.values()),
                    "test_labels": len(self._test),
                    "test_samples": sum(len(v) for v in self._test.values()),
                },
            }
            sidecar = LEARNED_PATH.with_suffix(".meta.json")
            sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[SmartRecognizer] Could not save metadata sidecar: {e}")

    def export_full_json(self) -> dict:
        """
        Return a JSON-friendly snapshot of train + test datasets,
        including the actual feature vectors. Use sparingly: large.
        """
        def serialise(bucket):
            return {
                label: [
                    {"session": int(e["s"]), "features": e["f"].tolist()}
                    for e in entries
                ]
                for label, entries in sorted(bucket.items())
            }
        return {
            "feat_dim": FEAT_DIM,
            "train": serialise(self._learned),
            "test": serialise(self._test),
        }

    def _next_session_id(self, label: str, mode: str) -> int:
        """Auto-incremented session id per (mode, label)."""
        bucket = self._learned if mode == "train" else self._test
        existing = bucket.get(label, [])
        if not existing:
            return 0
        return max(int(e.get("s", 0)) for e in existing) + 1

    def _load_blacklist(self) -> None:
        if not BLACKLIST_PATH.exists():
            return
        try:
            data = json.loads(BLACKLIST_PATH.read_text(encoding="utf-8"))
            self._delete_counts = dict(data.get("counts", {}))
            self._blacklist = set(data.get("banned", []))
        except Exception as e:
            print(f"[SmartRecognizer] Could not load blacklist: {e}")

    def _save_blacklist(self) -> None:
        try:
            BLACKLIST_PATH.write_text(
                json.dumps(
                    {
                        "counts": self._delete_counts,
                        "banned": sorted(self._blacklist),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[SmartRecognizer] Could not save blacklist: {e}")

    # --- Public API ---

    def get_model_signs(self) -> list[str]:
        # No pretrained model in teach-only mode.
        return []

    def get_learned_signs(self) -> list[str]:
        return sorted(self._learned.keys())

    def get_learned_signs_detailed(self) -> dict:
        """For UI: per-label sample count + session count, train and test."""
        out = {}
        labels = set(self._learned.keys()) | set(self._test.keys())
        for label in sorted(labels):
            tr = self._learned.get(label, [])
            te = self._test.get(label, [])
            out[label] = {
                "train_samples": len(tr),
                "train_sessions": len({e["s"] for e in tr}),
                "test_samples": len(te),
                "test_sessions": len({e["s"] for e in te}),
            }
        return out

    def get_blacklist_info(self) -> dict:
        return {
            "limit": BLACKLIST_LIMIT,
            "counts": dict(self._delete_counts),
            "banned": sorted(self._blacklist),
        }

    def clear_blacklist(self) -> None:
        self._delete_counts.clear()
        self._blacklist.clear()
        self._save_blacklist()
        self._rebuild_train_index()

    def register_delete(self, word: str) -> tuple[int, int, bool]:
        """
        Track that the user removed this word from the current sentence.
        We DO NOT auto-blacklist anymore — the user explicitly removes a
        learned sign via delete_learned() when they really want to forget it.
        Mistakes happen, so accidental sentence deletes never destroy training.
        """
        w = word.strip().lower()
        if not w:
            return (0, BLACKLIST_LIMIT, False)
        c = self._delete_counts.get(w, 0) + 1
        self._delete_counts[w] = c
        self._save_blacklist()
        return (c, BLACKLIST_LIMIT, False)

    def delete_learned(self, word: str, drop_test: bool = True) -> bool:
        """
        Permanently remove a learned sign (and any tracked counters).
        Also drops the corresponding test samples by default so the test
        set never references a sign the user no longer owns.
        Returns True if the sign existed in the train set and was removed.
        """
        w = word.strip().lower()
        removed = False
        if w in self._learned:
            del self._learned[w]
            removed = True
            self._save_learned()
        if drop_test and w in self._test:
            del self._test[w]
            self._save_test()
        if removed:
            self._delete_counts.pop(w, None)
            self._blacklist.discard(w)
            self._save_blacklist()
        return removed

    def undo_last(self) -> Optional[str]:
        # Client owns the sentence list; this is a no-op kept for API compat.
        return None

    def reset_buffer(self) -> None:
        """Clear all per-frame state used for word acceptance."""
        self._recent.clear()
        self._hold_buf.clear()
        self._hand_streak = 0

    @property
    def is_teaching(self) -> bool:
        return self._teach_sign is not None

    @property
    def teach_status(self) -> dict:
        if self._teach_sign is None and self._teach_phase != "done":
            return {"phase": "idle"}
        now = time.time()
        if self._teach_phase == "countdown":
            remaining = max(0.0, TEACH_COUNTDOWN_SEC - (now - self._teach_t0))
            return {
                "phase": "countdown",
                "sign": self._teach_sign,
                "remaining": round(remaining, 1),
                "use_glove": self._teach_use_glove,
                "glove_connected": self.glove_is_connected(),
            }
        if self._teach_phase == "capture":
            elapsed = now - self._teach_t0 - TEACH_COUNTDOWN_SEC
            progress = min(100, int(100 * elapsed / TEACH_CAPTURE_SEC))
            return {
                "phase": "capture",
                "sign": self._teach_sign,
                "progress": progress,
                "samples": len(self._teach_samples),
                "glove_samples": len(self._teach_glove_samples),
                "use_glove": self._teach_use_glove,
                "glove_connected": self.glove_is_connected(),
            }
        if self._teach_phase == "done":
            return {
                "phase": "done",
                "sign": self._teach_last_sign,
                "samples": self._teach_last_count,
                "glove_samples": self._teach_last_glove_count,
                "session": self._teach_last_session,
                "mode": self._teach_last_mode,
            }
        return {"phase": "idle"}

    def start_teach(
        self,
        sign: str,
        mode: str = "train",
        use_glove: Optional[bool] = None,
    ) -> None:
        """
        Begin a teach session for `sign`.
        mode = "train" -> samples go to _learned (used at recognition).
        mode = "test"  -> samples go to _test (held-out for evaluate.py).

        use_glove:
          None  -> implicit: foloseste manusa daca e conectata in acest moment
                   si TEACH_DEFAULT_USE_GLOVE = True (vezi config.py)
          True  -> incearca sa capturam glove chiar daca aparent nu e conectata
                   (utilizatorul a confirmat din UI ca o pune pe mana acum)
          False -> doar camera, ignora manusa chiar daca trimite date

        Captura reala de la manusa se intampla doar daca:
          - use_glove e True (dupa rezolvarea default-ului)
          - SI manusa trimite cadre noi (vezi _glove_connected)
          - SI suntem in faza "capture" (nu countdown)
        """
        sign = sign.strip().lower()
        if not sign:
            return
        if mode not in ("train", "test"):
            mode = "train"

        # Default smart: daca utilizatorul n-a fixat o preferinta, decidem
        # pe baza configului si a starii conexiunii actuale.
        if use_glove is None:
            use_glove = TEACH_DEFAULT_USE_GLOVE and self.glove_is_connected()

        self._teach_sign = sign
        self._teach_mode = mode
        self._teach_phase = "countdown"
        self._teach_t0 = time.time()
        self._teach_samples = []
        self._teach_glove_samples = []
        self._teach_use_glove = bool(use_glove)
        # Re-teaching a banned sign should rehabilitate it.
        self._blacklist.discard(sign)
        self._delete_counts.pop(sign, None)
        self._save_blacklist()

    def _finalize_teach(self) -> None:
        label = self._teach_sign
        mode = self._teach_mode
        count = len(self._teach_samples)
        glove_count = len(self._teach_glove_samples)
        session_id = self._next_session_id(label, mode) if label else 0
        if label and count > 0:
            entries = [{"s": session_id, "f": f} for f in self._teach_samples]
            bucket = self._learned if mode == "train" else self._test
            bucket[label] = bucket.get(label, []) + entries
            if mode == "train":
                self._save_learned()
            else:
                self._save_test()

        # Salvam si vectorii de manusa (daca am captat cel putin cativa).
        # Pragul de 3 mostre = same KNN_MIN_EXAMPLES pe care il cere si camera;
        # mai putin de atat nu e statistic util si nici n-ar trece de filtrul
        # din _rebuild_glove_index.
        if label and glove_count >= KNN_MIN_EXAMPLES:
            g_entries = [
                {"s": session_id, "f": f} for f in self._teach_glove_samples
            ]
            g_bucket = self._learned_glove if mode == "train" else self._test_glove
            g_bucket[label] = g_bucket.get(label, []) + g_entries
            if mode == "train":
                self._save_glove()
            else:
                self._save_test_glove()

        self._teach_last_sign = label
        self._teach_last_count = count
        self._teach_last_glove_count = glove_count
        self._teach_last_session = session_id
        self._teach_last_mode = mode
        self._teach_sign = None
        self._teach_samples = []
        self._teach_glove_samples = []
        self._teach_use_glove = False
        self._teach_phase = "done"
        self._recent.clear()

    def is_smoothed_match(self, label: str) -> bool:
        """True iff label has been the top guess for SMOOTH_MATCH of the last SMOOTH_N frames."""
        if not label:
            return False
        c = Counter(self._recent)
        return c.get(label, 0) >= SMOOTH_MATCH

    # --- Per-frame processing ---

    def process_frame(self, frame_rgb: np.ndarray) -> dict:
        """
        Process one RGB frame.

        Returns:
          {
            "hand_detected": bool,
            "hand_box": (x1, y1, x2, y2) or None,
            "prediction": {"label", "confidence", "source"} or None,
          }
        """
        h, w = frame_rgb.shape[:2]
        res = self._hands.process(frame_rgb)
        hands_lms = res.multi_hand_landmarks or []
        handedness = res.multi_handedness or []

        hand_detected = len(hands_lms) > 0
        hand_box = None
        feats = None

        if hand_detected:
            xs, ys = [], []
            right_lm = None
            left_lm = None
            for i, lm in enumerate(hands_lms):
                for p in lm.landmark:
                    xs.append(p.x * w)
                    ys.append(p.y * h)
                label = "Right"
                if i < len(handedness):
                    label = handedness[i].classification[0].label
                if label == "Right" and right_lm is None:
                    right_lm = lm
                elif label == "Left" and left_lm is None:
                    left_lm = lm
                elif right_lm is None:
                    right_lm = lm
                else:
                    left_lm = lm
            if xs:
                pad = 20
                x1 = max(0, int(min(xs)) - pad)
                y1 = max(0, int(min(ys)) - pad)
                x2 = min(w, int(max(xs)) + pad)
                y2 = min(h, int(max(ys)) + pad)
                hand_box = (x1, y1, x2, y2)

            feats = np.concatenate([
                _normalize_hand(right_lm),
                _normalize_hand(left_lm),
            ]).astype(np.float32)

        # Teach mode handling
        if self._teach_sign is not None:
            now = time.time()
            elapsed = now - self._teach_t0
            if self._teach_phase == "countdown":
                if elapsed >= TEACH_COUNTDOWN_SEC:
                    self._teach_phase = "capture"
            if self._teach_phase == "capture":
                if feats is not None and hand_detected:
                    self._teach_samples.append(feats.copy())
                # Captura paralela glove: o singura mostra per frame de
                # camera, luata din ultimul vector primit de la /ws/glove.
                # Daca manusa s-a deconectat in mijlocul sesiunii sau nu
                # a fost ceruta, sarim peste fara erori.
                if (
                    self._teach_use_glove
                    and self._glove_current is not None
                    and self.glove_is_connected()
                ):
                    self._teach_glove_samples.append(self._glove_current.copy())
                if elapsed >= TEACH_COUNTDOWN_SEC + TEACH_CAPTURE_SEC:
                    self._finalize_teach()

            return {
                "hand_detected": hand_detected,
                "hand_box": hand_box,
                "prediction": None,
            }

        # No hands -> no prediction. Reset all stability state so a sign
        # only counts if the user keeps their hand in frame the whole time.
        if not hand_detected or feats is None:
            self._recent.clear()
            self._hold_buf.clear()
            self._hand_streak = 0
            return {
                "hand_detected": False,
                "hand_box": None,
                "prediction": None,
            }

        # Hand is visible. Grow the streak so very brief detections don't
        # produce predictions.
        self._hand_streak += 1
        if self._hand_streak < HAND_REQUIRED_FRAMES:
            self._hold_buf.append(None)
            return {
                "hand_detected": True,
                "hand_box": hand_box,
                "prediction": None,
            }

        # Predictie cu FUZIUNE 60% camera + 40% manusa (vezi FUSION_WEIGHT_*
        # in config.py). Declansarea e naturala: ajungem aici DOAR cand camera
        # a detectat mana (manusa verde cu varfuri expuse), deci nu mai e nevoie
        # de o tasta de pornire si nu apare spam in repaus. Daca manusa nu e
        # conectata, predict_fused cade automat pe doar-camera.
        pred = self.predict_fused(feats)
        if pred is not None:
            self._recent.append(pred["label"])
            self._hold_buf.append(pred["label"])
        else:
            self._hold_buf.append(None)

        return {
            "hand_detected": True,
            "hand_box": hand_box,
            "prediction": pred,
        }

    # --- Word acceptance gating ---

    def try_accept_word(self, label: str, conf: float, threshold: float) -> dict:
        """
        Decide whether `label` (current top prediction) should be committed
        as a word. Returns a dict with:
            accepted: bool
            reason:   short string for debugging/UX
            progress: 0..1 — how close we are to acceptance (UI hint)

        A word is accepted only when ALL of the following hold:
          * conf >= threshold (caller already filtered, but double-check)
          * a hand has been visible for >= HAND_REQUIRED_FRAMES in a row
          * the same label dominates the recent hold buffer
            (>= WORD_HOLD_RATIO of the last WORD_HOLD_FRAMES frames)
          * at least WORD_COOLDOWN_SEC has passed since the previous accept
        """
        now = time.time()

        if not label or conf < threshold:
            return {"accepted": False, "reason": "low_conf", "progress": 0.0}

        if self._hand_streak < HAND_REQUIRED_FRAMES:
            return {"accepted": False, "reason": "hand_streak", "progress": 0.0}

        # Cooldown: don't fire two words back-to-back.
        if (now - self._last_accept_t) < WORD_COOLDOWN_SEC:
            return {"accepted": False, "reason": "cooldown", "progress": 0.0}

        # Hold check — count how many of the last frames agree on this label.
        if not self._hold_buf:
            return {"accepted": False, "reason": "warmup", "progress": 0.0}
        agree = sum(1 for x in self._hold_buf if x == label)
        ratio = agree / WORD_HOLD_FRAMES
        progress = min(1.0, ratio / WORD_HOLD_RATIO)

        if len(self._hold_buf) < WORD_HOLD_FRAMES:
            return {"accepted": False, "reason": "warmup", "progress": progress}
        if ratio < WORD_HOLD_RATIO:
            return {"accepted": False, "reason": "not_dominant", "progress": progress}

        # Accept!
        self._last_accept_t = now
        self._last_accept_label = label
        # Clear so the next word starts from scratch.
        self._hold_buf.clear()
        self._recent.clear()
        return {"accepted": True, "reason": "ok", "progress": 1.0}

    # --- Classification ---

    def _classify_knn(self, feats: np.ndarray) -> Optional[dict]:
        """
        Vectorised KNN with cosine distance over learned signs.
        Single matmul per frame regardless of how many signs are stored.
        Returns {label, confidence, source} or None when no confident match.
        """
        if (
            self._train_X_norm is None
            or len(self._train_labels) == 0
        ):
            return None

        q_norm = float(np.linalg.norm(feats))
        if q_norm < 1e-8:
            return None
        q = (feats / q_norm).astype(np.float32)

        # Cosine similarity for every stored sample in one shot.
        sims = self._train_X_norm @ q                  # shape (N,)
        dists = 1.0 - sims                              # cosine distance

        k = min(KNN_K, dists.shape[0])
        # argpartition gives the k smallest distances in O(N) without a sort.
        idx_part = np.argpartition(dists, k - 1)[:k]
        # Then sort those k for stable top-1 / vote ordering.
        idx_sorted = idx_part[np.argsort(dists[idx_part])]

        top_labels = [self._train_labels[i] for i in idx_sorted]
        top_dists = dists[idx_sorted]

        votes = Counter(top_labels)
        best_label, best_count = votes.most_common(1)[0]
        best_dist = float(min(
            d for lbl, d in zip(top_labels, top_dists) if lbl == best_label
        ))

        if best_dist > KNN_MAX_COSINE:
            return None

        # Geometric confidence: 1 at distance 0, 0 at KNN_MAX_COSINE,
        # with a small boost proportional to vote share.
        base = max(0.0, 1.0 - best_dist / KNN_MAX_COSINE)
        vote_frac = best_count / k
        conf = base * (0.6 + 0.4 * vote_frac)
        conf = float(max(0.0, min(1.0, conf)))

        return {"label": best_label, "confidence": conf, "source": "learned"}

    # --- Glove pipeline -------------------------------------------------

    GLOVE_STALE_AFTER_SEC = 0.5  # mai vechi de atat → consideram deconectata

    def feed_glove_frame(self, msg: dict) -> None:
        """
        Apelat de server.py de fiecare data cand soseste un glove_frame
        pe WebSocket-ul /ws/glove. Extrage vectorul 11D normalizat si il
        stocheaza ca "ultimul cadru cunoscut".

        Tot ce face mai departe — Teach Mode, predictie, fuziune — citeste
        din self._glove_current. Asa decuplam I/O-ul retelei de logica
        clasificatorului: serverul nu trebuie sa stie nimic despre KNN.
        """
        vec = extract_glove_vector(msg)
        if vec is None:
            return
        self._glove_current = vec
        self._glove_buffer.append(vec)
        self._glove_last_update_t = time.time()
        self._glove_connected = True

    def glove_disconnect(self) -> None:
        """Apelat de server cand WebSocket-ul manusii se inchide."""
        self._glove_connected = False
        self._glove_buffer.clear()  # nu amesteca cadre vechi dupa reconectare
        # Nu sterg _glove_current — poate vrem ultima valoare pentru debug.

    def glove_is_connected(self) -> bool:
        """
        Considera manusa "conectata" doar daca am primit un cadru in ultimele
        GLOVE_STALE_AFTER_SEC secunde. Asa evitam scenarii gen "manusa s-a
        oprit dar fanionul ramane True pentru ca n-am fost notificati".
        """
        if not self._glove_connected:
            return False
        return (time.time() - self._glove_last_update_t) <= self.GLOVE_STALE_AFTER_SEC

    def get_glove_state(self) -> dict:
        """Pentru UI / debug."""
        return {
            "connected": self.glove_is_connected(),
            "last_update_age": (
                time.time() - self._glove_last_update_t
                if self._glove_last_update_t > 0 else None
            ),
            "n_signs_with_glove": len(self._learned_glove),
            "n_glove_samples": sum(len(v) for v in self._learned_glove.values()),
        }

    def _classify_knn_glove(self, vec: np.ndarray) -> Optional[dict]:
        """
        KNN cosinus pe vectorul 11D. Aceeasi structura ca _classify_knn,
        dar foloseste indexul glove si pragul propriu (KNN_MAX_COSINE_GLOVE).
        Returnez {label, confidence, source: "glove"} sau None.
        """
        if self._train_glove_X_norm is None or not self._train_glove_labels:
            return None

        q_norm = float(np.linalg.norm(vec))
        if q_norm < 1e-8:
            return None
        q = (vec / q_norm).astype(np.float32)

        sims = self._train_glove_X_norm @ q
        dists = 1.0 - sims

        k = min(KNN_K, dists.shape[0])
        idx_part = np.argpartition(dists, k - 1)[:k]
        idx_sorted = idx_part[np.argsort(dists[idx_part])]

        top_labels = [self._train_glove_labels[i] for i in idx_sorted]
        top_dists = dists[idx_sorted]

        votes = Counter(top_labels)
        best_label, best_count = votes.most_common(1)[0]
        best_dist = float(min(
            d for lbl, d in zip(top_labels, top_dists) if lbl == best_label
        ))

        if best_dist > KNN_MAX_COSINE_GLOVE:
            return None

        base = max(0.0, 1.0 - best_dist / KNN_MAX_COSINE_GLOVE)
        vote_frac = best_count / k
        conf = base * (0.6 + 0.4 * vote_frac)
        conf = float(max(0.0, min(1.0, conf)))

        return {"label": best_label, "confidence": conf, "source": "glove"}

    def predict_glove(self) -> Optional[dict]:
        """
        Predictie folosind vectorul NETEZIT al manusii (media ultimelor
        GLOVE_SMOOTH_FRAMES cadre). Netezirea reduce zgomotul per-cadru, deci
        eticheta nu mai "sare" intre semne apropiate. Daca buffer-ul e gol
        (de ex. imediat dupa conectare), cade pe ultimul cadru brut.
        """
        if self._glove_current is None or not self.glove_is_connected():
            return None
        if self._glove_buffer:
            vec = np.mean(np.stack(self._glove_buffer), axis=0).astype(np.float32)
        else:
            vec = self._glove_current
        return self._classify_knn_glove(vec)

    def predict_fused(self, feats_cam: Optional[np.ndarray]) -> Optional[dict]:
        """
        Fuziune scor ponderat camera + manusa.

        Comportament:
          - Daca ambele canale sunt disponibile si dau aceeasi eticheta →
            confidente combinate (FUSION_WEIGHT_CAMERA * cam + FUSION_WEIGHT_GLOVE * glove).
          - Daca dau etichete diferite → castiga cea cu scorul ponderat mai mare,
            dar confidenta ei e usor penalizata (multiplicata cu 0.85) ca sa
            reflecte dezacordul.
          - Daca doar unul e disponibil → returnam direct rezultatul lui.
          - Daca niciunul nu da rezultat valid → None.
        """
        cam_pred = self._classify_knn(feats_cam) if feats_cam is not None else None
        glove_pred = self.predict_glove()

        if cam_pred is None and glove_pred is None:
            return None

        if glove_pred is None:
            return cam_pred  # fara manusa, exact ca inainte

        if cam_pred is None:
            # Cand n-avem camera, etichetam ca atare (e diferit de fuziune)
            return glove_pred

        # Avem ambele rezultate
        if cam_pred["label"] == glove_pred["label"]:
            fused_conf = (
                FUSION_WEIGHT_CAMERA * cam_pred["confidence"]
                + FUSION_WEIGHT_GLOVE * glove_pred["confidence"]
            )
            return {
                "label": cam_pred["label"],
                "confidence": float(min(1.0, fused_conf)),
                "source": "fused",
            }

        # Dezacord intre cele doua canale
        cam_score = FUSION_WEIGHT_CAMERA * cam_pred["confidence"]
        glove_score = FUSION_WEIGHT_GLOVE * glove_pred["confidence"]
        winner = cam_pred if cam_score >= glove_score else glove_pred
        return {
            "label": winner["label"],
            "confidence": float(max(cam_score, glove_score) * 0.85),
            "source": "fused_disagree",
        }
