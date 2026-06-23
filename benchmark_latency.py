"""
Măsoară latența reală a sistemului de recunoaștere ASL.

Rulează 200 de cadre dintr-un videoclip de test (sau folosește camera live)
și înregistrează timpul de execuție al fiecărei etape:
  - MediaPipe (detectare puncte cheie)
  - Construire feature 126D + normalizare
  - Clasificator KNN (matricial)
  - Total per-cadru

La final afișează statistici (min, max, medie, mediană, p95)

Rulare:
    cd C:\\Users\\Alina\\Desktop\\ASL_Web_App
    python benchmark_latency.py            # foloseste camera
    python benchmark_latency.py video.mp4  # foloseste un videoclip
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Forteaza recognizer-ul sa fie incarcat
from recognizers.smart import SmartRecognizer

N_FRAMES = 200          # cate cadre masuram
WARMUP = 10              # primele cadre se ignoră (JIT, cache, etc.)


def main():
    # Argument optional: cale catre un fisier video
    video_src = sys.argv[1] if len(sys.argv) > 1 else 0  # 0 = camera

    print(f"[benchmark] Sursa video: {video_src}")
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"[X] Nu pot deschide {video_src}")
        return

    print("[benchmark] Initializez recognizer-ul...")
    rec = SmartRecognizer()
    print(f"[benchmark] Recognizer ready, {len(rec.get_learned_signs())} semne invatate.\n")

    # Tinem timpii in liste separate
    t_mediapipe = []   # cat dureaza detectarea celor 21 puncte
    t_classify = []    # cat dureaza KNN-ul + normalizarea
    t_total = []       # cat dureaza process_frame() complet

    n = 0
    while n < N_FRAMES + WARMUP:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[benchmark] Stream terminat sau eroare.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Total: cat dureaza tot process_frame()
        t0 = time.perf_counter()
        result = rec.process_frame(frame_rgb)
        t1 = time.perf_counter()

        # Masuram separat MediaPipe (fara KNN)
        t2 = time.perf_counter()
        _ = rec._hands.process(frame_rgb)
        t3 = time.perf_counter()

        if n >= WARMUP:
            t_total.append((t1 - t0) * 1000)        # in milisecunde
            t_mediapipe.append((t3 - t2) * 1000)
            # KNN ≈ total - mediapipe (aprox)
            knn_estimate = max(0.0, (t1 - t0) - (t3 - t2))
            t_classify.append(knn_estimate * 1000)

        n += 1
        if n % 20 == 0:
            print(f"  ... {n}/{N_FRAMES + WARMUP} cadre procesate")

    cap.release()

    if not t_total:
        print("[X] Nu am masurat niciun cadru.")
        return

    def stats(name, arr):
        arr = np.array(arr)
        return (
            f"  {name:<22}  "
            f"medie={arr.mean():6.1f} ms   "
            f"mediana={np.median(arr):6.1f}   "
            f"min={arr.min():5.1f}   "
            f"max={arr.max():5.1f}   "
            f"p95={np.percentile(arr, 95):6.1f}"
        )

    print(f"\n{'=' * 80}")
    print(f"REZULTATE — masurate pe {len(t_total)} cadre (after {WARMUP} warmup)")
    print(f"{'=' * 80}")
    print(stats("MediaPipe Hands",      t_mediapipe))
    print(stats("KNN classification",   t_classify))
    print(stats("TOTAL per-cadru",      t_total))
    print(f"{'=' * 80}\n")

    # Concluzie automata pentru prezentare
    p95 = np.percentile(t_total, 95)
    median = np.median(t_total)
    print("[SUMAR pentru prezentare]")
    print(f"  Latenta tipica (median):  {median:.0f} ms")
    print(f"  Latenta in cel mai rau caz (p95): {p95:.0f} ms")
    print(f"  FPS efectiv (1000 / median): {1000/median:.1f} FPS")


if __name__ == "__main__":
    main()
