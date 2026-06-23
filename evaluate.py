"""
Evaluate the trained KNN recogniser on the data captured via the web app.

Produces a complete report suitable for the thesis "Rezultate experimentale"
section: top-1 / top-3 accuracy, per-class precision / recall / F1, macro
F1, confusion matrix (PNG), inference latency, and a few-shot accuracy curve.

Two evaluation modes are supported:

  --mode held_out        (default if test_signs.pkl exists)
      Train on every sample in learned_signs.pkl, test on every sample in
      test_signs.pkl. This is the strict "held-out test set" protocol — the
      most defensible cifra finala for the thesis.

  --mode kfold
      k-fold cross-validation over teach SESSIONS within learned_signs.pkl.
      Falls back to k-fold over individual SAMPLES (with a warning) when
      every label has only one session, so the script still produces some
      numbers on the legacy data that pre-dates session tracking.

Usage:
    python evaluate.py                       # auto-detects best mode
    python evaluate.py --mode kfold --k 4
    python evaluate.py --mode held_out
    python evaluate.py --few-shot            # also generate few-shot curve

Outputs go to eval_reports/<timestamp>/  (results.md, confusion.png, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Allow `python evaluate.py` from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LEARNED_SIGNS_PATH,
    TEST_SIGNS_PATH,
    EVAL_REPORTS_DIR,
    FEAT_DIM,
    KNN_K,
    KNN_MAX_COSINE,
    KNN_MIN_EXAMPLES,
)
from recognizers.smart import SmartRecognizer

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# --- Data loading -------------------------------------------------------

def load_signs(path: Path) -> dict[str, list[dict]]:
    """Use SmartRecognizer's loader so the legacy/new format handling is
    identical to what the live app sees."""
    return SmartRecognizer._load_pickle_signs(path)


def flatten(bucket: dict[str, list[dict]]):
    """dict[label, [{s,f}]] -> (X (N,D), y (N,), sessions (N,))"""
    feats, labels, sessions = [], [], []
    for label, entries in bucket.items():
        for e in entries:
            feats.append(e["f"])
            labels.append(label)
            sessions.append(int(e["s"]))
    if not feats:
        return (np.zeros((0, FEAT_DIM), dtype=np.float32),
                np.array([], dtype=object), np.array([], dtype=int))
    X = np.stack(feats).astype(np.float32)
    return X, np.array(labels, dtype=object), np.array(sessions, dtype=int)


# --- Vectorised KNN (same logic as smart.py, separated for testing) -----

class KNN:
    def __init__(self, k=KNN_K, max_cos=KNN_MAX_COSINE):
        self.k = k
        self.max_cos = max_cos
        self.X_norm = None
        self.labels = None

    def fit(self, X, y):
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        self.X_norm = (X / norms).astype(np.float32)
        self.labels = np.asarray(y, dtype=object)
        return self

    def predict(self, Q, top_n=1):
        """Returns (top_n predictions per query, distances)."""
        qn = np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8
        Qn = (Q / qn).astype(np.float32)
        sims = Qn @ self.X_norm.T                  # (M, N)
        dists = 1.0 - sims                          # cosine dist
        k = min(self.k, dists.shape[1])
        # k-nearest indices per query
        idx_part = np.argpartition(dists, k - 1, axis=1)[:, :k]
        # For each row we sort the k candidates by distance
        out_top = []
        out_dists = []
        for i in range(Q.shape[0]):
            ki = idx_part[i]
            order = np.argsort(dists[i, ki])
            ki = ki[order]
            di = dists[i, ki]
            # Vote among k neighbours, weight by 1/(dist+eps)
            votes = Counter()
            best_dist_per_label = {}
            for j, lbl in enumerate(self.labels[ki]):
                votes[lbl] += 1
                if lbl not in best_dist_per_label or di[j] < best_dist_per_label[lbl]:
                    best_dist_per_label[lbl] = di[j]
            # Sort labels by (vote desc, best_dist asc) and take top_n
            ranked = sorted(
                votes.keys(),
                key=lambda l: (-votes[l], best_dist_per_label[l]),
            )
            out_top.append(ranked[:top_n])
            out_dists.append([best_dist_per_label[l] for l in ranked[:top_n]])
        return out_top, out_dists


# --- Metrics ------------------------------------------------------------

def per_class_prf(y_true, y_pred, labels):
    """Precision / recall / F1 per class + macro averages."""
    res = {}
    for lbl in labels:
        tp = int(np.sum((y_true == lbl) & (y_pred == lbl)))
        fp = int(np.sum((y_true != lbl) & (y_pred == lbl)))
        fn = int(np.sum((y_true == lbl) & (y_pred != lbl)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = int(np.sum(y_true == lbl))
        res[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    macro = {
        "precision": float(np.mean([r["precision"] for r in res.values()])),
        "recall":    float(np.mean([r["recall"]    for r in res.values()])),
        "f1":        float(np.mean([r["f1"]        for r in res.values()])),
    }
    return res, macro


def confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


# --- Evaluation modes --------------------------------------------------

def eval_held_out(train_bucket, test_bucket, top_k_show=3):
    Xtr, ytr, _ = flatten(train_bucket)
    Xte, yte, _ = flatten(test_bucket)
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        return None

    labels = sorted(set(ytr.tolist()) | set(yte.tolist()))
    knn = KNN().fit(Xtr, ytr)

    top_preds, _ = knn.predict(Xte, top_n=top_k_show)
    y_pred1 = np.array([p[0] if p else "" for p in top_preds], dtype=object)
    top1 = float(np.mean(y_pred1 == yte))
    top3 = float(np.mean([yte[i] in top_preds[i] for i in range(len(yte))]))

    # Latency
    t0 = time.perf_counter()
    knn.predict(Xte, top_n=1)
    dt = (time.perf_counter() - t0) / max(1, Xte.shape[0]) * 1000

    per_cls, macro = per_class_prf(yte, y_pred1, labels)
    cm = confusion_matrix(yte, y_pred1, labels)

    return {
        "mode": "held_out",
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_labels": len(labels),
        "labels": labels,
        "top1": top1,
        "top3": top3,
        "macro": macro,
        "per_class": per_cls,
        "confusion": cm.tolist(),
        "latency_ms": float(dt),
    }


def eval_kfold(train_bucket, k=4):
    """K-fold over sessions when possible, otherwise over samples."""
    Xtr, ytr, str_ = flatten(train_bucket)
    if Xtr.shape[0] == 0:
        return None
    labels = sorted(set(ytr.tolist()))
    # Build (label, session) -> indices
    keyed = defaultdict(list)
    for i in range(Xtr.shape[0]):
        keyed[(ytr[i], int(str_[i]))].append(i)
    sessions_per_label = {l: sorted({s for (lbl, s) in keyed.keys() if lbl == l})
                          for l in labels}
    min_sessions = min(len(v) for v in sessions_per_label.values())

    fold_mode = "session" if min_sessions >= k else "sample"
    if fold_mode == "sample":
        print(f"[evaluate] WARN: at least one label has only {min_sessions} "
              f"session(s); falling back to sample-level k-fold (less rigorous).")

    fold_results = []
    rng = np.random.default_rng(42)

    for fold in range(k):
        train_mask = np.ones(Xtr.shape[0], dtype=bool)
        if fold_mode == "session":
            for l in labels:
                ss = sessions_per_label[l]
                test_sess = ss[fold % len(ss)]
                for i in keyed[(l, test_sess)]:
                    train_mask[i] = False
        else:
            # Sample-level k-fold (stratified)
            for l in labels:
                idx = np.array([i for i in range(Xtr.shape[0]) if ytr[i] == l])
                rng.shuffle(idx)
                fold_size = max(1, len(idx) // k)
                start = fold * fold_size
                end = start + fold_size if fold < k - 1 else len(idx)
                for i in idx[start:end]:
                    train_mask[i] = False

        Xtrain = Xtr[train_mask]; ytrain = ytr[train_mask]
        Xtest  = Xtr[~train_mask]; ytest  = ytr[~train_mask]
        if Xtrain.shape[0] == 0 or Xtest.shape[0] == 0:
            continue
        knn = KNN().fit(Xtrain, ytrain)
        top_preds, _ = knn.predict(Xtest, top_n=3)
        y_pred1 = np.array([p[0] if p else "" for p in top_preds], dtype=object)
        top1 = float(np.mean(y_pred1 == ytest))
        top3 = float(np.mean([ytest[i] in top_preds[i] for i in range(len(ytest))]))
        per_cls, macro = per_class_prf(ytest, y_pred1, labels)
        fold_results.append({
            "fold": fold + 1,
            "n_test": int(Xtest.shape[0]),
            "top1": top1, "top3": top3, "macro_f1": macro["f1"],
            "per_class": per_cls,
        })

    if not fold_results:
        return None

    # Aggregate
    top1s = [r["top1"] for r in fold_results]
    top3s = [r["top3"] for r in fold_results]
    macros = [r["macro_f1"] for r in fold_results]
    return {
        "mode": "kfold",
        "fold_mode": fold_mode,
        "k": k,
        "n_train_total": int(Xtr.shape[0]),
        "n_labels": len(labels),
        "labels": labels,
        "fold_results": fold_results,
        "top1_mean": float(np.mean(top1s)), "top1_std": float(np.std(top1s)),
        "top3_mean": float(np.mean(top3s)), "top3_std": float(np.std(top3s)),
        "macro_f1_mean": float(np.mean(macros)), "macro_f1_std": float(np.std(macros)),
    }


def few_shot_curve(train_bucket, test_bucket, sample_counts=(5, 10, 20, 40, 80, 160)):
    """Vary per-class train samples and measure top-1 accuracy on test set."""
    Xte_full, yte_full, _ = flatten(test_bucket)
    if Xte_full.shape[0] == 0:
        return []
    Xtr_full, ytr_full, _ = flatten(train_bucket)
    rng = np.random.default_rng(0)
    out = []
    for n in sample_counts:
        # Sample n per class from train
        keep = []
        for l in sorted(set(ytr_full.tolist())):
            idx = np.where(ytr_full == l)[0]
            if len(idx) <= n:
                keep.extend(idx.tolist())
            else:
                rng.shuffle(idx)
                keep.extend(idx[:n].tolist())
        if not keep:
            continue
        Xs = Xtr_full[keep]; ys = ytr_full[keep]
        knn = KNN().fit(Xs, ys)
        top_preds, _ = knn.predict(Xte_full, top_n=3)
        y_pred1 = np.array([p[0] if p else "" for p in top_preds], dtype=object)
        top1 = float(np.mean(y_pred1 == yte_full))
        top3 = float(np.mean([yte_full[i] in top_preds[i] for i in range(len(yte_full))]))
        out.append({"n_per_class": n, "n_train_total": int(Xs.shape[0]),
                    "top1": top1, "top3": top3})
    return out


# --- Reporting ----------------------------------------------------------

def write_report(out_dir: Path, result: dict, fewshot=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# ASL Recognition — Evaluation Report",
             f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"\nMode: **{result['mode']}**\n"]

    if result["mode"] == "held_out":
        lines += [
            f"- Train samples: **{result['n_train']}**",
            f"- Test samples:  **{result['n_test']}**",
            f"- Labels: **{result['n_labels']}**",
            f"- **Top-1 accuracy:** {result['top1']*100:.2f}%",
            f"- **Top-3 accuracy:** {result['top3']*100:.2f}%",
            f"- Macro F1: {result['macro']['f1']*100:.2f}%",
            f"- Macro precision: {result['macro']['precision']*100:.2f}%",
            f"- Macro recall: {result['macro']['recall']*100:.2f}%",
            f"- Inference latency: **{result['latency_ms']:.3f} ms / sample**",
            "",
            "## Per-class metrics",
            "",
            "| Label | Support | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for lbl in result["labels"]:
            r = result["per_class"][lbl]
            lines.append(
                f"| {lbl} | {r['support']} | "
                f"{r['precision']*100:.1f}% | {r['recall']*100:.1f}% | {r['f1']*100:.1f}% |"
            )

    elif result["mode"] == "kfold":
        lines += [
            f"- Fold strategy: **{result['fold_mode']}-level**",
            f"- Folds: **{result['k']}**",
            f"- Total train samples: **{result['n_train_total']}**",
            f"- Labels: **{result['n_labels']}**",
            f"- **Top-1 accuracy:** {result['top1_mean']*100:.2f}% ± {result['top1_std']*100:.2f}%",
            f"- **Top-3 accuracy:** {result['top3_mean']*100:.2f}% ± {result['top3_std']*100:.2f}%",
            f"- Macro F1:        {result['macro_f1_mean']*100:.2f}% ± {result['macro_f1_std']*100:.2f}%",
            "",
            "## Fold breakdown",
            "",
            "| Fold | N_test | Top-1 | Top-3 | Macro F1 |",
            "|---:|---:|---:|---:|---:|",
        ]
        for fr in result["fold_results"]:
            lines.append(
                f"| {fr['fold']} | {fr['n_test']} | "
                f"{fr['top1']*100:.1f}% | {fr['top3']*100:.1f}% | {fr['macro_f1']*100:.1f}% |"
            )

    if fewshot:
        lines += ["", "## Few-shot curve (top-1 accuracy by samples per class)", "",
                  "| N per class | N train total | Top-1 | Top-3 |",
                  "|---:|---:|---:|---:|"]
        for r in fewshot:
            lines.append(
                f"| {r['n_per_class']} | {r['n_train_total']} | "
                f"{r['top1']*100:.1f}% | {r['top3']*100:.1f}% |"
            )

    (out_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    if HAVE_MPL and result["mode"] == "held_out":
        cm = np.array(result["confusion"])
        labels = result["labels"]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.3),
                                         max(7, len(labels) * 0.3)))
        # Row-normalize so colors reflect per-class accuracy
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True) + 1e-9
        cm_norm = cm_norm / row_sums
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix (n={int(cm.sum())}, top-1={result['top1']*100:.1f}%)")
        plt.colorbar(im, ax=ax, fraction=0.04)
        plt.tight_layout()
        plt.savefig(out_dir / "confusion.png", dpi=150)
        plt.close(fig)

    if HAVE_MPL and fewshot:
        ns = [r["n_per_class"] for r in fewshot]
        t1 = [r["top1"] * 100 for r in fewshot]
        t3 = [r["top3"] * 100 for r in fewshot]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ns, t1, "-o", label="Top-1")
        ax.plot(ns, t3, "-s", label="Top-3")
        ax.set_xlabel("Samples per class (train)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Few-shot accuracy curve")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "few_shot.png", dpi=150)
        plt.close(fig)

    print(f"\n[evaluate] Report written to: {out_dir}")
    print(f"  - results.md")
    print(f"  - results.json")
    if HAVE_MPL and result["mode"] == "held_out":
        print(f"  - confusion.png")
    if HAVE_MPL and fewshot:
        print(f"  - few_shot.png")


# --- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["auto", "held_out", "kfold"], default="auto")
    parser.add_argument("--k", type=int, default=4, help="folds for k-fold mode")
    parser.add_argument("--few-shot", action="store_true",
                        help="also generate the few-shot accuracy curve")
    args = parser.parse_args()

    train = load_signs(LEARNED_SIGNS_PATH)
    test = load_signs(TEST_SIGNS_PATH)

    print(f"[evaluate] Train: {len(train)} labels, "
          f"{sum(len(v) for v in train.values())} samples")
    print(f"[evaluate] Test:  {len(test)} labels, "
          f"{sum(len(v) for v in test.values())} samples")

    mode = args.mode
    if mode == "auto":
        mode = "held_out" if test else "kfold"
        print(f"[evaluate] Auto-selected mode: {mode}")

    out_dir = EVAL_REPORTS_DIR / time.strftime("%Y%m%d_%H%M%S")
    fewshot = None

    if mode == "held_out":
        if not test:
            print("[evaluate] ERROR: held_out mode requires test_signs.pkl, "
                  "which is empty. Collect test data via 'Modul test' in the app, "
                  "or run with --mode kfold.")
            return 2
        result = eval_held_out(train, test)
        if args.few_shot:
            fewshot = few_shot_curve(train, test)
    else:
        result = eval_kfold(train, k=args.k)

    if result is None:
        print("[evaluate] ERROR: not enough data to evaluate.")
        return 1

    write_report(out_dir, result, fewshot=fewshot)
    print("\n--- Summary ---")
    if mode == "held_out":
        print(f"  Top-1: {result['top1']*100:.2f}%")
        print(f"  Top-3: {result['top3']*100:.2f}%")
        print(f"  Macro F1: {result['macro']['f1']*100:.2f}%")
        print(f"  Latency: {result['latency_ms']:.2f} ms/sample")
    else:
        print(f"  Top-1 (mean ± std): "
              f"{result['top1_mean']*100:.2f}% ± {result['top1_std']*100:.2f}%")
        print(f"  Top-3 (mean ± std): "
              f"{result['top3_mean']*100:.2f}% ± {result['top3_std']*100:.2f}%")
        print(f"  Macro F1:           "
              f"{result['macro_f1_mean']*100:.2f}% ± {result['macro_f1_std']*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
