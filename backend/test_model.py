"""
test_model.py — End-to-end model test using real IAM held-out data
===================================================================
Fixes from previous version:
  - Tests with real held-out IAM sessions, NOT synthetic data
  - Creates "anomalous" test cases by corrupting real IAM sessions
    (random pauses + speed bursts added to real strokes)
  - Adds risk_level display

Usage (from DyslexAI/ root or from backend/):
    python test_model.py --sessions_dir data/sessions
    python test_model.py --sessions_dir data/sessions --skip_train
"""

import argparse, json, pathlib, sys, time, copy
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for candidate in [HERE / "model", HERE / "backend" / "model"]:
    if (candidate / "lstm_anomaly.py").exists():
        sys.path.insert(0, str(candidate.parent))
        break
else:
    print("ERROR: cannot find model/lstm_anomaly.py")
    sys.exit(1)

from model.lstm_anomaly import (
    extract_features,
    load_model_and_meta,
    score_strokes,
    train,
)


# ── Risk level (mirrors main.py logic) ────────────────────────────────────────


def risk_level(confidence: float) -> str:
    if confidence < 0.35:
        return "low"
    if confidence < 0.65:
        return "moderate"
    return "high"


# ── Load real sessions from data/sessions/ ─────────────────────────────────────


def load_sessions(
    sessions_dir: pathlib.Path, max_n: int = 0
) -> list[tuple[str, list[list[dict]]]]:
    """Returns list of (filename, strokes) tuples."""
    files = sorted(sessions_dir.glob("*.json"))
    out = []
    for fp in files:
        try:
            sess = json.loads(fp.read_text(encoding="utf-8"))
            strokes = sess["strokes"]
            total = sum(len(s) for s in strokes)
            if total >= 20:
                out.append((fp.name, strokes))
        except Exception:
            pass
        if max_n and len(out) >= max_n:
            break
    return out


def load_raw_sessions(sessions_dir: pathlib.Path, max_n: int = 0) -> list[np.ndarray]:
    """Returns list of raw feature arrays for training."""
    sessions = load_sessions(sessions_dir, max_n)
    return [extract_features(s) for _, s in sessions]


# ── Corrupt a real session to simulate anomalous writing ──────────────────────


def corrupt_session(strokes: list[list[dict]], seed: int = 42) -> list[list[dict]]:
    """
    Inject realistic ADHD-like anomalies into a real IAM session:
      - Random long pauses (300–800ms gaps inserted at ~10% of points)
      - Sudden acceleration bursts (x/y jump ±20% of canvas range)
      - High-pressure spike noise
    Returns a new strokes list — original is not modified.
    """
    rng = np.random.default_rng(seed)
    corrupt = copy.deepcopy(strokes)

    for stroke in corrupt:
        t_offset = 0.0
        for i, pt in enumerate(stroke):
            # Inject a long pause at ~10% of points
            if rng.random() < 0.10:
                pause = float(rng.uniform(300, 800))
                t_offset += pause
            pt["t"] += t_offset

            # Inject a position burst at ~5% of points
            if rng.random() < 0.05:
                x_range = 6912.0
                y_range = 8798.0
                pt["x"] += float(rng.uniform(-0.15, 0.15) * x_range)
                pt["y"] += float(rng.uniform(-0.15, 0.15) * y_range)

            # Pressure spike at ~5% of points
            if rng.random() < 0.05:
                pt["pressure"] = float(rng.uniform(0.0, 0.3))

    return corrupt


# ── Print result ──────────────────────────────────────────────────────────────

RISK_ICONS = {"low": "🟢", "moderate": "🟡", "high": "🔴"}


def print_result(label: str, result: dict):
    rl = risk_level(result["confidence"])
    icon = RISK_ICONS.get(rl, "⚪")
    verdict = "ANOMALOUS ✗" if result["is_anomalous"] else "normal  ✓"
    print(f"\n  {icon}  {label}")
    print(f"     verdict       : {verdict}")
    print(
        f"     anomaly_score : {result['anomaly_score']:.4f}  "
        f"(threshold: {result['threshold']:.4f})"
    )
    print(f"     confidence    : {result['confidence']:.1%}  →  risk: {rl}")
    f = result.get("flags", {})
    print(
        f"     flags         : "
        f"pauses={f.get('long_pauses',0)}  "
        f"accel_peaks={f.get('acceleration_peaks',0)}  "
        f"speed_var={f.get('speed_variance',0):.1f}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions_dir", default="data/sessions")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--max_sessions", type=int, default=400)
    args = parser.parse_args()

    sessions_dir = (
        HERE / args.sessions_dir
        if not pathlib.Path(args.sessions_dir).is_absolute()
        else pathlib.Path(args.sessions_dir)
    )

    print("═══════════════════════════════════════")
    print("  ADHD Handwriting Model — E2E Test")
    print("═══════════════════════════════════════")

    # ── Load all real sessions ─────────────────────────────────────────────
    if not sessions_dir.exists():
        print(f"\nERROR: {sessions_dir} not found. Run iam_converter.py first.")
        sys.exit(1)

    all_sessions = load_sessions(sessions_dir, max_n=args.max_sessions)
    if not all_sessions:
        print("ERROR: No sessions loaded.")
        sys.exit(1)

    print(f"\n[0/3] Loaded {len(all_sessions)} real IAM sessions.")

    # Hold out last 10 for testing — train on everything else
    np.random.seed(42)
    np.random.shuffle(all_sessions)
    test_sessions = all_sessions[-10:]
    train_sessions = all_sessions[:-10]

    print(
        f"       Train: {len(train_sessions)}  |  Test (held-out): {len(test_sessions)}"
    )

    # ── Train or load ──────────────────────────────────────────────────────
    if args.skip_train:
        print("\n[1/3] Loading existing weights …")
        model, threshold = load_model_and_meta()
        print(f"  Threshold: {threshold:.4f}")
    else:
        raw_train = [extract_features(s) for _, s in train_sessions]
        raw_val = raw_train[int(len(raw_train) * 0.85) :]
        raw_train = raw_train[: int(len(raw_train) * 0.85)]

        print(f"\n[1/3] Training on {len(raw_train)} sessions …")
        t0 = time.perf_counter()
        model, threshold = train(raw_train, raw_val)
        print(f"  Done in {time.perf_counter()-t0:.1f}s  |  Threshold: {threshold:.4f}")

    # ── Score held-out real sessions ───────────────────────────────────────
    print("\n[2/3] Scoring held-out REAL IAM sessions (expect: normal ✓) …")
    normal_correct = 0
    for fname, strokes in test_sessions[:5]:
        result = score_strokes(strokes, model, threshold)
        print_result(f"Real IAM: {fname}", result)
        if not result["is_anomalous"]:
            normal_correct += 1
    print(f"\n  Correctly identified as normal: {normal_correct}/5")

    # ── Score corrupted sessions ───────────────────────────────────────────
    print("\n[3/3] Scoring CORRUPTED sessions (same IAM data + injected anomalies)")
    print("      (expect: anomalous ✗) …")
    anom_correct = 0
    for i, (fname, strokes) in enumerate(test_sessions[:5]):
        corrupted = corrupt_session(strokes, seed=i)
        result = score_strokes(corrupted, model, threshold)
        print_result(f"Corrupted: {fname}", result)
        if result["is_anomalous"]:
            anom_correct += 1
    print(f"\n  Correctly identified as anomalous: {anom_correct}/5")

    # ── Summary ───────────────────────────────────────────────────────────
    total_correct = normal_correct + anom_correct
    print(f"\n═══════════════════════════════════════")
    print(f"  Accuracy: {total_correct}/10 test sessions classified correctly")
    if total_correct >= 7:
        print("  ✅  Model is working. Ready to wire to FastAPI.")
    else:
        print(
            "  ⚠️   Low accuracy — try converting more sessions (--limit 0 in iam_converter.py)"
        )
        print("       and re-running with --max_sessions 400")
    print(f"\n  Next:  uvicorn main:app --reload --port 8000")
    print("═══════════════════════════════════════\n")


if __name__ == "__main__":
    main()
