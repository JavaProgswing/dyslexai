"""
Training Pipeline — ADHD Handwriting Anomaly Detection
=======================================================
Works from ANY directory. No TensorFlow/PyTorch/DLL needed.

Usage:
    python train_pipeline.py                              # synthetic only
    python train_pipeline.py --data_dir data/sessions     # real + synthetic supplement
    python train_pipeline.py --data_dir data/sessions --no_synthetic  # real only
"""

import argparse, json, sys, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for candidate in [HERE / "model", HERE.parent / "model"]:
    if (candidate / "lstm_anomaly.py").exists():
        sys.path.insert(0, str(candidate.parent))
        break
else:
    print("ERROR: Cannot find model/lstm_anomaly.py relative to this script.")
    sys.exit(1)

from model.lstm_anomaly import extract_features, train, WINDOW, FEATURES


# ── Synthetic data generator ───────────────────────────────────────────────────


def _synthetic_session(rng, n_pts=150, device="pen"):
    """
    Generate a synthetic 'normal' session.
    device="pen"   → IAM-like: high Hz, small dx, pressure variation
    device="mouse" → browser-like: ~30Hz, larger dx, pressure=1.0
    """
    if device == "mouse":
        dt_range = (14, 50)  # ~20-70fps
        dx_scale = 6.0  # larger per-event steps
        pressure = 1.0
    else:
        dt_range = (5, 20)  # ~50-200Hz pen
        dx_scale = 1.5
        pressure = None  # varies

    t = np.cumsum(rng.uniform(*dt_range, n_pts))
    x = np.clip(np.cumsum(rng.normal(dx_scale, dx_scale * 0.6, n_pts)) / 600, 0, 1)
    y = np.clip(0.5 + rng.normal(0, 0.05, n_pts), 0, 1)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    # Speed per second (matching the fixed extract_features)
    dt_s = np.diff(t / 1000.0, prepend=0.016)
    dt_s = np.where(dt_s < 0.001, 0.001, dt_s)
    sp = np.hypot(dx, dy) / dt_s
    ac = np.diff(sp, prepend=0.0)
    dt = np.diff(t, prepend=0.0)
    pr = (
        np.clip(rng.normal(0.8, 0.1, n_pts), 0, 1)
        if pressure is None
        else np.full(n_pts, pressure)
    )
    si = np.zeros(n_pts)
    cu = np.zeros(n_pts)
    return np.stack([x, y, dx, dy, sp, ac, dt, pr, si, cu], axis=-1).astype(np.float32)


def generate_synthetic(n=300, seed=0, mouse_fraction=0.3):
    """Generate synthetic sessions mixing pen and mouse styles."""
    rng = np.random.default_rng(seed)
    sessions = []
    n_mouse = int(n * mouse_fraction)
    n_pen = n - n_mouse
    for _ in range(n_pen):
        sessions.append(_synthetic_session(rng, rng.integers(80, 220), device="pen"))
    for _ in range(n_mouse):
        sessions.append(_synthetic_session(rng, rng.integers(80, 220), device="mouse"))
    print(f"  Synthetic: {n_pen} pen + {n_mouse} mouse")
    return sessions


# ── Real data loader ───────────────────────────────────────────────────────────


def load_sessions(data_dir: pathlib.Path) -> list[np.ndarray]:
    """
    Load all label=normal sessions regardless of source.
    IAM pen sessions and browser mouse sessions are both treated as normal —
    the model must learn both are acceptable.
    """
    files = sorted(data_dir.glob("*.json"))
    sessions, iam_n, mouse_n, skipped = [], 0, 0, 0

    for fp in files:
        try:
            sess = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [skip] {fp.name}: {e}")
            continue

        if sess.get("label", "normal").lower() != "normal":
            skipped += 1
            continue

        raw = extract_features(sess["strokes"])
        if len(raw) < 20:
            skipped += 1
            continue

        sessions.append(raw)
        if sess.get("source") == "browser_mouse" or sess.get("input_type") == "mouse":
            mouse_n += 1
        else:
            iam_n += 1

    print(f"  IAM pen: {iam_n}  |  Mouse: {mouse_n}  |  Skipped: {skipped}")
    return sessions


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=pathlib.Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--no_synthetic",
        action="store_true",
        help="Don't supplement with synthetic data",
    )
    args = parser.parse_args()

    print("── ADHD Handwriting Anomaly Detector — Training ──\n")

    sessions = []

    if args.data_dir:
        data_dir = (
            args.data_dir
            if args.data_dir.is_absolute()
            else pathlib.Path.cwd() / args.data_dir
        )
        if not data_dir.exists():
            print(f"ERROR: --data_dir does not exist: {data_dir}")
            sys.exit(1)
        print(f"Loading sessions from: {data_dir}")
        sessions = load_sessions(data_dir)

        # Supplement with synthetic if too few real sessions
        if not args.no_synthetic:
            if len(sessions) < 50:
                n_add = max(200, 50 - len(sessions))
                print(
                    f"\n  Only {len(sessions)} real sessions — adding {n_add} synthetic."
                )
                sessions += generate_synthetic(n=n_add)
            else:
                # Always add some synthetic mouse to ensure good mouse coverage
                mouse_in_real = sum(
                    1
                    for fp in data_dir.glob("*.json")
                    if json.loads(fp.read_text()).get("input_type") == "mouse"
                )
                if mouse_in_real < 15:
                    n_synth_mouse = 15 - mouse_in_real
                    print(
                        f"  Only {mouse_in_real} real mouse sessions — adding {n_synth_mouse} synthetic mouse."
                    )
                    rng = np.random.default_rng(999)
                    for _ in range(n_synth_mouse):
                        sessions.append(
                            _synthetic_session(rng, rng.integers(80, 220), "mouse")
                        )
    else:
        print("No --data_dir — using synthetic data (pen + mouse mix).")
        sessions = generate_synthetic(n=300)

    print(f"\nTotal sessions: {len(sessions)}")

    np.random.seed(42)
    np.random.shuffle(sessions)
    split = int(len(sessions) * 0.85)
    train_s, val_s = sessions[:split], sessions[split:]
    print(f"Train: {len(train_s)}  |  Val: {len(val_s)}\n")

    model, threshold = train(train_s, val_s)

    print(f"\n✅  Done. Threshold = {threshold:.6f}")
    print(f"   Weights: {HERE}/model/weights/sklearn_model.joblib")
    print("\nNext: uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
