# DyslexAI

A backend for analyzing reading and handwriting for dyslexia/ADHD-relevant signals. It is a
research prototype, not a diagnostic tool, and makes no medical claims — it surfaces patterns for a
human to review.

## Endpoints

- `POST /api/analyze` — PCA anomaly detection over handwriting stroke features.
- `POST /api/analyze-reading` — reading-error analysis on a transcript (reversals, substitutions,
  insertions, omissions) via spaCy + difflib alignment against the target text.
- `POST /api/analyze-image` — audio in, Whisper transcribes, then the reading analyzer runs.
- `GET /api/health`, `GET /api/model/info`.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000     # from backend/
```

## Data and models

Trained weights and the IAM handwriting dataset are not included (git-ignored). To rebuild:

```bash
python backend/iam_converter.py --original_dir ./data/original
python backend/inspect_iam.py --data_dir ./data
python backend/train_pipeline.py
```

Dataset paths default to `./data/...` and are overridable with the flags above.

## Layout

```
backend/
  main.py             FastAPI app, handwriting endpoints
  reading_routes.py   reading-error endpoints
  iam_converter.py    IAM dataset -> training format
  inspect_iam.py      dataset checks
  train_pipeline.py   training
  test_model.py       evaluation
```
