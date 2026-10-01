"""
IAM OnDB Converter — corrected for actual file structure
=========================================================
Actual structure discovered from inspect_iam.py:

  original/
    a01/
      a01-001/
        strokesz.xml        ← ONE file per FORM (entire page)

XML internals:
  <WhiteboardCaptureSession>
    <WhiteboardDescription>
      <DiagonallyOppositeCoords x="6912" y="8798"/>   ← canvas size
    <Transcription>
      <TextLine id="a01-001z-01" text="By Trevor Williams...">
        <Word .../>
    <StrokeSet>
      <Stroke colour="black" start_time="13090871.35">
        <Point x="895" y="992" time="13090871.35"/>    ← time in SECONDS (float)

What this script does:
  1. Walks original/ recursively finding every strokesz.xml
  2. Reads canvas size, transcription, and all strokes
  3. Converts absolute time → relative milliseconds (t=0 at first point)
  4. Writes one session JSON per form to data/sessions/

Usage (run from DyslexAI/ project root):
    python iam_converter.py
    python iam_converter.py --original_dir "./data/original"
    python iam_converter.py --original_dir "C:/..." --limit 0   # all forms
"""

import argparse, json, pathlib, xml.etree.ElementTree as ET
from typing import Optional

DEFAULT_ORIGINAL = r"./data/original"
# If you also downloaded original-xml-all.tar, use that path for all 657 writers:
# r"./data/original-all"


# ── Parse one strokesz.xml ────────────────────────────────────────────────────


def parse_form_xml(xml_path: pathlib.Path) -> Optional[dict]:
    """
    Returns a session dict or None if the file is unreadable / has no points.

    Session dict:
    {
      "label"        : "normal",
      "source"       : "iam_ondb",
      "file"         : "a01-001z",
      "writer_id"    : "10028",
      "transcription": ["By Trevor Williams. A move", "to stop Mr. Gaitskell..."],
      "strokes"      : [[{x, y, t, pressure}, ...], ...],
      "canvas_width" : 6912.0,
      "canvas_height": 8798.0,
    }
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  [parse error] {xml_path}: {e}")
        return None

    # ── Canvas dimensions ──────────────────────────────────────────────────
    canvas_w, canvas_h = 6912.0, 8798.0  # IAM default
    diag = root.find(".//DiagonallyOppositeCoords")
    if diag is not None:
        try:
            canvas_w = float(diag.attrib["x"])
            canvas_h = float(diag.attrib["y"])
        except (KeyError, ValueError):
            pass

    # ── Writer / form ID ───────────────────────────────────────────────────
    form_el = root.find(".//Form")
    form_id = (
        form_el.attrib.get("id", xml_path.parent.name)
        if form_el is not None
        else xml_path.parent.name
    )
    writer_id = (
        form_el.attrib.get("writerID", "unknown") if form_el is not None else "unknown"
    )

    # ── Transcription lines ────────────────────────────────────────────────
    transcription = []
    for tl in root.findall(".//TextLine"):
        text = tl.attrib.get("text", "").strip()
        if text:
            transcription.append(text)

    # ── Strokes ────────────────────────────────────────────────────────────
    strokes = []
    t_origin = None  # will be set to first point's absolute time

    for stroke_el in root.findall(".//Stroke"):
        points = []
        for pt in stroke_el.findall("Point"):
            try:
                x = float(pt.attrib["x"])
                y = float(pt.attrib["y"])
                t_abs = float(pt.attrib["time"])  # seconds, absolute
            except (KeyError, ValueError):
                continue

            if t_origin is None:
                t_origin = t_abs

            # Convert to relative milliseconds
            t_rel = (t_abs - t_origin) * 1000.0  # → ms from start of session

            points.append(
                {
                    "x": x,
                    "y": y,
                    "t": round(t_rel, 2),
                    "pressure": 1.0,  # IAM has no pressure data
                }
            )

        if len(points) >= 2:
            strokes.append(points)

    if not strokes:
        return None

    total_pts = sum(len(s) for s in strokes)
    if total_pts < 20:
        return None

    return {
        "label": "normal",
        "source": "iam_ondb",
        "file": form_id,
        "writer_id": writer_id,
        "transcription": transcription,
        "strokes": strokes,
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "total_points": total_pts,
        "stroke_count": len(strokes),
    }


# ── Walk directory ─────────────────────────────────────────────────────────────


def find_xml_files(original_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find all strokesz.xml files under the original/ tree."""
    files = sorted(original_dir.rglob("strokesz.xml"))
    if not files:
        # Fallback: any XML file
        files = sorted(original_dir.rglob("*.xml"))
    return files


# ── Convert ───────────────────────────────────────────────────────────────────


def convert(
    original_dir: pathlib.Path,
    out_dir: pathlib.Path,
    limit: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_files = find_xml_files(original_dir)

    if not xml_files:
        print(f"ERROR: No XML files found in {original_dir}")
        print("  Make sure you're pointing at the 'original' folder, e.g.:")
        print(r"  original-xml-part.tar\original")
        return

    print(f"Found {len(xml_files)} XML forms in {original_dir}")

    if limit and limit < len(xml_files):
        xml_files = xml_files[:limit]
        print(f"Processing first {limit} forms (pass --limit 0 for all).")

    ok, skipped = 0, 0

    for xml_path in xml_files:
        session = parse_form_xml(xml_path)
        if session is None:
            skipped += 1
            continue

        # Use form id as filename, sanitize
        safe_name = session["file"].replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{safe_name}.json"
        out_path.write_text(
            json.dumps(session, separators=(",", ":")), encoding="utf-8"
        )
        ok += 1

        if ok % 50 == 0 or ok == 1:
            pts = session["total_points"]
            sc = session["stroke_count"]
            print(
                f"  [{ok}] {session['file']}  "
                f"writer={session['writer_id']}  "
                f"strokes={sc}  points={pts}"
            )

    print(f"\n✅  Converted: {ok}  |  Skipped (empty/bad): {skipped}")
    print(f"   Sessions saved to: {out_dir.resolve()}")
    print(f"\nNext step:")
    print(f"   python test_model.py --sessions_dir {out_dir}")


# ── Verify one file (quick sanity check) ──────────────────────────────────────


def verify_one(original_dir: pathlib.Path):
    """Print a human-readable summary of the first converted session."""
    files = find_xml_files(original_dir)
    if not files:
        print("No files found.")
        return

    s = parse_form_xml(files[0])
    if s is None:
        print("Could not parse first file.")
        return

    print(f"\n=== Verification: {files[0]} ===")
    print(f"  Form ID     : {s['file']}")
    print(f"  Writer ID   : {s['writer_id']}")
    print(f"  Canvas      : {s['canvas_width']} × {s['canvas_height']}")
    print(f"  Strokes     : {s['stroke_count']}")
    print(f"  Total points: {s['total_points']}")
    print(f"  Transcript  : {s['transcription'][:2]}")
    print(f"\n  First 3 points of stroke 0:")
    for pt in s["strokes"][0][:3]:
        print(
            f"    x={pt['x']:.0f}  y={pt['y']:.0f}  t={pt['t']:.1f}ms  pressure={pt['pressure']}"
        )
    print(f"\n  Last 3 points of stroke 0:")
    for pt in s["strokes"][0][-3:]:
        print(f"    x={pt['x']:.0f}  y={pt['y']:.0f}  t={pt['t']:.1f}ms")

    # Check time is relative and sensible
    all_times = [pt["t"] for stroke in s["strokes"] for pt in stroke]
    print(f"\n  Time range  : {min(all_times):.0f}ms → {max(all_times):.0f}ms")
    print(
        f"  (Should start near 0ms and end in tens of seconds — looks good if < 120000ms)"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_dir",
        default=DEFAULT_ORIGINAL,
        help=r"Path to the 'original' folder, e.g. original-xml-part.tar\original",
    )
    parser.add_argument(
        "--out_dir",
        default="data/sessions",
        help="Where to write session JSON files (default: data/sessions)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max forms to convert. 0 = all. Start with 100 to verify.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Just print a summary of the first file and exit.",
    )
    args = parser.parse_args()

    original = pathlib.Path(args.original_dir)

    if not original.exists():
        print(f"ERROR: Path does not exist: {original}")
        print("Common fix: point directly at the 'original' subfolder, e.g.:")
        print(
            r'  --original_dir "./data/original"'
        )
        exit(1)

    if args.verify:
        verify_one(original)
    else:
        convert(
            original_dir=original,
            out_dir=pathlib.Path(args.out_dir),
            limit=args.limit,
        )
