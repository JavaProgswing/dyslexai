"""
Step 1 — Inspect IAM dataset before doing anything else.
Run this first to confirm which tar contains stroke data and what the XML looks like.

Usage:
    python inspect_iam.py
    python inspect_iam.py --data_dir ./data
"""

import pathlib, argparse, xml.etree.ElementTree as ET


def inspect_dir(root: pathlib.Path):
    print(f"\n=== Scanning: {root} ===")
    for tar_dir in sorted(root.iterdir()):
        if not tar_dir.is_dir():
            continue
        xml_files = list(tar_dir.rglob("*.xml"))
        png_files = list(tar_dir.rglob("*.png"))
        txt_files = list(tar_dir.rglob("*.txt"))
        print(f"\n  {tar_dir.name}/")
        print(f"    XML : {len(xml_files)}")
        print(f"    PNG : {len(png_files)}")
        print(f"    TXT : {len(txt_files)}")

        if xml_files:
            print(f"    → Peeking at: {xml_files[0]}")
            inspect_xml(xml_files[0])


def inspect_xml(path: pathlib.Path):
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        print(f"      Root tag    : <{root.tag}>")
        print(f"      Root attribs: {dict(root.attrib)}")

        # Walk first 3 levels
        for child in list(root)[:4]:
            print(f"      ├─ <{child.tag}> attribs={dict(child.attrib)}")
            for gchild in list(child)[:3]:
                print(f"      │   ├─ <{gchild.tag}> attribs={dict(gchild.attrib)}")
                for ggchild in list(gchild)[:2]:
                    print(
                        f"      │   │   └─ <{ggchild.tag}> attribs={dict(ggchild.attrib)}"
                    )

        # Check if it has Point elements (stroke data)
        points = root.findall(".//Point")
        strokes = root.findall(".//Stroke")
        print(f"      Stroke count : {len(strokes)}")
        print(f"      Point count  : {len(points)}")
        if points:
            p = points[0]
            print(f"      First point  : {dict(p.attrib)}")
            print(f"      ✅ HAS STROKE DATA — this is ONLINE handwriting (x, y, time)")
        else:
            print(f"      ❌ No <Point> elements — this is OFFLINE only (image/label)")

    except ET.ParseError as e:
        print(f"      Parse error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=r"./data/original",
    )
    args = parser.parse_args()
    inspect_dir(pathlib.Path(args.data_dir))
