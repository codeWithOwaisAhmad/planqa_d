"""
Stage 3 — Detection to PlanQA JSON Conversion
Converts VoteNet raw detection output to PlanQA-compatible JSON layout.
Works on both dummy data and real VoteNet output.
"""

import json
from pathlib import Path
from typing import Optional


CONFIDENCE_THRESHOLD = 0.50

# Maps VoteNet output categories to PlanQA-compatible categories
# Extend this table when you see mismatches on real data
CATEGORY_MAP = {
    "chair":     "chair",
    "table":     "table",
    "desk":      "desk",
    "sofa":      "sofa",
    "bookshelf": "bookshelf",
    "door":      "door",
    "window":    "window",
    "cabinet":   "cabinet",
    "counter":   "counter",
    "bed":       "bed",
    # VoteNet sometimes outputs these — map them to nearest PlanQA category
    "armchair":  "chair",
    "nightstand":"cabinet",
    "dresser":   "cabinet",
    "sink":      "counter",
    "bathtub":   "bed",      # edge case — flag for manual review
}


def convert_detection_to_layout(
    detection: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    min_objects: int = 3,
) -> Optional[dict]:
    """
    Converts a single VoteNet detection dict to PlanQA JSON layout.

    Args:
        detection: Raw VoteNet output dict (from Stage 2)
        confidence_threshold: Minimum confidence to include object
        min_objects: Minimum objects required — rooms below this are flagged

    Returns:
        PlanQA-compatible layout dict, or None if room is unusable
    """
    objects = []
    skipped = 0

    for det in detection["detections"]:
        # Filter by confidence
        if det["confidence"] < confidence_threshold:
            skipped += 1
            continue

        # Map category
        raw_category = det["category"]
        mapped_category = CATEGORY_MAP.get(raw_category, None)

        if mapped_category is None:
            print(f"  WARNING: Unknown category '{raw_category}' in {detection['room_id']} — skipping object")
            skipped += 1
            continue

        objects.append({
            "id":       det["object_id"],
            "category": mapped_category,
            "x":        det["center"]["x"],
            "y":        det["center"]["y"],
            "z":        det["center"]["z"],
            "width":    det["dimensions"]["width"],
            "depth":    det["dimensions"]["depth"],
            "height":   det["dimensions"]["height"],
        })

    # Flag rooms with too few objects — not useful for spatial reasoning questions
    if len(objects) < min_objects:
        print(f"  FLAGGED: {detection['room_id']} has only {len(objects)} objects after filtering — below minimum {min_objects}")
        return None

    return {
        "room_id":         detection["room_id"],
        "room_type":       detection["room_type"],
        "room_dimensions": detection["room_dimensions"],
        "num_objects":     len(objects),
        "objects_skipped": skipped,
        "objects":         objects,
        "source":          detection.get("source", "real_sensor"),
    }


def convert_all_detections(
    detections_dir: Path,
    layouts_dir: Path,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    min_objects: int = 3,
) -> dict:
    """
    Converts all detection JSON files in a directory to PlanQA layout JSONs.

    Returns:
        Summary dict with counts of converted, flagged, and failed rooms
    """
    detection_files = sorted(detections_dir.glob("*_detection.json"))

    if not detection_files:
        print(f"No detection files found in {detections_dir}")
        return {"converted": 0, "flagged": 0, "failed": 0}

    layouts_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    flagged   = 0
    failed    = 0

    print(f"Converting {len(detection_files)} detection files...")
    print(f"Confidence threshold : {confidence_threshold}")
    print(f"Minimum objects      : {min_objects}\n")

    for det_file in detection_files:
        try:
            with open(det_file, "r") as f:
                detection = json.load(f)

            layout = convert_detection_to_layout(
                detection,
                confidence_threshold=confidence_threshold,
                min_objects=min_objects,
            )

            if layout is None:
                flagged += 1
                continue

            # Save layout JSON
            layout_path = layouts_dir / det_file.name.replace("_detection.json", "_layout.json")
            with open(layout_path, "w") as f:
                json.dump(layout, f, indent=2)

            converted += 1
            print(f"  OK  {detection['room_id']} — "
                  f"{layout['num_objects']} objects "
                  f"({layout['objects_skipped']} skipped)")

        except Exception as e:
            print(f"  FAIL {det_file.name} — {e}")
            failed += 1

    print(f"\nSummary:")
    print(f"  Converted : {converted}")
    print(f"  Flagged   : {flagged}  (too few objects — excluded from analysis)")
    print(f"  Failed    : {failed}  (errors)")

    return {"converted": converted, "flagged": flagged, "failed": failed}


if __name__ == "__main__":
    BASE_DIR       = Path(__file__).resolve().parent.parent.parent
    DETECTIONS_DIR = BASE_DIR / "data" / "detections"
    LAYOUTS_DIR    = BASE_DIR / "data" / "layouts"

    summary = convert_all_detections(DETECTIONS_DIR, LAYOUTS_DIR)