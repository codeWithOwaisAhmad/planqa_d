"""
Stage 4 — PlanQA JSON Layout to Natural Language Description
Converts structured JSON room layout into natural language description
that gets passed to LLMs for spatial reasoning evaluation.
Matches the input format used in the original PlanQA paper.
"""

import json
from pathlib import Path
from typing import Optional


def get_relative_position(x: float, z: float, room_width: float, room_depth: float) -> str:
    """
    Converts absolute coordinates to a human-readable relative position.
    Uses thirds of the room as regions.
    """
    # Horizontal position (left / center / right)
    if x < room_width / 3:
        h_pos = "left side"
    elif x < 2 * room_width / 3:
        h_pos = "center"
    else:
        h_pos = "right side"

    # Depth position (near / middle / far)
    if z < room_depth / 3:
        d_pos = "near the entrance"
    elif z < 2 * room_depth / 3:
        d_pos = "in the middle"
    else:
        d_pos = "far end"

    return f"{h_pos}, {d_pos}"


def layout_to_natural_language(layout: dict) -> str:
    """
    Converts a PlanQA JSON layout dict to a natural language room description.
    Uses metric distances from room origin (door reference point) for
    unambiguous object identification. Matches ground truth protocol.
    """
    room_type = layout["room_type"].replace("_", " ")
    width     = layout["room_dimensions"]["width"]
    depth     = layout["room_dimensions"]["depth"]
    objects   = layout["objects"]

    lines = []
    lines.append(
        f"This is a {room_type} that is approximately "
        f"{width:.1f} meters wide and {depth:.1f} meters deep."
    )
    lines.append(
        f"The reference point is the main door, located at the near-left corner. "
        f"All positions are measured from this door."
    )
    lines.append(f"The room contains {len(objects)} objects:\n")

    for obj in objects:
        # Horizontal direction from door
        if obj["x"] < width / 2:
            h_dir = f"{obj['x']:.1f}m to the right"
        else:
            h_dir = f"{width - obj['x']:.1f}m from the right wall"

        # Depth direction from door
        d_dir = f"{obj['z']:.1f}m from the door"

        lines.append(
            f"- A {obj['category']}: "
            f"{h_dir}, {d_dir}, "
            f"dimensions {obj['width']:.1f}m wide x "
            f"{obj['depth']:.1f}m deep x "
            f"{obj['height']:.1f}m tall."
        )

    return "\n".join(lines)


def convert_all_layouts(
    layouts_dir: Path,
    descriptions_dir: Path,
) -> int:
    """
    Converts all layout JSON files to natural language text files.

    Returns:
        Number of descriptions generated
    """
    layout_files = sorted(layouts_dir.glob("*_layout.json"))

    if not layout_files:
        print(f"No layout files found in {layouts_dir}")
        return 0

    descriptions_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    print(f"Converting {len(layout_files)} layouts to natural language...\n")

    for layout_file in layout_files:
        with open(layout_file, "r") as f:
            layout = json.load(f)

        description = layout_to_natural_language(layout)

        desc_path = descriptions_dir / layout_file.name.replace("_layout.json", "_description.txt")
        with open(desc_path, "w") as f:
            f.write(description)

        converted += 1
        print(f"  OK  {layout['room_id']} — {layout['num_objects']} objects described")

    print(f"\nDone. {converted} descriptions saved to {descriptions_dir}")
    return converted


if __name__ == "__main__":
    BASE_DIR          = Path(__file__).resolve().parent.parent.parent
    LAYOUTS_DIR       = BASE_DIR / "data" / "layouts"
    DESCRIPTIONS_DIR  = BASE_DIR / "data" / "descriptions"

    convert_all_layouts(LAYOUTS_DIR, DESCRIPTIONS_DIR)