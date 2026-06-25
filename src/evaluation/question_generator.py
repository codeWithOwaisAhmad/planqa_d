"""
Stage 5a — Question Generator
Generates PlanQA-style spatial reasoning questions from layout JSON.
Produces exactly 8 questions per room: 2 per question type.
Only generates questions about objects present in the layout.
Every question has exactly one verifiable correct answer.
"""

import json
import random
from pathlib import Path
from itertools import combinations
from typing import Optional

random.seed(42)

QUESTION_TYPES = ["distance_comparison", "direction", "existence", "count"]


def euclidean_distance(obj1: dict, obj2: dict) -> float:
    return ((obj1["x"] - obj2["x"]) ** 2 + (obj1["z"] - obj2["z"]) ** 2) ** 0.5


def get_direction(from_obj: dict, to_obj: dict) -> str:
    dx = to_obj["x"] - from_obj["x"]
    dz = to_obj["z"] - from_obj["z"]
    if abs(dx) >= abs(dz):
        return "to the right of" if dx > 0 else "to the left of"
    else:
        return "farther from the door than" if dz > 0 else "closer to the door than"


def generate_distance_comparison(objects: list, room_id: str) -> Optional[dict]:
    """
    'Is the [A] closer to the [B] or to the [C]?'
    Requires at least 3 distinct object categories.
    """
    if len(objects) < 3:
        return None

    # Pick 3 distinct objects
    triple = random.sample(objects, 3)
    a, b, c = triple

    dist_ab = euclidean_distance(a, b)
    dist_ac = euclidean_distance(a, c)

    if abs(dist_ab - dist_ac) < 0.3:
        return None  # Too close to call — skip

    closer = b if dist_ab < dist_ac else c
    answer = f"{closer['category']} (id {closer['id']})"


    # Disambiguate if same category appears multiple times
    b_label = f"{b['category']} (id {b['id']})" if b["category"] == c["category"] else b["category"]
    c_label = f"{c['category']} (id {c['id']})" if b["category"] == c["category"] else c["category"]

    return {
        "room_id":       room_id,
        "question_type": "distance_comparison",
        "question":      f"Is the {a['category']} closer to the {b_label} or to the {c_label}?",
        "answer":        answer,
        "answer_id":     closer["id"],
    }


def generate_direction(objects: list, room_id: str) -> Optional[dict]:
    """
    'What object is to the [direction] of the [A]?'
    """
    if len(objects) < 2:
        return None

    random.shuffle(objects)
    for ref in objects:
        candidates = [o for o in objects if o["id"] != ref["id"]]
        random.shuffle(candidates)
        for target in candidates:
            direction = get_direction(ref, target)
            return {
                "room_id":       room_id,
                "question_type": "direction",
                "question":      f"What object is {direction} the {ref['category']}?",
                "answer":        target["category"],
                "answer_id":     target["id"],
            }
    return None


def generate_existence(objects: list, room_id: str, all_categories: list) -> dict:
    """
    'Is there a [category] in this room?' — mix of yes and no answers.
    """
    present_cats    = list({o["category"] for o in objects})
    all_cats        = list(set(all_categories))
    absent_cats     = [c for c in all_cats if c not in present_cats]

    # Alternate yes/no
    if random.random() < 0.5 and present_cats:
        cat    = random.choice(present_cats)
        answer = "yes"
    elif absent_cats:
        cat    = random.choice(absent_cats)
        answer = "no"
    else:
        cat    = random.choice(present_cats)
        answer = "yes"

    return {
        "room_id":       room_id,
        "question_type": "existence",
        "question":      f"Is there a {cat} in this room?",
        "answer":        answer,
        "answer_id":     None,
    }


def generate_count(objects: list, room_id: str, exclude_category: str = None) -> dict:
    from collections import Counter
    cat_counts = Counter(o["category"] for o in objects)
    # Pick from top 2 most frequent, excluding already-used category
    candidates = [(cat, cnt) for cat, cnt in cat_counts.most_common(3) 
                  if cat != exclude_category]
    if not candidates:
        candidates = list(cat_counts.most_common(1))
    category, count = candidates[0]
    return {
        "room_id":       room_id,
        "question_type": "count",
        "question":      f"How many {category}s are in this room?",
        "answer":        str(count),
        "answer_id":     None,
    }


def generate_questions_for_room(layout: dict, all_categories: list) -> list[dict]:
    objects   = layout["objects"]
    room_id   = layout["room_id"]
    questions = []
    seen_questions = set()  # prevent duplicates

    generators = {
        "distance_comparison": lambda: generate_distance_comparison(objects, room_id),
        "direction":           lambda: generate_direction(objects, room_id),
        "existence":           lambda: generate_existence(objects, room_id, all_categories),
        "count":               lambda: generate_count(objects, room_id),
    }

    for q_type, gen_fn in generators.items():
        count    = 0
        attempts = 0
        while count < 2 and attempts < 20:
            q = gen_fn()
            if q is not None and q["question"] not in seen_questions:
                q["question_index"] = len(questions) + 1
                questions.append(q)
                seen_questions.add(q["question"])
                count += 1
            attempts += 1

    return questions


def generate_all_questions(layouts_dir: Path, questions_dir: Path) -> int:
    """
    Generates questions for all layout files.
    Saves one JSON per room and one combined JSONL file.
    """
    layout_files = sorted(layouts_dir.glob("*_layout.json"))

    if not layout_files:
        print(f"No layout files found in {layouts_dir}")
        return 0

    # Collect all categories across all rooms for existence questions
    all_categories = []
    all_layouts    = []
    for lf in layout_files:
        with open(lf) as f:
            layout = json.load(f)
        all_layouts.append(layout)
        all_categories.extend(o["category"] for o in layout["objects"])

    questions_dir.mkdir(parents=True, exist_ok=True)
    all_questions = []
    total = 0

    print(f"Generating questions for {len(layout_files)} rooms...\n")

    for layout in all_layouts:
        questions = generate_questions_for_room(layout, all_categories)

        # Save per-room
        room_q_path = questions_dir / f"{layout['room_id']}_questions.json"
        with open(room_q_path, "w") as f:
            json.dump(questions, f, indent=2)

        all_questions.extend(questions)
        total += len(questions)
        print(f"  {layout['room_id']} — {len(questions)} questions generated")

    # Save combined file for experiment runner
    combined_path = questions_dir / "all_questions.jsonl"
    with open(combined_path, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")

    print(f"\nDone. {total} total questions across {len(layout_files)} rooms")
    print(f"Combined file: {combined_path}")
    return total


if __name__ == "__main__":
    BASE_DIR      = Path(__file__).resolve().parent.parent.parent
    LAYOUTS_DIR   = BASE_DIR / "data" / "layouts"
    QUESTIONS_DIR = BASE_DIR / "data" / "questions"

    generate_all_questions(LAYOUTS_DIR, QUESTIONS_DIR)