"""
Stage 5b — LLM Query Engine
Queries Gemini with spatial reasoning questions.
Loads room description + question, returns model answer.
Saves raw responses for scoring in Stage 6.
"""

import json
import time
import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
# NEW — replace with this
from google import genai
from google.genai import types
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are evaluating spatial reasoning about indoor room layouts.
You will be given a description of a room and a question about the spatial relationships between objects.
Answer with the shortest possible response — a single word, number, or object name.
Do not explain your reasoning. Do not add punctuation. Just the answer."""


def query_gemini(description: str, question: str, retries: int = 3) -> str:
    prompt = f"{description}\n\nQuestion: {question}\nAnswer:"
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=50,
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            return response.text.strip().lower()
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt + 1}/{retries} — {e}")
                time.sleep(2)
            else:
                return f"ERROR: {e}"


def run_evaluation(
    questions_path: Path,
    descriptions_dir: Path,
    raw_outputs_dir: Path,
    model_name: str = MODEL_NAME,
    max_questions: int = None,   # Set to small number for pilot testing
) -> Path:
    """
    Runs full LLM evaluation on all questions.
    Saves raw outputs to CSV for Stage 6 scoring.

    Args:
        max_questions: If set, only evaluate this many questions (pilot mode)

    Returns:
        Path to raw outputs CSV
    """
    import pandas as pd

    # Load questions
    questions = []
    with open(questions_path) as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    if max_questions:
        questions = questions[:max_questions]
        print(f"PILOT MODE — evaluating {max_questions} questions only\n")

    raw_outputs_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Model          : {model_name}")
    print(f"Total questions: {len(questions)}")
    print(f"Temperature    : 0.0 (deterministic)\n")

    for q in tqdm(questions, desc="Querying LLM"):
        room_id = q["room_id"]

        # Load description for this room
        desc_path = descriptions_dir / f"{room_id}_description.txt"
        if not desc_path.exists():
            print(f"  WARNING: No description found for {room_id} — skipping")
            continue

        with open(desc_path) as f:
            description = f.read()

        # Query model
        model_answer = query_gemini(description, q["question"])

        results.append({
            "room_id":       room_id,
            "question_index": q["question_index"],
            "question_type": q["question_type"],
            "question":      q["question"],
            "correct_answer": q["answer"],
            "model_answer":  model_answer,
            "model":         model_name,
        })

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Save raw outputs
    output_path = raw_outputs_dir / f"raw_outputs_{model_name.replace('-', '_')}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"\nRaw outputs saved to {output_path}")
    print(f"Total responses: {len(results)}")
    return output_path


if __name__ == "__main__":
    BASE_DIR         = Path(__file__).resolve().parent.parent.parent
    QUESTIONS_PATH   = BASE_DIR / "data" / "questions" / "all_questions.jsonl"
    DESCRIPTIONS_DIR = BASE_DIR / "data" / "descriptions"
    RAW_OUTPUTS_DIR  = BASE_DIR / "results" / "raw_outputs"

    # PILOT: only 10 questions to verify pipeline before full run
    run_evaluation(
        QUESTIONS_PATH,
        DESCRIPTIONS_DIR,
        RAW_OUTPUTS_DIR,
        max_questions=10,
    )