"""
main.py
───────
Entry point for the Handwritten Answer Evaluation System.

Usage:
    python main.py              → Launch full GUI
    python main.py --cli        → Command-line pipeline (headless)
    python main.py --test       → Quick self-test (no image needed)
"""

import sys
import os
import logging
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger("main")


# ── CLI / headless pipeline ───────────────────────────────────────────────────

def run_cli(image_path: str, model_answer_path: str, max_marks: float = 10.0,
            student_name: str = "Student", question: str = ""):
    """Run the full evaluation pipeline without a GUI."""
    from src.image_loader    import load_image, resize_if_needed
    from src.preprocess      import preprocess_image
    from src.ocr_reader      import extract_text
    from src.text_cleaner    import clean_text
    from src.similarity      import calculate_similarity
    from src.marks_generator import generate_marks
    from src.report_generator import generate_report

    logger.info("=== Handwritten Answer Evaluation System ===")
    logger.info("Image       : %s", image_path)
    logger.info("Model ans.  : %s", model_answer_path)

    # Load model answer
    with open(model_answer_path, "r", encoding="utf-8") as f:
        model_answer = f.read()

    # Pipeline
    image    = load_image(image_path)
    image    = resize_if_needed(image, max_dim=2000)
    proc, gr = preprocess_image(image)
    raw, _   = extract_text(proc, gr, confidence_threshold=0.30)

    print("\n── Extracted Text ─────────────────────────────────────────")
    print(raw)

    student_clean = clean_text(raw)
    model_clean   = clean_text(model_answer)

    similarity, breakdown = calculate_similarity(student_clean, model_clean)
    result = generate_marks(
        similarity,
        max_marks=max_marks,
        semantic_score=breakdown.get("semantic", similarity),
    )

    print("\n── Results ────────────────────────────────────────────────")
    print(f"  Semantic Similarity  : {round(breakdown.get('semantic',0)*100,2)}%")
    print(f"  Semantic Marks       : {result.semantic_marks} / {result.max_marks}")
    print(f"  Token Sort           : {round(breakdown.get('token_sort',0)*100,2)}%")
    print(f"  Partial Match        : {round(breakdown.get('partial',0)*100,2)}%")
    print(f"  Jaccard              : {round(breakdown.get('jaccard',0)*100,2)}%")
    print(f"\n  ▶ Final Score        : {result.percentage}%")
    print(f"  ▶ Marks Awarded      : {result.marks} / {result.max_marks}")
    print(f"  ▶ Grade              : {result.grade}")
    print(f"  ▶ Feedback           : {result.feedback}")
    print("────────────────────────────────────────────────────────────\n")

    # Save report
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    generate_report(
        result=result,
        student_name=student_name,
        question_text=question,
        extracted_text=raw,
        model_answer=model_answer,
        score_breakdown=breakdown,
        output_path=report_path,
    )
    print(f"  📄 PDF Report saved  : {report_path}\n")


def run_test():
    """Quick self-test using synthetic text (no image required)."""
    from src.text_cleaner    import clean_text
    from src.similarity      import calculate_similarity
    from src.marks_generator import generate_marks

    student = "Photosynthesis is how plants make food using sunlight carbon dioxide and water"
    model   = ("Photosynthesis is the process by which green plants use sunlight, "
               "carbon dioxide and water to produce glucose and oxygen.")

    sc = clean_text(student)
    mc = clean_text(model)

    sim, breakdown = calculate_similarity(sc, mc)
    result = generate_marks(sim, max_marks=10, semantic_score=breakdown.get("semantic", sim))

    print("\n── Self-Test Results ───────────────────────────────────────")
    for k, v in breakdown.items():
        print(f"  {k:15s}: {round(v*100,2)}%")
    print(f"\n  Final Score  : {result.percentage}%")
    print(f"  Marks        : {result.marks} / {result.max_marks}")
    print(f"  Grade        : {result.grade}")
    print(f"  Feedback     : {result.feedback}")
    print("────────────────────────────────────────────────────────────\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--test" in args:
        run_test()

    elif "--cli" in args:
        # python main.py --cli <image_path> <model_answer_path> [max_marks] [student_name]
        if len(args) < 3:
            print("Usage: python main.py --cli <image_path> <model_answer.txt> [max_marks] [student_name]")
            sys.exit(1)
        image_path   = args[1]
        model_path   = args[2]
        max_marks    = float(args[3]) if len(args) > 3 else 10.0
        student_name = args[4] if len(args) > 4 else "Student"
        run_cli(image_path, model_path, max_marks, student_name)

    else:
        # Default: launch GUI
        from src.gui import run
        run()