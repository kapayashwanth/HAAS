from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.image_loader import resize_if_needed
from src.preprocess import preprocess_image
from src.ocr_reader import extract_text
from src.text_cleaner import clean_text
from src.similarity import calculate_similarity
from src.marks_generator import generate_marks
from src.report_generator import generate_report

app = FastAPI(title="Handwritten Answer Assessment System (HAAS)")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "error": None,
            "model_answer": "",
            "question": "",
            "student_name": "Student",
            "max_marks": 10,
        },
    )


def _decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image. Use JPG/PNG image files.")
    return image


def _run_pipeline(
    image: np.ndarray,
    model_answer: str,
    max_marks: float,
    student_name: str,
    question: str,
    create_pdf: bool,
) -> dict[str, Any]:
    image = resize_if_needed(image, max_dim=2000)
    processed, gray = preprocess_image(image)
    raw_text, detections = extract_text(processed, gray, confidence_threshold=0.30)

    student_clean = clean_text(raw_text)
    model_clean = clean_text(model_answer)

    similarity, breakdown = calculate_similarity(student_clean, model_clean)
    result = generate_marks(
        similarity,
        max_marks=max_marks,
        semantic_score=breakdown.get("semantic", similarity),
    )

    report_path = None
    if create_pdf:
        os.makedirs("reports", exist_ok=True)
        report_path = f"reports/report_web_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        generate_report(
            result=result,
            student_name=student_name,
            question_text=question,
            extracted_text=raw_text,
            model_answer=model_answer,
            score_breakdown=breakdown,
            output_path=report_path,
        )

    return {
        "student_name": student_name,
        "question": question,
        "raw_text": raw_text,
        "cleaned_text": student_clean,
        "model_cleaned": model_clean,
        "breakdown": breakdown,
        "result": asdict(result),
        "detections_count": len(detections),
        "report_path": report_path,
    }


@app.post("/evaluate", response_class=HTMLResponse)
async def evaluate_form(
    request: Request,
    image_file: UploadFile = File(...),
    model_answer: str = Form(...),
    student_name: str = Form("Student"),
    question: str = Form(""),
    max_marks: float = Form(10.0),
    create_pdf: bool = Form(False),
):
    try:
        if not model_answer.strip():
            raise ValueError("Model answer is required.")

        file_bytes = await image_file.read()
        image = _decode_uploaded_image(file_bytes)
        payload = _run_pipeline(
            image=image,
            model_answer=model_answer,
            max_marks=max_marks,
            student_name=student_name,
            question=question,
            create_pdf=create_pdf,
        )

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": payload,
                "error": None,
                "model_answer": model_answer,
                "question": question,
                "student_name": student_name,
                "max_marks": max_marks,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": str(exc),
                "model_answer": model_answer,
                "question": question,
                "student_name": student_name,
                "max_marks": max_marks,
            },
            status_code=400,
        )


@app.post("/api/evaluate")
async def evaluate_api(
    image_file: UploadFile = File(...),
    model_answer: str = Form(...),
    student_name: str = Form("Student"),
    question: str = Form(""),
    max_marks: float = Form(10.0),
    create_pdf: bool = Form(False),
):
    try:
        if not model_answer.strip():
            raise ValueError("model_answer is required")

        file_bytes = await image_file.read()
        image = _decode_uploaded_image(file_bytes)
        payload = _run_pipeline(
            image=image,
            model_answer=model_answer,
            max_marks=max_marks,
            student_name=student_name,
            question=question,
            create_pdf=create_pdf,
        )
        return JSONResponse(payload)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/health")
def health():
    return {"status": "ok", "app": "HAAS"}
