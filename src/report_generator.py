"""PDF report generation utilities using ReportLab."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _safe(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _fmt_pct(value: float) -> str:
    return f"{round(float(value) * 100, 2)}%"


def _extract_result_fields(result: Any) -> dict[str, str]:
    if result is None:
        return {
            "marks": "-",
            "max_marks": "-",
            "semantic_score": "-",
            "semantic_marks": "-",
            "percentage": "-",
            "grade": "-",
            "feedback": "-",
        }

    return {
        "marks": _safe(getattr(result, "marks", "-")),
        "max_marks": _safe(getattr(result, "max_marks", "-")),
        "semantic_score": _safe(getattr(result, "semantic_score", "-")),
        "semantic_marks": _safe(getattr(result, "semantic_marks", "-")),
        "percentage": _safe(getattr(result, "percentage", "-")),
        "grade": _safe(getattr(result, "grade", "-")),
        "feedback": _safe(getattr(result, "feedback", "-")),
    }


def generate_report(
    result=None,
    student_name: str = "Student",
    question_text: str = "",
    extracted_text: str = "",
    model_answer: str = "",
    score_breakdown: dict | None = None,
    output_path: str = "reports/evaluation_report.pdf",
    data: dict | None = None,
) -> str:
    """Generate a PDF evaluation report and return its saved path.

    Supported call styles:
      1) New style (used by GUI/CLI): keyword args shown in signature.
      2) Legacy style: ``generate_report(data=<dict>, output_path="...")``.
    """
    if data and not result:
        result = data.get("result")
        student_name = data.get("student_name", student_name)
        question_text = data.get("question_text", question_text)
        extracted_text = data.get("extracted_text", extracted_text)
        model_answer = data.get("model_answer", model_answer)
        score_breakdown = data.get("score_breakdown", score_breakdown)

    breakdown = score_breakdown or {}
    fields = _extract_result_fields(result)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    styles = getSampleStyleSheet()
    title = styles["Title"]
    heading = styles["Heading3"]
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Handwritten Answer Evaluation Report", title))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))
    story.append(Spacer(1, 10))

    meta_table = Table(
        [
            ["Student", _safe(student_name)],
            ["Question", _safe(question_text, "(Not provided)")],
            ["Marks", f"{fields['marks']} / {fields['max_marks']}"],
            [
                "Semantic",
                f"{round(float(fields['semantic_score']) * 100, 2)}% "
                f"({fields['semantic_marks']} / {fields['max_marks']})"
                if fields["semantic_score"] != "-" else "-",
            ],
            ["Score", f"{fields['percentage']}%"],
            ["Grade", fields["grade"]],
        ],
        colWidths=[35 * mm, 140 * mm],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F4F7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Similarity Breakdown", heading))
    breakdown_rows = [["Metric", "Value"]]
    for key in ("semantic", "token_sort", "partial", "jaccard"):
        value = breakdown.get(key)
        breakdown_rows.append([key.replace("_", " ").title(), _fmt_pct(value) if value is not None else "-"])

    breakdown_table = Table(breakdown_rows, colWidths=[70 * mm, 40 * mm])
    breakdown_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6F4FF")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5DD")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(breakdown_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Feedback", heading))
    story.append(Paragraph(_safe(fields["feedback"]), body))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Extracted Answer (OCR)", heading))
    story.append(Paragraph(_safe(extracted_text).replace("\n", "<br/>"), body))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Model Answer", heading))
    story.append(Paragraph(_safe(model_answer).replace("\n", "<br/>"), body))

    doc.build(story)
    return output_path
