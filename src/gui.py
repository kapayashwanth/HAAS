"""
gui.py
──────
Professional Tkinter GUI for the Handwritten Answer Evaluation System.

Layout:
  ┌─────────────────────────────────────────┐
  │  HEADER (title + subtitle)              │
  ├─────────────────────────────────────────┤
  │  LEFT PANEL          │  RIGHT PANEL      │
  │  • Upload image      │  • Results        │
  │  • Image preview     │  • Score bars     │
  │  • Model answer      │  • Feedback       │
  │  • Settings          │  • Buttons        │
  └─────────────────────────────────────────┘
  │  LOG / STATUS BAR                        │
  └─────────────────────────────────────────┘
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import os
import importlib
from pathlib import Path
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG        = "#0F1923"    # Dark navy background
SURFACE   = "#1A2638"    # Card surface
ACCENT    = "#00B4D8"    # Cyan accent
ACCENT2   = "#0077B6"    # Darker cyan
SUCCESS   = "#52B788"    # Green
WARNING   = "#F4A261"    # Orange
DANGER    = "#E63946"    # Red
TEXT      = "#E0E0E0"    # Primary text
SUBTEXT   = "#90A4AE"    # Secondary text
WHITE     = "#FFFFFF"
FONT      = "Segoe UI"


class EvaluationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Answer Evaluation System")
        self.configure(bg=BG)
        self.geometry("1280x820")
        self.minsize(1100, 700)
        self.resizable(True, True)

        # State
        self._image_path: str | None = None
        self._image_tk = None
        self._result = None
        self._breakdown = None
        self._is_evaluating = False

        self._build_ui()
        self._log("System ready. Select an answer sheet to begin.")

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_main_area()
        self._build_status_bar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=ACCENT2, height=64)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="✦ Handwritten Answer Evaluation System",
                 bg=ACCENT2, fg=WHITE, font=(FONT, 16, "bold")).pack(side="left", padx=20, pady=14)
        tk.Label(hdr, text="AI-Powered OCR + NLP Grading",
                 bg=ACCENT2, fg="#B2EBF2", font=(FONT, 10)).pack(side="left", padx=4)

    def _build_main_area(self):
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=(12, 0))

        # Left panel (40%)
        left = tk.Frame(main, bg=SURFACE, bd=0, relief="flat")
        left.pack(side="left", fill="both", expand=False, ipadx=8, ipady=8,
                  padx=(0, 8))
        left.configure(width=480)

        # Right panel (60%)
        right = tk.Frame(main, bg=SURFACE, bd=0, relief="flat")
        right.pack(side="left", fill="both", expand=True, ipady=8)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        self._section_label(parent, "INPUT")

        # Upload button
        btn_row = tk.Frame(parent, bg=SURFACE)
        btn_row.pack(fill="x", padx=12, pady=(6, 4))

        self._btn_upload = self._accent_button(btn_row, "📂  Select Answer Sheet", self._select_image)
        self._btn_upload.pack(side="left", ipadx=10, ipady=4)

        self._lbl_filename = tk.Label(btn_row, text="No file selected", bg=SURFACE,
                                      fg=SUBTEXT, font=(FONT, 9))
        self._lbl_filename.pack(side="left", padx=10)

        # Image preview
        preview_frame = tk.LabelFrame(parent, text="Image Preview", bg=SURFACE,
                                       fg=SUBTEXT, font=(FONT, 9), bd=1, relief="solid")
        preview_frame.pack(fill="x", padx=12, pady=4)

        self._canvas = tk.Canvas(preview_frame, bg="#0A1218", width=420, height=220,
                                  highlightthickness=0)
        self._canvas.pack(padx=6, pady=6)
        self._canvas.create_text(210, 110, text="No image loaded",
                                  fill=SUBTEXT, font=(FONT, 10))

        # Settings
        settings_frame = tk.LabelFrame(parent, text="Settings", bg=SURFACE,
                                        fg=SUBTEXT, font=(FONT, 9), bd=1, relief="solid")
        settings_frame.pack(fill="x", padx=12, pady=4)

        self._var_max_marks = self._labeled_spinbox(settings_frame, "Max Marks:", 10, 1, 100)
        self._var_student   = self._labeled_entry(settings_frame, "Student Name:", "Student")
        self._var_question  = self._labeled_entry(settings_frame, "Question:", "")

        # Model answer
        ma_frame = tk.LabelFrame(parent, text="Model Answer", bg=SURFACE,
                                  fg=SUBTEXT, font=(FONT, 9), bd=1, relief="solid")
        ma_frame.pack(fill="both", expand=True, padx=12, pady=4)

        self._txt_model = scrolledtext.ScrolledText(
            ma_frame, height=5, bg="#0A1218", fg=TEXT,
            font=(FONT, 9), insertbackground=ACCENT, relief="flat", wrap="word"
        )
        self._txt_model.pack(fill="both", expand=True, padx=4, pady=4)
        self._txt_model.insert("1.0",
            "Photosynthesis is the process by which green plants use sunlight, "
            "carbon dioxide and water to produce glucose and oxygen.")

        # Evaluate button
        self._btn_eval = self._accent_button(
            parent, "▶   EVALUATE", self._start_evaluation,
            bg=SUCCESS, active_bg="#3B9B6E", pady=8,
        )
        self._btn_eval.pack(fill="x", padx=12, pady=(6, 4), ipady=6)

    def _build_right_panel(self, parent):
        self._section_label(parent, "RESULTS")

        # Score cards row
        cards_row = tk.Frame(parent, bg=SURFACE)
        cards_row.pack(fill="x", padx=12, pady=(4, 0))

        self._card_marks   = self._score_card(cards_row, "MARKS",      "—  /  —")
        self._card_grade   = self._score_card(cards_row, "GRADE",      "—")
        self._card_percent = self._score_card(cards_row, "SCORE",      "— %")
        self._card_semantic= self._score_card(cards_row, "SEMANTIC",   "— %")

        for card in (self._card_marks, self._card_grade,
                     self._card_percent, self._card_semantic):
            card.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # Score bars
        bars_frame = tk.LabelFrame(parent, text="Similarity Breakdown", bg=SURFACE,
                                    fg=SUBTEXT, font=(FONT, 9), bd=1, relief="solid")
        bars_frame.pack(fill="x", padx=12, pady=4)

        self._bars: dict[str, ttk.Progressbar] = {}
        self._bar_labels: dict[str, tk.Label] = {}
        metrics = [
            ("semantic",   "Semantic (55%)"),
            ("token_sort", "Token Sort (25%)"),
            ("partial",    "Partial Match (10%)"),
            ("jaccard",    "Jaccard (10%)"),
        ]
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Teal.Horizontal.TProgressbar",
                         troughcolor="#0A1218", background=ACCENT,
                         lightcolor=ACCENT, darkcolor=ACCENT2, bordercolor=SURFACE)

        for key, label in metrics:
            row = tk.Frame(bars_frame, bg=SURFACE)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=label, bg=SURFACE, fg=TEXT,
                     font=(FONT, 9), width=22, anchor="w").pack(side="left")
            pb = ttk.Progressbar(row, style="Teal.Horizontal.TProgressbar",
                                  length=220, mode="determinate")
            pb.pack(side="left", padx=6)
            lbl = tk.Label(row, text="0%", bg=SURFACE, fg=ACCENT,
                           font=(FONT, 9, "bold"), width=5)
            lbl.pack(side="left")
            self._bars[key] = pb
            self._bar_labels[key] = lbl

        # Extracted text
        ocr_frame = tk.LabelFrame(parent, text="Extracted Text (OCR Output)", bg=SURFACE,
                                   fg=SUBTEXT, font=(FONT, 9), bd=1, relief="solid")
        ocr_frame.pack(fill="both", expand=True, padx=12, pady=4)

        self._txt_extracted = scrolledtext.ScrolledText(
            ocr_frame, height=6, bg="#0A1218", fg=TEXT,
            font=("Consolas", 9), insertbackground=ACCENT, relief="flat", wrap="word",
        )
        self._txt_extracted.pack(fill="both", expand=True, padx=4, pady=4)

        # Feedback box
        fb_frame = tk.LabelFrame(parent, text="Feedback", bg=SURFACE,
                                  fg=SUBTEXT, font=(FONT, 9), bd=1, relief="solid")
        fb_frame.pack(fill="x", padx=12, pady=4)

        self._lbl_feedback = tk.Label(fb_frame, text="—", bg=SURFACE, fg=WARNING,
                                       font=(FONT, 10), wraplength=550, justify="left")
        self._lbl_feedback.pack(anchor="w", padx=8, pady=6)

        # Action buttons
        action_row = tk.Frame(parent, bg=SURFACE)
        action_row.pack(fill="x", padx=12, pady=(2, 8))

        self._btn_report = self._accent_button(action_row, "📄  Export PDF Report",
                                                self._export_report,
                                                bg=ACCENT2, active_bg="#005F8E")
        self._btn_report.pack(side="left", ipadx=8, ipady=4, padx=(0, 8))
        self._btn_report.config(state="disabled")

        self._btn_clear = self._accent_button(action_row, "🗑  Clear",
                                               self._clear_all,
                                               bg="#37474F", active_bg="#263238")
        self._btn_clear.pack(side="left", ipadx=8, ipady=4)

    def _build_status_bar(self):
        bar = tk.Frame(self, bg="#0A1218", height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self._status_var, bg="#0A1218", fg=SUBTEXT,
                 font=(FONT, 9), anchor="w").pack(side="left", padx=12)

        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=200)
        self._progress.pack(side="right", padx=12, pady=4)

    # ── Helper widgets ─────────────────────────────────────────────────────────

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, bg=SURFACE, fg=ACCENT,
                 font=(FONT, 8, "bold")).pack(anchor="w", padx=14, pady=(8, 2))

    def _accent_button(self, parent, text, command, bg=None, active_bg=None, pady=4):
        bg = bg or ACCENT2
        active_bg = active_bg or "#005F8E"
        return tk.Button(parent, text=text, command=command,
                         bg=bg, fg=WHITE, activebackground=active_bg,
                         activeforeground=WHITE, font=(FONT, 10, "bold"),
                         relief="flat", cursor="hand2", pady=pady)

    def _score_card(self, parent, label, value):
        frame = tk.Frame(parent, bg="#0A1218", bd=0)
        tk.Label(frame, text=label, bg="#0A1218", fg=SUBTEXT,
                 font=(FONT, 8)).pack(pady=(6, 0))
        val_lbl = tk.Label(frame, text=value, bg="#0A1218", fg=ACCENT,
                            font=(FONT, 14, "bold"))
        val_lbl.pack(pady=(0, 6))
        frame._val_label = val_lbl
        return frame

    def _labeled_spinbox(self, parent, label, default, from_, to):
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=label, bg=SURFACE, fg=TEXT,
                 font=(FONT, 9), width=16, anchor="w").pack(side="left")
        var = tk.StringVar(value=str(default))
        tk.Spinbox(row, from_=from_, to=to, textvariable=var,
                   bg="#0A1218", fg=TEXT, font=(FONT, 9),
                   buttonbackground=SURFACE, relief="flat", width=8).pack(side="left")
        return var

    def _labeled_entry(self, parent, label, default):
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=label, bg=SURFACE, fg=TEXT,
                 font=(FONT, 9), width=16, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        tk.Entry(row, textvariable=var, bg="#0A1218", fg=TEXT,
                 font=(FONT, 9), relief="flat", insertbackground=ACCENT).pack(side="left", fill="x", expand=True)
        return var

    # ── Event Handlers ─────────────────────────────────────────────────────────

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="Select Handwritten Answer Sheet",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                       ("All Files", "*.*")],
        )
        if not path:
            return
        self._image_path = path
        self._lbl_filename.config(text=Path(path).name)
        self._load_preview(path)
        self._log(f"Loaded: {path}")

    def _load_preview(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((420, 220))
            self._image_tk = ImageTk.PhotoImage(img)
            self._canvas.delete("all")
            self._canvas.create_image(210, 110, image=self._image_tk, anchor="center")
        except Exception as e:
            self._log(f"Preview error: {e}")

    def _start_evaluation(self):
        if self._is_evaluating:
            return
        if not self._image_path:
            messagebox.showwarning("No Image", "Please select an answer sheet image first.")
            return
        model_answer = self._txt_model.get("1.0", "end").strip()
        if not model_answer:
            messagebox.showwarning("No Model Answer", "Please enter a model answer.")
            return

        self._is_evaluating = True
        self._btn_eval.config(state="disabled", text="⏳  Evaluating…")
        self._progress.start(12)
        self._status("Running evaluation pipeline…")

        threading.Thread(target=self._run_pipeline,
                         args=(self._image_path, model_answer), daemon=True).start()

    def _run_pipeline(self, image_path, model_answer):
        try:
            from src.image_loader     import load_image, resize_if_needed
            from src.preprocess       import preprocess_image
            from src.ocr_reader       import extract_text
            from src.text_cleaner     import clean_text
            from src.similarity       import calculate_similarity, keyword_coverage
            from src.marks_generator  import generate_marks

            self._status("Loading image…")
            image = load_image(image_path)
            image = resize_if_needed(image, max_dim=2000)

            self._status("Preprocessing image…")
            processed, gray = preprocess_image(image)

            self._status("Extracting text (OCR)…")
            raw_text, detections = extract_text(processed, gray, confidence_threshold=0.30)

            self._status("Cleaning text…")
            student_clean = clean_text(raw_text)
            model_clean   = clean_text(model_answer)

            self._status("Calculating similarity…")
            similarity, breakdown = calculate_similarity(student_clean, model_clean)

            self._status("Generating marks…")
            max_marks = float(self._var_max_marks.get() or 10)
            result = generate_marks(
                similarity,
                max_marks=max_marks,
                semantic_score=breakdown.get("semantic", similarity),
            )

            self._result    = result
            self._breakdown = breakdown
            self._raw_text  = raw_text
            self._model_answer_str = model_answer

            self.after(0, self._display_results, result, breakdown, raw_text)

        except Exception as e:
            logger.exception("Pipeline error")
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, self._reset_eval_state)

    def _display_results(self, result, breakdown, raw_text):
        # Cards
        grade_color = SUCCESS if result.grade in ("A+","A") else (WARNING if result.grade in ("B","C") else DANGER)
        self._card_marks._val_label.config(
            text=f"{result.marks}  /  {result.max_marks}", fg=grade_color)
        self._card_grade._val_label.config(text=result.grade, fg=grade_color)
        self._card_percent._val_label.config(
            text=f"{result.percentage}%", fg=grade_color)
        sem = getattr(result, "semantic_score", breakdown.get("semantic", 0))
        self._card_semantic._val_label.config(
            text=f"{round(sem*100,1)}%", fg=ACCENT)

        # Bars
        for key, pb in self._bars.items():
            val = breakdown.get(key, 0)
            pct = round(val * 100, 1)
            pb["value"] = pct
            self._bar_labels[key].config(text=f"{pct}%")

        # OCR text
        self._txt_extracted.delete("1.0", "end")
        self._txt_extracted.insert("1.0", raw_text)

        # Feedback
        self._lbl_feedback.config(
            text=(
                f"{result.feedback}\n"
                f"Semantic Score: {round(sem * 100, 2)}%"
                f" ({getattr(result, 'semantic_marks', round(sem * result.max_marks, 2))}/{result.max_marks})"
            )
        )

        self._btn_report.config(state="normal")
        self._reset_eval_state()
        self._status(
            f"✔  Evaluation complete — {result.marks}/{result.max_marks} marks "
            f"({result.percentage}%) | Grade: {result.grade}"
        )
        self._log(f"Result: {result.marks}/{result.max_marks} | Grade: {result.grade} | Feedback: {result.feedback}")

    def _export_report(self):
        if not self._result:
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
            initialfile="evaluation_report.pdf",
        )
        if not save_path:
            return
        try:
            report_mod = importlib.import_module("src.report_generator")
            report_mod = importlib.reload(report_mod)
            generate_report = report_mod.generate_report

            payload = {
                "result": self._result,
                "student_name": self._var_student.get() or "Student",
                "question_text": self._var_question.get() or "",
                "extracted_text": getattr(self, "_raw_text", ""),
                "model_answer": getattr(self, "_model_answer_str", ""),
                "score_breakdown": self._breakdown or {},
            }

            try:
                generate_report(output_path=save_path, **payload)
            except TypeError:
                # Backward compatibility with older placeholder signature.
                generate_report(payload, save_path)

            messagebox.showinfo("Saved", f"Report saved to:\n{save_path}")
            self._log(f"PDF report saved: {save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _clear_all(self):
        self._image_path = None
        self._result = None
        self._breakdown = None
        self._lbl_filename.config(text="No file selected")
        self._canvas.delete("all")
        self._canvas.create_text(210, 110, text="No image loaded",
                                  fill=SUBTEXT, font=(FONT, 10))
        self._txt_extracted.delete("1.0", "end")
        self._lbl_feedback.config(text="—")
        for card in (self._card_marks, self._card_grade,
                     self._card_percent, self._card_semantic):
            card._val_label.config(text="—", fg=ACCENT)
        for pb in self._bars.values():
            pb["value"] = 0
        for lbl in self._bar_labels.values():
            lbl.config(text="0%")
        self._btn_report.config(state="disabled")
        self._status("Cleared. Ready for next evaluation.")

    def _reset_eval_state(self):
        self._is_evaluating = False
        self._progress.stop()
        self._btn_eval.config(state="normal", text="▶   EVALUATE")

    def _status(self, msg):
        self._status_var.set(msg)

    def _log(self, msg):
        logger.info(msg)


def run():
    app = EvaluationApp()
    app.mainloop()