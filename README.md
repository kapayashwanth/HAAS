# Handwritten Answer Assessment System (HAAS)

HAAS is an AI-assisted assessment platform designed to evaluate handwritten student answers using OCR, natural language processing, semantic similarity, and rubric-aligned automated scoring.

## Executive Summary

Educational teams face growing evaluation workloads, inconsistent scoring, and turnaround delays. HAAS addresses these challenges by combining machine vision and language intelligence to deliver fast, explainable, and teacher-aligned answer assessment.

The platform supports both desktop workflow and scripted execution, making it suitable for pilot deployments, lab evaluations, and scalable institutional workflows.

## Core Capabilities

- High-recall handwritten OCR with multi-pass extraction and confidence-aware merging.
- OCR-oriented preprocessing pipeline (deskew, denoise, contrast enhancement, adaptive binarization).
- NLP normalization and correction pipeline (tokenization, noise cleanup, lemmatization, optional spell correction).
- Multi-metric similarity engine with semantic weighting and OCR-noise resilience.
- Balanced, teacher-like mark generation with grade bands and feedback narratives.
- Semantic score visibility for student-level transparency.
- Exportable PDF reports for documentation and audit trails.

## Business Value

- Improves grading consistency across large student cohorts.
- Reduces evaluator effort and turnaround time.
- Provides transparent scoring signals (semantic score, breakdown metrics, feedback).
- Supports evidence-driven continuous improvement of answer quality.

## Solution Architecture

HAAS follows a modular pipeline architecture:

1. Image Ingestion
2. Image Preprocessing
3. OCR Extraction
4. Text Cleaning and Normalization
5. Similarity and Relevance Scoring
6. Mark and Grade Generation
7. Report Generation and Export

## Technology Stack

- Python 3.10+
- EasyOCR
- OpenCV
- Sentence Transformers
- scikit-learn
- NLTK
- SymSpellPy
- ReportLab
- Tkinter (desktop UI)
- FastAPI + Jinja2 (web app mode)

## Repository Structure

- main.py: Application entrypoint (GUI, CLI, self-test)
- requirements.txt: Dependency manifest
- answers/model_answer.txt: Reference answer input
- images/: Input handwritten images
- reports/: Generated PDF reports
- logs/: Runtime logs
- src/image_loader.py: Image loading and validation
- src/preprocess.py: OCR preprocessing pipeline
- src/ocr_reader.py: OCR extraction engine
- src/text_cleaner.py: NLP cleanup and normalization
- src/similarity.py: Similarity and relevance scoring
- src/marks_generator.py: Marks, grading, and feedback
- src/report_generator.py: PDF report generation
- src/gui.py: Desktop interface

## Getting Started

### 1. Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Application

```bash
python main.py
```

### 3. Run Self-Test

```bash
python main.py --test
```

### 4. Run CLI Mode

```bash
python main.py --cli <image_path> <model_answer_path> [max_marks] [student_name]
```

### 5. Run Web App Locally

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --reload
```

Open: http://127.0.0.1:8000

## Output and Reporting

- Student answer extraction preview
- Metric-level similarity breakdown
- Final score, marks, grade, and feedback
- Semantic score and semantic marks
- Exportable PDF assessment report

## Operational Notes

- First semantic scoring run may be slower due to model initialization.
- For production reliability, use an isolated virtual environment.
- OCR quality depends on image clarity, orientation, and handwriting legibility.

## Roadmap

- Web-hosted API and frontend deployment mode
- Multi-question batch processing
- Rubric customization per subject
- Institutional analytics dashboard

## Railway Deployment

HAAS now includes Railway-ready web mode.

### Files used for Railway
- Procfile
- web_app.py
- requirements.txt

### Deploy Steps
1. Push this repository to GitHub.
2. In Railway, create a new project from your GitHub repo.
3. Railway will detect Procfile and run:
	- uvicorn web_app:app --host 0.0.0.0 --port $PORT
4. Wait for build and deploy to complete.
5. Open the generated Railway public URL.

### Runtime notes
- For smoother OCR/model inference in production, choose a plan with at least 4 GB RAM.
- First request may be slower because NLP/embedding models warm up on first load.

## Contribution

Internal collaboration and contribution guidelines can be defined in a dedicated CONTRIBUTING document.

## License

License policy is currently to be defined by project owners.

## Project Identity

Official name: Handwritten Answer Assessment System (HAAS)
