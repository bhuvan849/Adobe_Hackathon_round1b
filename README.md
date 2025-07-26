# 🤖 Persona-Driven Document Intelligence System  
### Adobe India Hackathon 2025 – Round 1B Submission

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)  
[![Docker](https://img.shields.io/badge/Docker-Build-blue)](https://www.docker.com/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is an advanced document analysis system designed for the “Connecting the Dots” challenge. It intelligently extracts and prioritizes the most relevant sections from a collection of PDFs based on a specific user persona and their job-to-be-done.

---

## ✨ Key Features

- **🧠 Multi-Strategy Parsing:**  
  The system uses a robust, two‑stage parser. It first attempts to use a PDF’s built‑in bookmarks for perfect accuracy and falls back to a sophisticated visual and structural analysis if bookmarks are unavailable.

- **🎯 Constraint‑Aware AI:**  
  The analysis engine is fully generic. It uses a dynamic constraint system defined in `analysis_config.json` to strictly filter and re‑rank results based on the specific requirements of a task (e.g., filtering for “vegetarian” content).

- **✍️ High‑Quality Summarization:**  
  It generates coherent, multi‑sentence summaries for the top‑ranked sections, providing actionable intelligence at a glance.

- **📦 Dockerized Environment:**  
  The entire application is containerized with Docker for consistent, reliable, and platform‑independent execution.

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.9  
- **Core Libraries:**
  - `PyMuPDF` for robust PDF parsing  
  - `sentence-transformers` for state‑of‑the‑art semantic analysis (`all-mpnet-base-v2`)  
  - `spaCy` for advanced natural language processing  
  - `NumPy` & `scikit-learn` for numerical operations and ML components  

- **Containerization:** Docker  

---

## 🚀 How to Run

The solution is designed to be built and run inside a Docker container.

### 1. Build the Docker Image

From the root directory of the project, run:

```bash
docker build -t mysolution:latest .
```

### 2. Prepare the Input Directory

Your input folder must be structured as follows:

```
input/
├── PDFs/
│   ├── document1.pdf
│   └── document2.pdf
└── challenge1b_input.json
```

The `challenge1b_input.json` file defines the persona and job for the analysis.

### 3. Run the Container

Execute the following command to start the analysis. The final results will be saved to `output/challenge1b_output.json`.

#### Windows (Command Prompt)

```cmd
docker run --rm -v "%cd%/input:/app/input" -v "%cd%/output:/app/output" --network none mysolution:latest
```

#### Windows (PowerShell)

```powershell
docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" --network none mysolution:latest
```

---

Enjoy your persona‑driven insights! 🎉

