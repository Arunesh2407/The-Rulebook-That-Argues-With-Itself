# The Rulebook That Argues With Itself

> **AI Policy Verification, Contradiction Detection & Automatic Rule Reconciliation Pipeline**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F05032?style=for-the-badge&logo=groq&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0055FF?style=for-the-badge&logo=chromadb&logoColor=white)

---

## 📌 Problem Statement & Overview

University policy handbooks, academic regulations, and fee schedules are sprawling, multi-author documents updated across different academic years. As a result, they frequently contain **latent policy contradictions**—where one section states a strict non-waivable 75% attendance floor, while another empowers individual instructors to reduce it to 65%.

Standard conversational Retrieval-Augmented Generation (RAG) systems fail on these documents because they:
1. **Hallucinate answers** when policy facts do not exist in the text.
2. **Force arbitrary reconciliation** when two passages directly contradict each other, masking institutional rule conflicts from administrators.

**The Rulebook That Argues With Itself** is an evidence-grounded verification engine designed to eliminate speculation. It strictly classifies policy queries into one of three verifiable states: **ANSWERS**, **UNANSWERED (Abstain)**, or **CONTRADICTION**, providing verbatim citations and optional AI-synthesized policy reconciliation proposals.

---

## 🏗️ Key Architecture & Pipeline

The pipeline follows a modular, three-tier architecture:

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│      Document          │      │     Local Semantic     │      │     Groq Reasoning     │
│       Parser           │ ───► │       Retriever        │ ───► │        Verifier        │
│ (MD, PDF, CSV, TXT)    │      │  (ChromaDB + MiniLM)   │      │ (llama-3.3-70b-v)      │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

1. **Parser (`src/parser.py`)**: 
   Heterogeneous document ingestion engine. Parses Markdown headers (`#`), PDF sections (`pypdf`), and CSV tabular rows (`csv.DictReader`). Preserves source metadata (`file`, `section`, `page`) and assigns paragraph-aware chunk IDs.
2. **Retriever (`src/retriever.py`)**: 
   Local vector retrieval pipeline built on **ChromaDB** with `sentence-transformers/all-MiniLM-L6-v2` dense embeddings. Performs top-$k$ similarity queries over structured chunk passages.
3. **Verifier (`src/verifier.py`)**: 
   Logic reasoning chain powered by the **Groq API** (`llama-3.3-70b-versatile` with automated fallback). Enforces structured output via `response_format={"type": "json_object"}`. Restricts model knowledge exclusively to retrieved passages with zero external extrapolation.

---

## 🎯 The Three Handled Verification States

| Verification State | Description & Behavior | Output Schema |
| :--- | :--- | :--- |
| **`ANSWERS`** | Direct, explicit textual proof exists in the passages without conflicting statements. Synthesizes a factual summary. | Returns synthesized `answer` narrative and array of exact `citations`. |
| **`UNANSWERED`** | The required facts are absent, incomplete, or tangential. Rejects speculation and abstains cleanly. | Returns `answer` stating absence of evidence and empty `citations` `[]`. |
| **`CONTRADICTION`** | Two or more passages assert conflicting or incompatible rules for the same scenario. | Surfaces side-by-side conflicting provisions with $\ge 2$ verbatim `citations`. |

---

## 📑 Corpus & Benchmark Details

### Heterogeneous Knowledge Base (>6,000 Words)
The primary knowledge base synthesizes multiple administrative domain documents stored in `data/`:
- **`data/handbook.md`**: University Academic Handbook covering attendance, academic advising, prerequisite waivers, and administrative authority.
- **`data/academic_regulations.pdf`**: Official PDF regulations covering medical absence petitions, grading policies, and examination eligibility.
- **`data/fee_schedule.csv`**: Tabular CSV detailing payment deadlines, late payment penalties, and pro-rata tuition refund percentages.

### Planted Contradictions (`contradictions.md`)
The corpus contains 3 intentionally planted institutional rule conflicts:
1. **Attendance Eligibility**: Section 2.4 requires 75% minimum attendance with no waivers vs. Section 5.2 allowing instructors to lower the floor to 65%.
2. **Prerequisite Waiver Authority**: Section 4.1 vests exclusive waiver authority in the Dean vs. Section 7.3 giving Department Heads final, non-reviewable authority for majors.
3. **Medical Documentation Timelines**: PDF Section 3.2 mandates hospital documentation within 48 hours vs. Section 6.1 allowing Campus Health Center validation within 7 business days after returning.

### Evaluation Benchmark Suite (`eval_dataset.json`)
A curated 38-query benchmark dataset designed to test edge-case robustness:
- **10 Direct Answer Queries**: Factoid queries with clear policy coverage.
- **3 Contradiction Queries**: Specific queries triggering planted rule conflicts.
- **25 Hard Near-Miss Unanswerable Questions**: Queries asking about unstated policies (e.g., missing practical labs for sibling weddings, fee waivers for intramural sports) to test strict abstention.

---

## 📊 Benchmark Results (Honest Metrics)

Evaluated across the 38-query test suite via `eval.py`:

| Metric | Raw Count | Score / Target | Benchmark Notes |
| :--- | :--- | :--- | :--- |
| **Overall State Accuracy** | **32/38** | **84.21%** | Comprehensive classification accuracy across all states. |
| **Contradiction Detection** | **2/3** | **66.67%** | Successfully isolated side-by-side contradictory provisions. |
| **Unanswerable Rejection Rate** | **21/25** | **84.00%** | Rejected near-miss queries without hallucinating unstated rules. |
| **Direct Answers Accuracy** | **9/10** | **90.00%** | Accurately answered direct policy questions with verbatim quotes. |

### Analytical Performance Note
- **Strengths**: The system excels at strict negative rejection, avoiding hallucinated administrative policies on 84% of unanswerable edge cases. Response latency averages <0.5s per query via Groq's high-throughput API.
- **Edge Cases**: Minor mismatches occur on highly nuanced unanswerable questions where broad catch-all clauses in the handbook closely resemble specific policy rules.

---

## 💻 Local Setup & Usage

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/Arunesh2407/The-Rulebook-That-Argues-With-Itself.git
cd The-Rulebook-That-Argues-With-Itself

# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or install core packages:
pip install groq chromadb sentence-transformers pypdf streamlit pandas tabulate python-dotenv
```

### 3. Configure API Key
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run Evaluation Benchmark
Execute the automated evaluation script over the 38-query benchmark:
```bash
python eval.py
```

### 5. Launch Interactive Web Application
Launch the multi-tab Streamlit suite:
```bash
python -m streamlit run app.py
```

---

## 🎬 Demo & Deliverables

- **Video Demonstration**: [Link to Project Demo Video](#) *(Placeholder)*
- **Repository Link**: [GitHub Repository](https://github.com/Arunesh2407/The-Rulebook-That-Argues-With-Itself)

### Project Directory Structure
```
The-Rulebook-That-Argues-With-Itself/
├── data/
│   ├── handbook.md                 # Academic handbook document
│   ├── academic_regulations.pdf    # PDF regulations document
│   └── fee_schedule.csv            # Tabular fee schedule
├── src/
│   ├── parser.py                   # Multi-format document parser
│   ├── retriever.py                # ChromaDB vector indexer & retriever
│   └── verifier.py                 # Groq 3-state verifier & policy resolver
├── app.py                          # Streamlit web application & analytics dashboard
├── eval.py                         # Evaluation benchmark runner script
├── eval_dataset.json               # 38-query benchmark evaluation dataset
├── eval_results.json              # Full evaluation results & scorecard metrics
├── contradictions.md               # Planted contradiction documentation
├── requirements.txt                # Python dependencies manifest
└── README.md                       # Project documentation
```
