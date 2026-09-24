# 🎯 Resume to Job Description Matching & Skill Extraction

**Course:** UCS3564 Natural Language Processing — Micro Project (Theme 7)  

An AI-powered system designed for automated candidate profile matching against job descriptions, classification (*Strong Match*, *Moderate Match*, *Weak Match*), real-time skill extraction, and skill-gap identification.

---

## 📌 Problem Formulation

- **Use Case:** Real-time recruiter candidate shortlisting and candidate skill-gap feedback.
- **Input:** Free text candidate profile / resume + free text job description.
- **Output:**
  - **Match Classification:** `Strong Match`, `Moderate Match`, or `Weak Match`
  - **Skill Analysis:** Matching required skills vs missing skills
  - **Confidence Signals:** Class probabilities for both Sentence-Transformer and TF-IDF baseline models
- **Performance Targets:** Weighted F1 score beating classical baselines, sub-second single CPU inference (~<0.1s), graceful degradation on empty, code-mixed, or long inputs.

---

## 📁 Repository Structure

```
├── data/                                      # Downloaded dataset JSON files
├── models/                                    # Saved trained model artifacts
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_classifier.pkl
│   ├── transformer_classifier.pkl
│   └── transformer_model/                     # Saved Sentence-Transformer weights
├── results/                                   # Evaluation results and data splits
│   ├── train.csv / validation.csv / test.csv
│   ├── skill_vocabulary.json                  # Extracted 164-skill vocabulary
│   ├── tfidf_results.csv / transformer_results.csv
│   ├── model_comparison.csv
│   └── error_analysis_10_cases.csv
├── nlp_micro_project_with_additional_graphs.ipynb # Interactive Jupyter Notebook
├── nlp_micro_project_with_additional_graphs.py    # Python module for training & inference
├── streamlit_app.py                           # Interactive Streamlit Web Application
├── requirements.txt                           # Project dependencies
└── README.md                                  # Project documentation
```

---

## 📊 Dataset Provenance & Licence

- **Source:** Hugging Face Dataset (`jimyones/role-radar-dataset`)
- **Contents:**
  - `scraped_jobs.json` — Job titles, descriptions, requirements, locations, industries.
  - `synthetic_profiles.json` — Roles, primary/secondary skills, domain expertise.
  - `phase3_labels.json` & `gold_labels.json` — Composite matching scores and ground truth annotations.

---

## 🏗️ Architecture & Model Pipeline

1. **Text Preprocessing:** URL stripping, special character cleaning preserving programming language syntax (`c++`, `c#`, `node.js`), whitespace normalization.
2. **Skill Vocabulary Extraction:** Data-driven skill extraction from candidate profiles generating an active 164-skill vocabulary.
3. **TF-IDF Baseline:** `TfidfVectorizer` (ngram range 1-2, max 5000 features) + `LogisticRegression` (balanced class weighting).
4. **Sentence-Transformer Model:** Pretrained `sentence-transformers/all-MiniLM-L6-v2` dense embeddings (384 dimensions) + `LogisticRegression` classifier.

---

## 🚀 Quick Start & Execution Order

### 1. Environment Setup

```bash
# Clone repository or navigate to folder
cd e:\NLP_MICROPROJECT

# Activate virtual environment
.\venvmicro\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Pipeline / Model Training (Optional)

> *Note: Pre-trained model artifacts are already included in `models/` and `results/`.*

To run the complete data download, preprocessing, model training, and evaluation:

**Option A (Python Script):**
```bash
python nlp_micro_project_with_additional_graphs.py
```

**Option B (Jupyter Notebook):**
```bash
jupyter notebook nlp_micro_project_with_additional_graphs.ipynb
```

### 4. Launch the Web Application

To open the interactive Streamlit user interface:

```bash
streamlit run streamlit_app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 🖥️ Web App Features

- **Preset Test Scenarios:** 1-click test cases for quick evaluation (*Full-Stack Dev*, *Data Scientist*, *Junior Python Dev vs DevOps*).
- **Live Match Inference:** Instant classification using Sentence-Transformers and TF-IDF models.
- **Skill Gap Badges:** Visual breakdown of matched skills (green badges) and missing skills (red badges).
- **Confidence Probabilities:** Progress bars displaying model confidence across all classes.

---

## 📜 License

Created for UCS3564 Natural Language Processing Micro Project — Theme 7.
