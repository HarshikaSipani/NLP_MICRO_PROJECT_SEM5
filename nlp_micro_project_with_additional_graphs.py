"""
UCS3564 Natural Language Processing - Micro Project
Theme 7: Resume to Job Description Matching and Skill Extraction

Module for data preprocessing, training, model artifact loading, and inference.
"""

import json
import os
import re
import random
import time
import tracemalloc

import numpy as np
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sentence_transformers import SentenceTransformer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"
MAX_WORDS = 500  # practical per-field length cap for one inference


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    repo_id = "jimyones/role-radar-dataset"
    files = [
        "scraped_jobs.json",
        "synthetic_profiles.json",
        "phase3_labels.json",
        "gold_labels.json",
    ]
    for f in files:
        try:
            hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset", local_dir=DATA_DIR)
            print(f, "downloaded")
        except Exception as e:
            print("Could not download", f, "-", e)


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)              # strip URLs
    text = re.sub(r"[^a-zA-Z0-9+#.\- ]", " ", text)           # keep skill-friendly chars (c++, c#, node.js)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def profile_to_text(profile):
    roles = profile.get("roles", [])
    primary = profile.get("skills_primary", [])
    secondary = profile.get("skills_secondary", [])
    domains = profile.get("domains", [])
    preferences = profile.get("preferences", {})

    roles = roles if isinstance(roles, list) else [str(roles)]
    primary = primary if isinstance(primary, list) else [str(primary)]
    secondary = secondary if isinstance(secondary, list) else [str(secondary)]
    domains = domains if isinstance(domains, list) else [str(domains)]

    text = " ".join(roles) + " " + " ".join(primary) + " " + " ".join(secondary) + " " + " ".join(domains)
    text += " " + str(profile.get("seniority", "")) + " " + str(profile.get("experience_years", ""))
    if isinstance(preferences, dict):
        text += " " + " ".join(str(v) for v in preferences.values())
    return clean_text(text)


def job_to_text(job):
    text = (
        str(job.get("title", "")) + " "
        + str(job.get("description", "")) + " "
        + str(job.get("seniority_level", "")) + " "
        + str(job.get("industry", "")) + " "
        + str(job.get("role_family_hint", "")) + " "
        + str(job.get("location", ""))
    )
    return clean_text(text)


def build_skill_vocabulary(profiles):
    """Data-driven skill vocabulary, built from the actual dataset."""
    vocab = set()
    for profile in profiles:
        for key in ("skills_primary", "skills_secondary"):
            values = profile.get(key, [])
            if isinstance(values, list):
                for skill in values:
                    cleaned = clean_text(skill)
                    if cleaned:
                        vocab.add(cleaned)
    return vocab


def extract_skills(text, vocabulary):
    text = clean_text(text)
    found = []
    for skill in vocabulary:
        if len(skill) < 2:
            continue
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)
    return sorted(set(found))


def compare_skills(resume, job, vocabulary):
    resume_skills = extract_skills(resume, vocabulary)
    job_skills = extract_skills(job, vocabulary)
    matching = [s for s in resume_skills if s in job_skills]
    missing = [s for s in job_skills if s not in resume_skills]
    return matching, missing


def create_class(score):
    if score >= 80:
        return "Strong Match"
    elif score >= 65:
        return "Moderate Match"
    else:
        return "Weak Match"


def prepare_data():
    jobs = load_json("scraped_jobs.json")
    profiles = load_json("synthetic_profiles.json")
    labels = load_json("phase3_labels.json")
    gold_labels = load_json("gold_labels.json")

    print("Jobs:", len(jobs), "| Profiles:", len(profiles),
          "| Labels:", len(labels), "| Gold labels:", len(gold_labels))

    job_lookup = {str(j.get("id")): j for j in jobs}
    profile_lookup = {str(p.get("profile_id")): p for p in profiles}

    skill_vocabulary = build_skill_vocabulary(profiles)
    print("Skills extracted from data:", len(skill_vocabulary))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "skill_vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(skill_vocabulary), f, indent=2)

    records = []
    for item in labels:
        pair_id = str(item.get("pair_id", ""))
        parts = pair_id.split("_")
        if len(parts) < 2:
            continue
        profile_id, job_id = parts[0], "_".join(parts[1:])
        if profile_id not in profile_lookup or job_id not in job_lookup:
            continue

        profile_text = profile_to_text(profile_lookup[profile_id])
        job_text = job_to_text(job_lookup[job_id])
        combined_text = profile_text + " [SEP] " + job_text

        composite = item.get("composite", None)
        try:
            composite = float(composite)
        except (TypeError, ValueError):
            continue

        records.append({
            "pair_id": pair_id,
            "profile_id": profile_id,
            "job_id": job_id,
            "profile_text": profile_text,
            "job_text": job_text,
            "combined_text": combined_text,
            "composite": composite,
        })

    df = pd.DataFrame(records)
    print("Final usable pairs:", len(df))

    df["label"] = df["composite"].apply(create_class)
    print("Class distribution:\n", df["label"].value_counts())

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("TRAIN:", len(train_df), "| VALIDATION:", len(val_df), "| TEST:", len(test_df))

    train_df.to_csv(os.path.join(RESULTS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(RESULTS_DIR, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(RESULTS_DIR, "test.csv"), index=False)
    print("Data preparation completed.\n")


def evaluate_predictions(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{model_name} RESULTS")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    return {"Model": model_name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


def train_baseline():
    train_df = pd.read_csv(os.path.join(RESULTS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(RESULTS_DIR, "validation.csv"))
    test_df = pd.read_csv(os.path.join(RESULTS_DIR, "test.csv"))

    X_train, y_train = train_df["combined_text"].fillna(""), train_df["label"]
    X_val, y_val = val_df["combined_text"].fillna(""), val_df["label"]
    X_test, y_test = test_df["combined_text"].fillna(""), test_df["label"]

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(X_train_vec, y_train)

    val_preds = clf.predict(X_val_vec)
    print("\nTF-IDF BASELINE - VALIDATION")
    print(classification_report(y_val, val_preds, zero_division=0))

    start = time.time()
    test_preds = clf.predict(X_test_vec)
    inference_time = time.time() - start

    result = evaluate_predictions(y_test, test_preds, "TF-IDF + Logistic Regression (Baseline)")
    result["Test_Inference_Time_Seconds"] = inference_time
    pd.DataFrame([result]).to_csv(os.path.join(RESULTS_DIR, "tfidf_results.csv"), index=False)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "tfidf_classifier.pkl"))
    print("Baseline model saved.\n")


def train_transformer():
    train_df = pd.read_csv(os.path.join(RESULTS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(RESULTS_DIR, "validation.csv"))
    test_df = pd.read_csv(os.path.join(RESULTS_DIR, "test.csv"))

    X_train, y_train = train_df["combined_text"].fillna("").tolist(), train_df["label"].tolist()
    X_val, y_val = val_df["combined_text"].fillna("").tolist(), val_df["label"].tolist()
    X_test, y_test = test_df["combined_text"].fillna("").tolist(), test_df["label"].tolist()

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    transformer = SentenceTransformer(model_name)
    print("Transformer loaded:", model_name)

    start = time.time()
    X_train_emb = transformer.encode(X_train, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    X_val_emb = transformer.encode(X_val, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    X_test_emb = transformer.encode(X_test, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    embedding_time = time.time() - start
    print("Embedding time:", round(embedding_time, 2), "s | Dim:", X_train_emb.shape[1])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(X_train_emb, y_train)

    val_preds = clf.predict(X_val_emb)
    print("\nTRANSFORMER - VALIDATION")
    print(classification_report(y_val, val_preds, zero_division=0))

    start = time.time()
    test_preds = clf.predict(X_test_emb)
    inference_time = time.time() - start

    result = evaluate_predictions(y_test, test_preds, "Transformer + Logistic Regression")
    result["Embedding_Time_Seconds"] = embedding_time
    result["Test_Inference_Time_Seconds"] = inference_time
    pd.DataFrame([result]).to_csv(os.path.join(RESULTS_DIR, "transformer_results.csv"), index=False)

    tfidf_result = pd.read_csv(os.path.join(RESULTS_DIR, "tfidf_results.csv"))
    comparison = pd.concat(
        [
            tfidf_result[["Model", "Accuracy", "Precision", "Recall", "F1"]],
            pd.DataFrame([result])[["Model", "Accuracy", "Precision", "Recall", "F1"]],
        ],
        ignore_index=True,
    )
    print("\nMODEL COMPARISON\n", comparison)
    comparison.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    error_df = test_df.copy()
    error_df["actual"], error_df["predicted"] = y_test, test_preds
    errors = error_df[error_df["actual"] != error_df["predicted"]].copy()
    errors.head(10)[["pair_id", "profile_id", "job_id", "actual", "predicted", "composite"]].to_csv(
        os.path.join(RESULTS_DIR, "error_analysis_10_cases.csv"), index=False
    )
    print("Error analysis saved (up to 10 cases).")

    os.makedirs(MODELS_DIR, exist_ok=True)
    transformer.save(os.path.join(MODELS_DIR, "transformer_model"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "transformer_classifier.pkl"))
    print("Transformer model saved.\n")


def load_deployment_models():
    """Load trained models and skill vocabulary for inference."""
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    tfidf_clf = joblib.load(os.path.join(MODELS_DIR, "tfidf_classifier.pkl"))
    transformer = SentenceTransformer(os.path.join(MODELS_DIR, "transformer_model"), device="cpu")
    transformer_clf = joblib.load(os.path.join(MODELS_DIR, "transformer_classifier.pkl"))
    
    vocab_path = os.path.join(RESULTS_DIR, "skill_vocabulary.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            skill_vocabulary = set(json.load(f))
    else:
        skill_vocabulary = set()

    return tfidf, tfidf_clf, transformer, transformer_clf, skill_vocabulary


def predict_match(resume, job, models):
    tfidf, tfidf_clf, transformer, transformer_clf, skill_vocabulary = models

    resume_clean = clean_text(resume)
    job_clean = clean_text(job)

    # ---- empty input check ----
    if not resume_clean:
        return {"status": "error", "message": "Resume text is empty or invalid."}
    if not job_clean:
        return {"status": "error", "message": "Job description text is empty or invalid."}

    # ---- length truncation check ----
    words_resume = resume_clean.split()
    words_job = job_clean.split()
    original_length = len(words_resume) + len(words_job)

    resume_capped = " ".join(words_resume[:MAX_WORDS])
    job_capped = " ".join(words_job[:MAX_WORDS])

    combined = resume_capped + " [SEP] " + job_capped

    # ---- TF-IDF prediction ----
    tfidf_vec = tfidf.transform([combined])
    tfidf_pred = tfidf_clf.predict(tfidf_vec)[0]
    tfidf_probs = {}
    if hasattr(tfidf_clf, "predict_proba"):
        probs = tfidf_clf.predict_proba(tfidf_vec)[0]
        for cls_name, prob in zip(tfidf_clf.classes_, probs):
            tfidf_probs[cls_name] = round(float(prob), 4)

    # ---- Transformer prediction ----
    start = time.time()
    embedding = transformer.encode([combined], convert_to_numpy=True)
    transformer_pred = transformer_clf.predict(embedding)[0]
    latency = time.time() - start

    transformer_probs = {}
    if hasattr(transformer_clf, "predict_proba"):
        probs = transformer_clf.predict_proba(embedding)[0]
        for cls_name, prob in zip(transformer_clf.classes_, probs):
            transformer_probs[cls_name] = round(float(prob), 4)

    # ---- Skill extraction & comparison ----
    matching, missing = compare_skills(resume_clean, job_clean, skill_vocabulary)
    resume_skills = extract_skills(resume_clean, skill_vocabulary)
    job_skills = extract_skills(job_clean, skill_vocabulary)

    return {
        "status": "success",
        "transformer_prediction": transformer_pred,
        "transformer_probs": transformer_probs,
        "tfidf_prediction": tfidf_pred,
        "tfidf_probs": tfidf_probs,
        "matching_skills": matching,
        "missing_skills": missing,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "input_was_truncated": original_length > (MAX_WORDS * 2),
        "transformer_latency_seconds": round(latency, 4),
    }


def measure_single_inference_memory(models):
    tracemalloc.start()
    predict_match("Python developer with SQL", "Backend developer needed with Python and SQL", models)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nSingle inference memory - current: {current / 1e6:.2f} MB, peak: {peak / 1e6:.2f} MB")


def test_edge_cases(models):
    print("=" * 60)
    print("EDGE CASE TESTING")
    print("=" * 60)

    cases = [
        ("Empty resume", "", "Python developer required"),
        ("Empty job description", "Python developer with SQL", ""),
        ("OOV / unknown terms", "qwerty zxcvb asdfgh", "Python Java SQL developer"),
        ("Code-mixed text", "Python developer hoon aur SQL aata hai", "Looking for a Python developer with SQL"),
        ("Very long input", "Python Java SQL Machine Learning " * 1000, "Python developer with machine learning"),
        ("Normal input", "Python developer with SQL and FastAPI", "We need a Python backend developer with SQL"),
    ]

    for i, (name, resume, job) in enumerate(cases, 1):
        result = predict_match(resume, job, models)
        print(f"\n{i}. {name}")
        print(result)

    measure_single_inference_memory(models)


if __name__ == "__main__":
    download_dataset()
    prepare_data()
    train_baseline()
    train_transformer()

    deployment_models = load_deployment_models()
    test_edge_cases(deployment_models)
