"""
Work Expected step viii: Deliver and present.
Run with:  streamlit run streamlit_app.py
(requires nlp_micro_project_with_additional_graphs.py and the models/ + results/ directories)
"""

import streamlit as st
import pandas as pd
import time
from nlp_micro_project_with_additional_graphs import load_deployment_models, predict_match

st.set_page_config(
    page_title="Resume to Job Matcher | NLP Micro Project",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .skill-badge-match {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .skill-badge-missing {
        display: inline-block;
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .skill-badge-general {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_models():
    return load_deployment_models()


# Pre-set examples for quick testing
PRESETS = {
    "Select a preset example...": ("", ""),
    "Full-Stack Developer vs Senior Engineer": (
        "Experienced Full-Stack Developer skilled in Python, JavaScript, React, Node.js, PostgreSQL, Docker, and REST APIs. 4 years experience building scalable web applications and microservices with FastAPI and MongoDB.",
        "Senior Web Engineer needed. Requirements: Strong experience in Python, React, PostgreSQL, Docker, REST APIs, and Cloud deployment. Familiarity with FastAPI or Django is a plus.",
    ),
    "Data Scientist vs Machine Learning Engineer": (
        "Data Scientist with expertise in Python, PyTorch, Scikit-learn, Pandas, NumPy, SQL, and data visualization. Experience in NLP, transformer models, and statistical analysis.",
        "Machine Learning Engineer role. Seeking candidates proficient in Python, PyTorch, ML algorithms, NLP, SQL, and model deployment on AWS.",
    ),
    "Junior Python Dev vs Senior DevOps Lead": (
        "Junior Software Developer with basic knowledge of Python, Git, HTML, and CSS. Passionate about learning web development and automation.",
        "Senior DevOps Lead required. Must have extensive expertise in Kubernetes, Docker, Terraform, AWS, CI/CD pipelines, Linux kernel tuning, and Python automation.",
    ),
}

# Sidebar Info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.title("Role Radar NLP")
    st.caption("UCS3564 NLP Micro Project - Theme 7")
    st.markdown("---")
    st.markdown("### 📌 Pipeline Overview")
    st.markdown(
        """
        - **Baseline Model:** TF-IDF Vectorizer + Logistic Regression
        - **Transformer:** Sentence-Transformers (`all-MiniLM-L6-v2`) + Logistic Regression
        - **Skill Extraction:** Data-driven vocabulary matching
        - **Latency:** Real-time inference (~<0.1s CPU)
        """
    )
    st.markdown("---")
    st.info("💡 **Tip:** Pick a preset scenario from the dropdown above to auto-fill inputs.")

# Main Header
st.markdown('<div class="main-header">🎯 Resume to Job Description Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Resume-to-Job Matching, Skill Gap Analysis & Real-Time Classification</div>',
    unsafe_allow_html=True,
)

# Load Models
try:
    with st.spinner("Loading AI models and skill vocabulary..."):
        models = get_models()
        tfidf_model, tfidf_clf, transformer_model, transformer_clf, skill_vocab = models
    st.success(f"Models loaded successfully! Active Skill Vocabulary: **{len(skill_vocab)} skills**", icon="✅")
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure model training files exist in `models/` and `results/`.")
    st.stop()

# Presets Dropdown
selected_preset = st.selectbox("🚀 Try Preset Test Cases:", list(PRESETS.keys()))

default_resume, default_job = PRESETS[selected_preset]

# Layout for Inputs
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.subheader("📄 Candidate Resume / Profile")
    resume_input = st.text_area(
        "Paste resume text or candidate profile here:",
        value=default_resume,
        height=220,
        placeholder="e.g. Python Developer skilled in SQL, React, Docker...",
    )

with col_input2:
    st.subheader("📋 Job Description")
    job_input = st.text_area(
        "Paste job description text here:",
        value=default_job,
        height=220,
        placeholder="e.g. Looking for a Python Developer proficient in SQL and AWS...",
    )

# Action Button
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    match_clicked = st.button("⚡ Run Match Inference", type="primary", use_container_width=True)

if match_clicked or (selected_preset != "Select a preset example..." and resume_input and job_input):
    if not resume_input.strip() or not job_input.strip():
        st.warning("⚠️ Please provide both Candidate Resume text and Job Description text before matching.")
    else:
        with st.spinner("Analyzing text, computing embeddings, and extracting skills..."):
            start_time = time.time()
            result = predict_match(resume_input, job_input, models)
            total_exec_time = round(time.time() - start_time, 4)

        if result["status"] == "error":
            st.error(f"❌ Match Failed: {result['message']}")
        else:
            st.markdown("---")

            # Metrics Summary Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            t_pred = result["transformer_prediction"]
            if t_pred == "Strong Match":
                color_badge = "🟢"
            elif t_pred == "Moderate Match":
                color_badge = "🟡"
            else:
                color_badge = "🔴"

            with m_col1:
                st.metric("Transformer Prediction", f"{color_badge} {t_pred}")

            with m_col2:
                st.metric("TF-IDF Baseline", result["tfidf_prediction"])

            with m_col3:
                num_matched = len(result["matching_skills"])
                num_missing = len(result["missing_skills"])
                total_req = num_matched + num_missing
                match_pct = round((num_matched / total_req * 100)) if total_req > 0 else 0
                st.metric("Skill Match Rate", f"{match_pct}%", f"{num_matched} / {total_req} skills")

            with m_col4:
                st.metric("Inference Latency", f"{result['transformer_latency_seconds']} s")

            if result.get("input_was_truncated"):
                st.warning("⚠️ Input exceeded maximum length limit (500 words per section) and was truncated for optimal inference.")

            # Tabbed Detailed Results
            tab1, tab2, tab3 = st.tabs(["🔍 Match Analysis", "🧩 Skill Extraction & Gaps", "📊 Model Probabilities"])

            with tab1:
                st.markdown("### Match Overview")
                st.write(
                    f"Based on semantic embeddings (`all-MiniLM-L6-v2`), the overall alignment between this profile and job description is classified as **{t_pred}**."
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Sentence-Transformer Classification")
                    st.info(f"**Result:** {result['transformer_prediction']}")
                    if result.get("transformer_probs"):
                        df_probs_trans = pd.DataFrame(
                            list(result["transformer_probs"].items()), columns=["Class", "Probability"]
                        ).sort_values(by="Probability", ascending=False)
                        st.dataframe(df_probs_trans, hide_index=True, use_container_width=True)

                with c2:
                    st.markdown("#### TF-IDF Baseline Classification")
                    st.info(f"**Result:** {result['tfidf_prediction']}")
                    if result.get("tfidf_probs"):
                        df_probs_tfidf = pd.DataFrame(
                            list(result["tfidf_probs"].items()), columns=["Class", "Probability"]
                        ).sort_values(by="Probability", ascending=False)
                        st.dataframe(df_probs_tfidf, hide_index=True, use_container_width=True)

            with tab2:
                st.markdown("### Skill Breakdown")
                sk_col1, sk_col2 = st.columns(2)

                with sk_col1:
                    st.markdown("#### ✅ Matching Required Skills")
                    if result["matching_skills"]:
                        badges_html = "".join([f'<span class="skill-badge-match">{s}</span>' for s in result["matching_skills"]])
                        st.markdown(badges_html, unsafe_allow_html=True)
                    else:
                        st.write("None found")

                with sk_col2:
                    st.markdown("#### ❌ Missing Required Skills")
                    if result["missing_skills"]:
                        badges_html = "".join([f'<span class="skill-badge-missing">{s}</span>' for s in result["missing_skills"]])
                        st.markdown(badges_html, unsafe_allow_html=True)
                    else:
                        st.write("None missing!")

                st.markdown("---")
                st.markdown("#### 📌 Full Skill Extraction")
                col_res_sk, col_job_sk = st.columns(2)
                with col_res_sk:
                    st.markdown("**Skills detected in Candidate Resume:**")
                    if result.get("resume_skills"):
                        b_html = "".join([f'<span class="skill-badge-general">{s}</span>' for s in result["resume_skills"]])
                        st.markdown(b_html, unsafe_allow_html=True)
                    else:
                        st.write("No vocabulary skills recognized")

                with col_job_sk:
                    st.markdown("**Skills required in Job Description:**")
                    if result.get("job_skills"):
                        b_html = "".join([f'<span class="skill-badge-general">{s}</span>' for s in result["job_skills"]])
                        st.markdown(b_html, unsafe_allow_html=True)
                    else:
                        st.write("No vocabulary skills recognized")

            with tab3:
                st.markdown("### Class Probabilities Breakdown")
                st.caption("Comparing confidence scores across target classes: Strong Match, Moderate Match, Weak Match.")

                if result.get("transformer_probs"):
                    st.markdown("#### Sentence-Transformer Model Confidence")
                    for cls_name, prob in result["transformer_probs"].items():
                        st.write(f"**{cls_name}**: {prob * 100:.1f}%")
                        st.progress(prob)

                if result.get("tfidf_probs"):
                    st.markdown("---")
                    st.markdown("#### TF-IDF Baseline Confidence")
                    for cls_name, prob in result["tfidf_probs"].items():
                        st.write(f"**{cls_name}**: {prob * 100:.1f}%")
                        st.progress(prob)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "UCS3564 Natural Language Processing Micro Project • Theme 7: Resume to Job Matching"
    "</div>",
    unsafe_allow_html=True,
)
