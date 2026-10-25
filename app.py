import re
from io import BytesIO

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import extract_resume_text

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from xml.sax.saxutils import escape


# ---------- PDF REPORT ----------
def generate_pdf_report(report_text):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    content = [
        Paragraph(
            "Career Readiness AI - Analysis Report",
            styles["Title"],
        ),
        Spacer(1, 20),
    ]

    for line in report_text.splitlines():
        if line.strip():
            content.append(
                Paragraph(escape(line), styles["Normal"])
            )
            content.append(Spacer(1, 6))
        else:
            content.append(Spacer(1, 8))

    doc.build(content)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="Career Readiness AI",
    page_icon="🎯",
    layout="wide",
)


# ---------- SKILLS ----------
SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript",
    "C++", "C#", "HTML", "CSS", "React",
    "Node.js", "Django", "Flask", "FastAPI",
    "REST API", "SQL", "MySQL", "PostgreSQL",
    "MongoDB", "Git", "GitHub", "Docker",
    "Kubernetes", "AWS", "Azure", "Google Cloud",
    "Machine Learning", "Deep Learning",
    "Generative AI", "Artificial Intelligence",
    "NLP", "Data Analysis", "Pandas", "NumPy",
    "Scikit-learn", "TensorFlow", "PyTorch",
    "Power BI", "Excel", "Communication",
    "Problem Solving", "Data Structures",
    "OOP", "DBMS", "Operating Systems",
    "Cybersecurity", "Linux", "Agile",
]


# ---------- AI MODEL ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------- FIND SKILLS ----------
def find_skills(text):
    found = []
    text = text.lower()

    for skill in SKILLS:
        pattern = (
            r"(?<!\w)"
            + re.escape(skill.lower())
            + r"(?!\w)"
        )

        if re.search(pattern, text):
            found.append(skill)

    return sorted(found, key=str.lower)


# ---------- CREATE REPORT TEXT ----------
def create_report(data):
    lines = [
        "CAREER READINESS AI - ANALYSIS REPORT",
        "=" * 42,
        "",
        f"Resume-Job Similarity: {data['score']:.2f}%",
        "",
        "SKILLS FOUND IN RESUME",
        ", ".join(data["resume_skills"])
        or "No recognized skills found",
        "",
        "REQUIRED JOB SKILLS",
        ", ".join(data["job_skills"])
        or "No recognized skills found",
        "",
        "MATCHING SKILLS",
        ", ".join(data["matched"])
        or "No matching skills detected",
        "",
        "SKILLS TO IMPROVE",
        ", ".join(data["missing"])
        or "No gaps detected in recognized skills",
        "",
        "SUGGESTED NEXT STEPS",
    ]

    if data["missing"]:
        for skill in data["missing"]:
            lines.append(f"- Learn and practice {skill}.")
    else:
        lines.append(
            "- Review job requirements and strengthen your projects."
        )

    lines.extend([
        "",
        "Note: Similarity is an estimate, not an ATS score "
        "or a probability of getting hired.",
        "This report checks recognized skills and semantic similarity.",
    ])

    return "\n".join(lines)


# ---------- MAIN UI ----------
st.title("🎯 Career Readiness AI")
st.subheader("AI-Powered Resume and Job Matching System")

st.write(
    "Upload your resume and compare it with your target job "
    "description to identify relevant skills and improvement areas."
)

job_description = st.text_area(
    "Paste your target job description",
    height=180,
    key="job_description_input",
)

resume = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"],
    key="resume_pdf_upload",
)


# ---------- ANALYZE RESUME ----------
if st.button("Analyze Resume", key="analyze_resume_button"):

    if resume is None:
        st.warning("Please upload your resume PDF.")

    elif not job_description.strip():
        st.warning("Please enter a target job description.")

    else:
        try:
            with st.spinner("Analyzing your resume..."):

                resume_text = extract_resume_text(resume)

                if not resume_text or not resume_text.strip():
                    st.error(
                        "No readable text found in the PDF. "
                        "Please try a text-based PDF resume."
                    )

                else:
                    resume_skills = find_skills(resume_text)
                    job_skills = find_skills(job_description)

                    model = load_model()

                    embeddings = model.encode([
                        resume_text,
                        job_description,
                    ])

                    similarity = float(
                        cosine_similarity(
                            [embeddings[0]],
                            [embeddings[1]],
                        )[0][0]
                    )

                    score = max(
                        0.0,
                        min(100.0, similarity * 100),
                    )

                    matched = sorted(
                        set(resume_skills) & set(job_skills),
                        key=str.lower,
                    )

                    missing = sorted(
                        set(job_skills) - set(resume_skills),
                        key=str.lower,
                    )

                    extra = sorted(
                        set(resume_skills) - set(job_skills),
                        key=str.lower,
                    )

                    report_data = {
                        "score": score,
                        "resume_skills": resume_skills,
                        "job_skills": job_skills,
                        "matched": matched,
                        "missing": missing,
                    }

                    # Store report so download remains available
                    st.session_state["career_report"] = (
                        create_report(report_data)
                    )

                    st.success("Resume analysis completed!")
                    st.header("📊 Analysis Results")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Resume-Job Similarity",
                        f"{score:.2f}%",
                    )

                    col2.metric(
                        "Matching Skills",
                        len(matched),
                    )

                    col3.metric(
                        "Skills to Improve",
                        len(missing),
                    )

                    st.progress(int(score))

                    st.caption(
                        "Similarity is a semantic estimate, not a hiring "
                        "probability or official ATS score."
                    )

                    st.subheader("✅ Skills Found in Your Resume")

                    if resume_skills:
                        st.write(", ".join(resume_skills))
                    else:
                        st.info(
                            "No recognized skills found. "
                            "Check whether your PDF contains readable text."
                        )

                    st.subheader("🎯 Matching Skills")

                    if matched:
                        st.write(", ".join(matched))
                    else:
                        st.info("No matching recognized skills found.")

                    st.subheader("📚 Skills to Improve")

                    if missing:
                        for skill in missing:
                            st.write(
                                f"- Learn and practice **{skill}**."
                            )
                    else:
                        st.success(
                            "No gaps found among recognized job skills."
                        )

                    st.subheader("💡 Improvement Suggestions")

                    if missing:
                        st.write(
                            "Prioritize missing skills listed in the job "
                            "description. Build small projects to practice "
                            "important skills and add verified experience "
                            "to your resume."
                        )
                    else:
                        st.write(
                            "Strengthen your projects, add measurable "
                            "achievements, and tailor your resume to the "
                            "target role."
                        )

                    if extra:
                        st.subheader("➕ Additional Resume Skills")
                        st.write(", ".join(extra))

        except Exception as error:
            st.error("Resume analysis failed.")
            st.exception(error)


# ---------- PDF DOWNLOAD ----------
if "career_report" in st.session_state:
    st.subheader("📥 Download Your Analysis")

    try:
        pdf_bytes = generate_pdf_report(
            st.session_state["career_report"]
        )

        st.download_button(
            label="Download Analysis Report (PDF)",
            data=pdf_bytes,
            file_name="career_readiness_report.pdf",
            mime="application/pdf",
            key="download_analysis_report",
        )

    except Exception as error:
        st.error(f"PDF generation failed: {error}")