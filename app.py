import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# =========================================================
# PAGE CONFIG (clean executive look)
# =========================================================
st.set_page_config(
    page_title="Attendance Intelligence System",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================
st.title("📊 Attendance Intelligence System")
st.markdown("""
### Barrier Analysis • Predictive Risk Scoring • Intervention Intelligence
This system transforms attendance notes into actionable intervention signals.
""")

# =========================================================
# SAFE UTILITIES
# =========================================================
def safe_df(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna("")
    return df.astype(str)


def clean_columns(df):
    seen = {}
    new_cols = []

    for c in df.columns:
        c = str(c).strip().lower()
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
    df.columns = new_cols
    return df


def extract_id(x):
    if pd.isna(x):
        return ""
    match = re.search(r"\d+", str(x))
    return match.group(0) if match else str(x).strip()


# =========================================================
# UPLOAD SECTION (VISUAL BLOCK)
# =========================================================
st.header("📂 Upload Data")

col1, col2 = st.columns(2)

with col1:
    att_file = st.file_uploader("Attendance File", type=["xlsx"])

with col2:
    note_file = st.file_uploader("Notes File", type=["xlsx"])


# =========================================================
# MAIN PIPELINE
# =========================================================
if att_file and note_file:

    attendance = pd.read_excel(att_file, header=None)
    notes = pd.read_excel(note_file, header=None)

    attendance = clean_columns(attendance)
    notes = clean_columns(notes)

    attendance = attendance.fillna("")
    notes = notes.fillna("")

    # =========================================================
    # IDENTIFY ID COLUMNS
    # =========================================================
    att_id = [c for c in attendance.columns if "name" in c or "student" in c or "id" in c]
    note_id = [c for c in notes.columns if "name" in c or "student" in c or "id" in c]

    if not att_id or not note_id:
        st.error("Could not detect student ID columns")
        st.stop()

    attendance["student_id"] = attendance[att_id[0]].apply(extract_id)
    notes["student_id"] = notes[note_id[0]].apply(extract_id)

    # =========================================================
    # NOTES COLUMN DETECTION
    # =========================================================
    note_cols = [c for c in notes.columns if "note" in c]

    if not note_cols:
        st.error("No Notes column detected")
        st.stop()

    note_col = note_cols[0]

    # =========================================================
    # FILTER ATTENDANCE VISITS
    # =========================================================
    visit_cols = [c for c in notes.columns if "visit" in c]

    if not visit_cols:
        st.error("No Visit column detected")
        st.stop()

    visit_col = visit_cols[0]

    attendance_visits = notes[
        notes[visit_col].astype(str).str.lower().str.contains("attendance", na=False)
    ].copy()

    # =========================================================
    # KPI ROW (EXECUTIVE VIEW)
    # =========================================================
    st.header("📌 Key Indicators")

    k1, k2, k3 = st.columns(3)

    k1.metric("Attendance Visit Records", len(attendance_visits))
    k2.metric("Students Tracked", attendance_visits["student_id"].nunique())
    k3.metric("Total Notes Analyzed", len(notes))

    # =========================================================
    # 🧠 BARRIER CLASSIFICATION ENGINE
    # =========================================================
    def classify(note):
        n = str(note).lower()

        if any(x in n for x in ["bus", "transport", "ride", "pickup", "drop"]):
            return "Transportation"

        if any(x in n for x in ["voicemail", "no answer", "left message", "called"]):
            return "Contact Attempted"

        if any(x in n for x in ["family", "home", "housing"]):
            return "Family/Home"

        if any(x in n for x in ["sick", "medical", "doctor"]):
            return "Health"

        if any(x in n for x in ["test", "nwea", "assignment", "classwork"]):
            return "School/Academic"

        return "Other"

    attendance_visits["barrier"] = attendance_visits[note_col].apply(classify)

    # =========================================================
    # BARRIER SUMMARY (VISUAL SECTION)
    # =========================================================
    st.header("🧠 Barrier Intelligence")

    barrier_summary = attendance_visits["barrier"].value_counts().reset_index()
    barrier_summary.columns = ["Barrier", "Count"]

    c1, c2 = st.columns([1, 1])

    with c1:
        st.plotly_chart(
            px.bar(barrier_summary, x="Barrier", y="Count", title="Barrier Frequency"),
            use_container_width=True
        )

    with c2:
        st.dataframe(barrier_summary)

    # =========================================================
    # RISK ENGINE
    # =========================================================
    st.header("🔥 Predictive Risk Engine")

    weights = {
        "Transportation": 0.9,
        "Contact Attempted": 0.7,
        "Family/Home": 0.8,
        "Health": 0.6,
        "School/Academic": 0.5,
        "Other": 0.3
    }

    attendance_visits["impact"] = attendance_visits["barrier"].map(weights)

    student_risk = attendance_visits.groupby("student_id").agg(
        barrier_count=("barrier", "count"),
        avg_impact=("impact", "mean")
    ).reset_index()

    student_risk["avg_impact"] = student_risk["avg_impact"].fillna(0)

    student_risk["risk_score"] = (
        (student_risk["barrier_count"] / student_risk["barrier_count"].max() if student_risk["barrier_count"].max() > 0 else 0) * 40 +
        student_risk["avg_impact"] * 60
    )

    def label(x):
        if x >= 75:
            return "Critical 🔴"
        if x >= 50:
            return "High 🟠"
        if x >= 25:
            return "Medium 🟡"
        return "Low 🟢"

    student_risk["risk_level"] = student_risk["risk_score"].apply(label)

    student_risk = student_risk.sort_values("risk_score", ascending=False)

    # =========================================================
    # RISK VISUAL SECTION
    # =========================================================
    c1, c2 = st.columns([1, 1])

    with c1:
        st.plotly_chart(
            px.histogram(student_risk, x="risk_score", title="Risk Distribution"),
            use_container_width=True
        )

    with c2:
        st.dataframe(student_risk)

    # =========================================================
    # STUDENT DRILLDOWN (POWER FEATURE)
    # =========================================================
    st.header("🧍 Student Drilldown")

    student = st.selectbox("Select Student", student_risk["student_id"].unique())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Profile")
        st.dataframe(student_risk[student_risk["student_id"] == student])

    with col2:
        st.subheader("Attendance Notes")
        st.dataframe(
            safe_df(attendance_visits[attendance_visits["student_id"] == student])
        )

else:
    st.info("Upload both Attendance and Notes Excel files to begin analysis.")
