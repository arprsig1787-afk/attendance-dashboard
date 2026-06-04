import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Attendance Intelligence System", layout="wide")

st.title("📊 Attendance Intelligence System")
st.markdown("Barrier Intelligence • Predictive Risk Scoring • Intervention Insights")


# =========================================================
# SAFE DATA HANDLING
# =========================================================
def safe_df(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna("")
    return df.astype(str)


# =========================================================
# FIX COLUMN ISSUES (duplicate + messy headers)
# =========================================================
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


# =========================================================
# EXTRACT ID FROM MIXED CELLS
# =========================================================
def extract_id(val):
    if pd.isna(val):
        return ""
    val = str(val)
    match = re.search(r"\d+", val)
    return match.group(0) if match else val.strip()


# =========================================================
# ROBUST COLUMN DETECTION (FIXES YOUR ERROR)
# =========================================================
def detect_best_column(df):
    best_col = None
    best_score = -1

    for col in df.columns:
        s = df[col].astype(str)

        # score structure: numeric presence + text presence
        num_score = s.str.contains(r"\d", na=False).mean()
        text_score = s.str.contains(r"[a-zA-Z]", na=False).mean()

        score = num_score + text_score

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


# =========================================================
# UPLOAD SECTION
# =========================================================
st.header("📂 Upload Data")

col1, col2 = st.columns(2)

with col1:
    att_file = st.file_uploader("Attendance File", type=["xlsx"])

with col2:
    note_file = st.file_uploader("Notes File", type=["xlsx"])


# =========================================================
# MAIN APP
# =========================================================
if att_file and note_file:

    # -------------------------
    # LOAD DATA
    # -------------------------
    attendance = pd.read_excel(att_file, header=None)
    notes = pd.read_excel(note_file, header=None)

    attendance = clean_columns(attendance)
    notes = clean_columns(notes)

    attendance = attendance.fillna("")
    notes = notes.fillna("")

    # =========================================================
    # AUTO DETECT ID COLUMNS (FIXED)
    # =========================================================
    att_id_col = detect_best_column(attendance)
    note_id_col = detect_best_column(notes)

    attendance["student_id"] = attendance[att_id_col].apply(extract_id)
    notes["student_id"] = notes[note_id_col].apply(extract_id)

    st.success(f"Detected Attendance ID Column: {att_id_col}")
    st.success(f"Detected Notes ID Column: {note_id_col}")

    # =========================================================
    # DETECT NOTES COLUMN
    # =========================================================
    note_cols = [c for c in notes.columns if "note" in c]

    if not note_cols:
        st.error(f"No Notes column found. Available: {list(notes.columns)}")
        st.stop()

    note_col = note_cols[0]

    # =========================================================
    # DETECT VISIT COLUMN
    # =========================================================
    visit_cols = [c for c in notes.columns if "visit" in c]

    if not visit_cols:
        st.error(f"No Visit column found. Available: {list(notes.columns)}")
        st.stop()

    visit_col = visit_cols[0]

    # =========================================================
    # FILTER ATTENDANCE VISITS
    # =========================================================
    attendance_visits = notes[
        notes[visit_col].astype(str).str.lower().str.contains("attendance", na=False)
    ].copy()

    st.subheader("Attendance Visit Volume")
    st.write(len(attendance_visits))

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

        if any(x in n for x in ["sick", "medical", "doctor", "ill"]):
            return "Health"

        if any(x in n for x in ["test", "nwea", "assignment", "classwork"]):
            return "School/Academic"

        return "Other"

    attendance_visits["barrier"] = attendance_visits[note_col].apply(classify)

    # =========================================================
    # BARRIER SUMMARY
    # =========================================================
    st.header("🧠 Barrier Intelligence")

    barrier_summary = attendance_visits["barrier"].value_counts().reset_index()
    barrier_summary.columns = ["Barrier", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.bar(barrier_summary, x="Barrier", y="Count"), use_container_width=True)

    with col2:
        st.dataframe(barrier_summary)

    # =========================================================
    # 🔥 RISK ENGINE
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

    max_count = student_risk["barrier_count"].max()
    student_risk["barrier_norm"] = (
        student_risk["barrier_count"] / max_count if max_count > 0 else 0
    )

    student_risk["risk_score"] = (
        student_risk["barrier_norm"] * 40 +
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
    # VISUALS
    # =========================================================
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.histogram(student_risk, x="risk_score"), use_container_width=True)

    with col2:
        st.dataframe(student_risk)

    # =========================================================
    # STUDENT DRILLDOWN
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
    st.info("Upload both Attendance and Notes Excel files to begin.")
