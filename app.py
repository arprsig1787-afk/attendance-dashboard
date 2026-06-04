import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Attendance Intelligence Engine", layout="wide")
st.title("📊 Attendance Intelligence Engine (Barrier Impact Model)")


# =========================================================
# SAFE DATA HANDLING (prevents Streamlit crashes)
# =========================================================
def safe_ui(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna("")
    return df.astype(str)


def force_unique_columns(df):
    cols = {}
    new_cols = []

    for c in df.columns:
        c = str(c).strip().lower()
        if c not in cols:
            cols[c] = 0
            new_cols.append(c)
        else:
            cols[c] += 1
            new_cols.append(f"{c}_{cols[c]}")

    df.columns = new_cols
    return df


def fix_headers(df):
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        if any("student" in x for x in row) or any("note" in x for x in row):
            df.columns = df.iloc[i]
            df = df[i+1:]
            return df.reset_index(drop=True)
    return df


def extract_id(x):
    if pd.isna(x):
        return ""
    match = re.search(r"\d+", str(x))
    return match.group(0) if match else str(x).strip()


# =========================================================
# FILE UPLOAD
# =========================================================
att_file = st.file_uploader("Upload Attendance File", type=["xlsx"])
note_file = st.file_uploader("Upload Notes File", type=["xlsx"])


# =========================================================
# MAIN PIPELINE
# =========================================================
if att_file and note_file:

    attendance = pd.read_excel(att_file, header=None)
    notes = pd.read_excel(note_file, header=None)

    attendance = force_unique_columns(fix_headers(attendance))
    notes = force_unique_columns(fix_headers(notes))

    attendance = attendance.replace([np.inf, -np.inf], np.nan).fillna("")
    notes = notes.replace([np.inf, -np.inf], np.nan).fillna("")

    # ---------------------------
    # ID DETECTION
    # ---------------------------
    att_id = [c for c in attendance.columns if "name" in c or "student" in c or "id" in c][0]
    note_id = [c for c in notes.columns if "name" in c or "student" in c or "id" in c][0]

    attendance["student_id"] = attendance[att_id].apply(extract_id)
    notes["student_id"] = notes[note_id].apply(extract_id)

    # ---------------------------
    # NOTES COLUMN DETECTION
    # ---------------------------
    note_col = [c for c in notes.columns if "note" in c]

    if not note_col:
        st.error("No notes column detected")
        st.stop()

    note_col = note_col[0]
    notes["notes_text"] = notes[note_col].astype(str)

    # ---------------------------
    # VISIT FILTER (SMART)
    # ---------------------------
    visit_col = [c for c in notes.columns if "visit" in c]

    if not visit_col:
        st.error("No visit column detected")
        st.stop()

    visit_col = visit_col[0]

    attendance_visits = notes[
        notes[visit_col].astype(str).str.lower().str.contains("attendance", na=False)
    ].copy()

    st.subheader("Attendance Visit Volume")
    st.write(len(attendance_visits))

    # =========================================================
    # 🧠 BARRIER IMPACT SCORING ENGINE
    # =========================================================

    def classify_barrier(note):
        n = str(note).lower()

        if any(x in n for x in ["bus", "transport", "ride", "pickup", "drop off"]):
            return "Transportation"

        if any(x in n for x in ["voicemail", "no answer", "left message", "called"]):
            return "Contact Attempted"

        if any(x in n for x in ["family", "home", "housing", "guardian"]):
            return "Family/Home"

        if any(x in n for x in ["sick", "doctor", "medical", "ill"]):
            return "Health"

        if any(x in n for x in ["test", "nwea", "classwork", "assignment"]):
            return "School/Academic"

        return "Other"

    attendance_visits["barrier"] = attendance_visits[note_col].apply(classify_barrier)

    # =========================================================
    # 📊 IMPACT SCORING (NEW VALUE ADD)
    # =========================================================

    impact_weights = {
        "Transportation": 0.9,
        "Contact Attempted": 0.7,
        "Family/Home": 0.8,
        "Health": 0.6,
        "School/Academic": 0.5,
        "Other": 0.3
    }

    attendance_visits["impact_score"] = attendance_visits["barrier"].map(impact_weights)

    barrier_summary = attendance_visits.groupby("barrier").agg(
        count=("barrier", "count"),
        avg_impact=("impact_score", "mean")
    ).reset_index()

    barrier_summary["total_weighted_impact"] = barrier_summary["count"] * barrier_summary["avg_impact"]

    barrier_summary = barrier_summary.sort_values("total_weighted_impact", ascending=False)

    st.header("📊 Barrier Impact Dashboard")

    st.dataframe(barrier_summary)

    st.plotly_chart(px.bar(barrier_summary, x="barrier", y="total_weighted_impact"))

    # =========================================================
    # 🧍 STUDENT VIEW
    # =========================================================
    st.header("🧍 Student View")

    students = attendance["student_id"].unique()

    student = st.selectbox("Select Student", students)

    st.subheader("Attendance Records")
    st.dataframe(safe_ui(attendance[attendance["student_id"] == student]))

    st.subheader("Attendance Notes")
    st.dataframe(safe_ui(attendance_visits[attendance_visits["student_id"] == student]))

else:
    st.info("Upload both files to begin")
