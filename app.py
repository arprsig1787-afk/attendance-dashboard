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
st.markdown("Barrier Intelligence • Risk Scoring • Student Insights")


# =========================================================
# SAFE DATA HANDLING
# =========================================================
def safe_df(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna("")
    return df.astype(str)


def clean_columns(df):
    seen = {}
    cols = []

    for c in df.columns:
        c = str(c).strip().lower()
        if c not in seen:
            seen[c] = 0
            cols.append(c)
        else:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")

    df.columns = cols
    return df


# =========================================================
# 🧠 FIXED ID EXTRACTION (NO ZIP CODES / NO ADDRESS NOISE)
# =========================================================
def extract_id(val):
    """
    Robust student ID extraction:
    - ignores ZIP codes (5-digit standalone numbers)
    - ignores house numbers
    - keeps real student IDs embedded in text
    """

    if pd.isna(val):
        return ""

    text = str(val)

    nums = re.findall(r"\d+", text)

    if not nums:
        return text.strip()

    # remove ZIP codes (exact 5-digit standalone numbers)
    nums = [n for n in nums if len(n) > 3 and len(n) != 5]

    if not nums:
        return ""

    return max(nums, key=len)


# =========================================================
# CLEAN MIXED NAME + ID CELLS
# =========================================================
def clean_name_id_column(df, col):
    def clean(x):
        if pd.isna(x):
            return ""

        x = str(x)

        ids = re.findall(r"\d{4,}", x)

        # remove ZIP-like values
        ids = [i for i in ids if len(i) != 5]

        if not ids:
            return x.strip()

        return max(ids, key=len)

    df[col] = df[col].apply(clean)
    return df


# =========================================================
# HEADER DETECTION (FIXED FOR SIS EXPORTS)
# =========================================================
def find_header_row(df):
    for i in range(min(15, len(df))):
        row = df.iloc[i].astype(str).str.lower()

        if any("student" in x for x in row) or any("note" in x for x in row):
            return i

    return 0


# =========================================================
# COLUMN DETECTION
# =========================================================
def detect_column(df, keywords):
    best_col = None
    best_score = -1

    for col in df.columns:
        s = df[col].astype(str).str.lower()
        score = sum(s.str.contains(k, na=False).mean() for k in keywords)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


# =========================================================
# UI
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
    # LOAD RAW FILES
    # -------------------------
    raw_att = pd.read_excel(att_file, header=None)
    raw_notes = pd.read_excel(note_file, header=None)

    att_header = find_header_row(raw_att)
    note_header = find_header_row(raw_notes)

    attendance = pd.read_excel(att_file, header=att_header)
    notes = pd.read_excel(note_file, header=note_header)

    attendance = clean_columns(attendance)
    notes = clean_columns(notes)

    attendance = attendance.fillna("")
    notes = notes.fillna("")

    st.success(f"Attendance header row: {att_header}")
    st.success(f"Notes header row: {note_header}")

    # =========================================================
    # AUTO DETECT ID COLUMNS
    # =========================================================
    att_id_col = detect_column(attendance, ["student", "name", "id"])
    note_id_col = detect_column(notes, ["student", "name", "id"])

    # CLEAN MIXED NAME+ID CELLS FIRST
    attendance = clean_name_id_column(attendance, att_id_col)
    notes = clean_name_id_column(notes, note_id_col)

    # APPLY FINAL IDS
    attendance["student_id"] = attendance[att_id_col].apply(extract_id)
    notes["student_id"] = notes[note_id_col].apply(extract_id)

    # REMOVE BAD IDS (CRITICAL FIX)
    attendance = attendance[attendance["student_id"].str.len() > 3]
    notes = notes[notes["student_id"].str.len() > 3]

    st.success(f"Attendance ID Column: {att_id_col}")
    st.success(f"Notes ID Column: {note_id_col}")

    # =========================================================
    # DETECT NOTES + VISITS
    # =========================================================
    note_col = detect_column(notes, ["note", "comment", "visit", "description"])
    visit_col = detect_column(notes, ["visit", "type", "description"])

    if not note_col or not visit_col:
        st.error("Could not detect Notes or Visit column")
        st.stop()

    # =========================================================
    # DO NOT OVER-FILTER DATA (FIX FOR STUDENT LOSS ISSUE)
    # =========================================================
    attendance_visits = notes.copy()

    # =========================================================
    # BARRIER ENGINE
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
    # STUDENT DRILLDOWN (FIXED)
    # =========================================================
    st.header("🧍 Student Drilldown")

    students = sorted(student_risk["student_id"].dropna().unique())

    student = st.selectbox("Select Student", students)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Profile")
        st.dataframe(student_risk[student_risk["student_id"] == student])

    with col2:
        st.subheader("All Notes")
        st.dataframe(
            safe_df(attendance_visits[attendance_visits["student_id"] == student])
        )

else:
    st.info("Upload both Attendance and Notes files to begin.")
