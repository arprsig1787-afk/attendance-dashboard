import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Data Story", layout="wide")

st.title("📊 Attendance Intelligence System")
st.markdown("Barrier Intelligence • Risk Scoring • Student Insights")


# =========================================================
# SAFE HELPERS
# =========================================================
def safe_df(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna("")
    return df


def clean_columns(df):
    df = df.copy()
    cols = []
    seen = {}

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
# 🧠 CORE FIX: SPLIT NAME + ID (ROBUST)
# =========================================================
def split_name_id(val):
    """
    Handles:
    - John Smith 12078
    - 12078 - John Smith
    - Smith, John (12078)
    """

    if pd.isna(val):
        return "", ""

    text = str(val)

    # extract all numeric groups
    nums = re.findall(r"\d{4,}", text)

    # remove year-like / zip-like values
    nums = [n for n in nums if n not in ["2025", "2026"] and len(n) != 5]

    student_id = max(nums, key=len) if nums else ""

    # clean name
    name = re.sub(r"\d{4,}", "", text)
    name = re.sub(r"[\(\)\-\|,:]", " ", name)
    name = " ".join(name.split()).strip()

    return name, student_id


# =========================================================
# HEADER DETECTION (SIS SAFE)
# =========================================================
def find_header_row(df):
    for i in range(min(15, len(df))):
        row = df.iloc[i].astype(str).str.lower()

        if any("student" in x for x in row) or any("note" in x for x in row):
            return i

    return 0


# =========================================================
# COLUMN DETECTION (NOT USED FOR ID ANYMORE)
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
    # 🔥 CRITICAL FIX: DO NOT USE DETECTED ID COLUMNS
    # =========================================================

    # force a stable identity source (usually first column)
    attendance["raw_identity"] = attendance.iloc[:, 0]
    notes["raw_identity"] = notes.iloc[:, 0]

    # split name + ID
    attendance[["student_name", "student_id"]] = attendance["raw_identity"].apply(
        lambda x: pd.Series(split_name_id(x))
    )

    notes[["student_name", "student_id"]] = notes["raw_identity"].apply(
        lambda x: pd.Series(split_name_id(x))
    )

    # remove bad IDs
    attendance = attendance[attendance["student_id"] != ""]
    notes = notes[notes["student_id"] != ""]

    # final cleanup
    attendance["student_id"] = attendance["student_id"].astype(str)
    notes["student_id"] = notes["student_id"].astype(str)

    # remove year contamination
    attendance = attendance[~attendance["student_id"].isin(["2025", "2026"])]
    notes = notes[~notes["student_id"].isin(["2025", "2026"])]

    st.success("Student identity extraction complete")

    # =========================================================
    # NOTES COLUMN DETECTION
    # =========================================================
    note_col = detect_column(notes, ["note", "comment", "visit", "description"])

    if not note_col:
        st.error("Could not detect notes column")
        st.stop()

    # =========================================================
    # BARRIER CLASSIFICATION
    # =========================================================
    def classify(note):
        n = str(note).lower()

        if any(x in n for x in ["bus", "transport", "ride", "pickup"]):
            return "Transportation"

        if any(x in n for x in ["voicemail", "no answer", "called", "left message"]):
            return "Contact Attempted"

        if any(x in n for x in ["family", "home", "housing"]):
            return "Family/Home"

        if any(x in n for x in ["sick", "doctor", "medical"]):
            return "Health"

        if any(x in n for x in ["test", "nwea", "class", "assignment"]):
            return "School/Academic"

        return "Other"

    notes["barrier"] = notes[note_col].apply(classify)

    # =========================================================
    # BARRIER SUMMARY
    # =========================================================
    st.header("🧠 Barrier Intelligence")

    barrier_summary = notes["barrier"].value_counts().reset_index()
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

    notes["impact"] = notes["barrier"].map(weights)

    student_risk = notes.groupby(["student_id", "student_name"]).agg(
        barrier_count=("barrier", "count"),
        avg_impact=("impact", "mean")
    ).reset_index()

    student_risk["avg_impact"] = student_risk["avg_impact"].fillna(0)

    max_count = student_risk["barrier_count"].max()
    student_risk["barrier_norm"] = student_risk["barrier_count"] / max_count if max_count > 0 else 0

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

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.histogram(student_risk, x="risk_score"), use_container_width=True)

    with col2:
        st.dataframe(student_risk)

    # =========================================================
    # STUDENT DRILLDOWN (FIXED)
    # =========================================================
    st.header("🧍 Student Drilldown")

    students = sorted(student_risk["student_id"].unique())

    student = st.selectbox("Select Student", students)

    student_name = student_risk[
        student_risk["student_id"] == student
    ]["student_name"].iloc[0]

    st.subheader(f"{student_name} ({student})")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Profile")
        st.dataframe(student_risk[student_risk["student_id"] == student])

    with col2:
        st.subheader("Notes")
        st.dataframe(
            safe_df(notes[notes["student_id"] == student])
        )

else:
    st.info("Upload both files to begin.")
