import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# =========================================================
# 📖 DATA STORY
# =========================================================
st.set_page_config(page_title="Data Story", layout="wide")

st.title("📖 Data Story")
st.markdown("Narrative Analytics for Attendance & Intervention Insight")
st.divider()


# =========================================================
# HELPERS
# =========================================================
def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def split_name_id(val):
    if pd.isna(val):
        return "", ""

    text = str(val)

    nums = re.findall(r"\d{3,}", text)
    nums = [n for n in nums if n not in ["2025", "2026"]]

    student_id = nums[-1] if nums else ""

    name = re.sub(r"\d{3,}", "", text)
    name = re.sub(r"[^a-zA-Z ,'-]", " ", name)
    name = " ".join(name.split()).strip()

    return name, student_id


# =========================================================
# UPLOAD
# =========================================================
col1, col2 = st.columns(2)

with col1:
    att_file = st.file_uploader("Attendance File", type=["xlsx"])

with col2:
    note_file = st.file_uploader("Notes File", type=["xlsx"])


# =========================================================
# MAIN
# =========================================================
if att_file and note_file:

    attendance = clean_columns(pd.read_excel(att_file))
    notes = clean_columns(pd.read_excel(note_file))

    # FORCE ID SOURCE
    attendance["raw_identity"] = attendance.iloc[:, 0]
    notes["raw_identity"] = notes.iloc[:, 0]

    attendance[["student_name", "student_id"]] = attendance["raw_identity"].apply(
        lambda x: pd.Series(split_name_id(x))
    )

    notes[["student_name", "student_id"]] = notes["raw_identity"].apply(
        lambda x: pd.Series(split_name_id(x))
    )

    attendance = attendance[attendance["student_id"] != ""]
    notes = notes[notes["student_id"] != ""]

    attendance["student_id"] = attendance["student_id"].astype(str)
    notes["student_id"] = notes["student_id"].astype(str)

    # =====================================================
    # 📖 CHAPTER 1: IDENTITY
    # =====================================================
    st.markdown("## 📖 Chapter 1: Identity Resolution")

    st.dataframe(attendance[["student_name", "student_id"]].head())

    # =====================================================
    # NOTES COLUMN
    # =====================================================
    note_col = [c for c in notes.columns if "note" in c or "visit" in c or "comment" in c]
    note_col = note_col[0] if note_col else None

    if note_col is None:
        st.error("No notes column found")
        st.stop()

    # =====================================================
    # CHAPTER 2: BARRIERS
    # =====================================================
    st.markdown("## 📖 Chapter 2: Barrier Narrative")

    def classify(x):
        x = str(x).lower()

        if "bus" in x or "transport" in x:
            return "Transportation"
        if "voicemail" in x or "called" in x or "left message" in x:
            return "Contact Attempted"
        if "home" in x or "family" in x:
            return "Family/Home"
        if "sick" in x:
            return "Health"
        if "nwea" in x or "test" in x:
            return "School"
        return "Other"

    notes["barrier"] = notes[note_col].apply(classify)

    barrier_summary = notes["barrier"].value_counts().reset_index()
    barrier_summary.columns = ["Barrier", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Page: Barrier Distribution")
        st.plotly_chart(px.bar(barrier_summary, x="Barrier", y="Count"), use_container_width=True)

    with col2:
        st.dataframe(barrier_summary)

    # =====================================================
    # CHAPTER 3: RISK ENGINE
    # =====================================================
    st.markdown("## 📖 Chapter 3: Risk Narrative")

    weights = {
        "Transportation": 0.9,
        "Contact Attempted": 0.7,
        "Family/Home": 0.8,
        "Health": 0.6,
        "School": 0.5,
        "Other": 0.3
    }

    notes["impact"] = notes["barrier"].map(weights)

    student_risk = notes.groupby(["student_id", "student_name"]).agg(
        barrier_count=("barrier", "count"),
        avg_impact=("impact", "mean")
    ).reset_index()

    student_risk["avg_impact"] = student_risk["avg_impact"].fillna(0)

    maxc = student_risk["barrier_count"].max()
    student_risk["norm"] = student_risk["barrier_count"] / maxc if maxc > 0 else 0

    student_risk["risk_score"] = student_risk["norm"] * 40 + student_risk["avg_impact"] * 60

    # =====================================================
    # FIXED DROPDOWN (ALL STUDENTS SHOW)
    # =====================================================
    student_risk["label"] = (
        student_risk["student_name"].fillna("Unknown")
        + " | "
        + student_risk["student_id"].astype(str)
    )

    student_choice = st.selectbox(
        "Select Student",
        student_risk["label"].sort_values().unique()
    )

    selected_id = student_choice.split("|")[-1].strip()

    student_row = student_risk[student_risk["student_id"] == selected_id]

    # =====================================================
    # CHAPTER 3 VISUALS
    # =====================================================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Risk Distribution")
        st.plotly_chart(px.histogram(student_risk, x="risk_score"), use_container_width=True)

    with col2:
        st.markdown("### 📄 Risk Table")
        st.dataframe(student_risk)

    # =====================================================
    # CHAPTER 4: STUDENT STORY
    # =====================================================
    st.markdown("## 📖 Chapter 4: Student Story")

    st.markdown(f"### Selected: {student_row['student_name'].iloc[0]}")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(student_row)

    with col2:
        st.dataframe(notes[notes["student_id"] == selected_id])

else:
    st.info("Upload both files to begin Data Story")
