import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# =========================================================
# 📖 PAGE CONFIG (DATA STORY BRAND)
# =========================================================
st.set_page_config(page_title="Data Story", layout="wide")

st.markdown(
    """
    <div style="text-align:center;">
        <h1>📖 Data Story</h1>
        <h4 style="color:gray;">Narrative Analytics for Attendance & Intervention Insight</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


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
# 🧠 SPLIT NAME + ID (CRITICAL FIX)
# =========================================================
def split_name_id(val):
    if pd.isna(val):
        return "", ""

    text = str(val)

    nums = re.findall(r"\d{4,}", text)

    # remove ZIP/year noise
    nums = [n for n in nums if n not in ["2025", "2026"] and len(n) != 5]

    student_id = max(nums, key=len) if nums else ""

    name = re.sub(r"\d{4,}", "", text)
    name = re.sub(r"[\(\)\-\|,:]", " ", name)
    name = " ".join(name.split()).strip()

    return name, student_id


# =========================================================
# HEADER DETECTION
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
# 📂 UPLOAD SECTION
# =========================================================
st.markdown("## 📂 Upload to Start Your Story")

col1, col2 = st.columns(2)

with col1:
    att_file = st.file_uploader("Attendance File", type=["xlsx"])

with col2:
    note_file = st.file_uploader("Engagements Notes File", type=["xlsx"])


# =========================================================
# MAIN APP
# =========================================================
if att_file and note_file:

    # =====================================================
    # LOAD FILES
    # =====================================================
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

    st.success(f"Attendance Header Row: {att_header}")
    st.success(f"Notes Header Row: {note_header}")

    # =====================================================
    # 📖 CHAPTER 1: IDENTITY RESOLUTION
    # =====================================================
    st.markdown("## 📖 Chapter 1: Identity Resolution")

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

    st.markdown("### 📄 Page 1: Student Identity Cleaned")
    st.dataframe(attendance[["student_name", "student_id"]].head())

    # =====================================================
    # 📖 CHAPTER 2: BARRIER STORY
    # =====================================================
    st.markdown("## 📖 Chapter 2: Barrier Narrative")

    note_col = detect_column(notes, ["note", "comment", "visit", "description"])

    def classify(note):
        n = str(note).lower()

        if any(x in n for x in ["bus", "transport", "ride"]):
            return "Transportation"

        if any(x in n for x in ["voicemail", "called", "left message"]):
            return "Contact Attempted"

        if any(x in n for x in ["family", "home", "housing"]):
            return "Family/Home"

        if any(x in n for x in ["sick", "medical"]):
            return "Health"

        if any(x in n for x in ["test", "nwea", "class"]):
            return "School/Academic"

        return "Other"

    notes["barrier"] = notes[note_col].apply(classify)

    barrier_summary = notes["barrier"].value_counts().reset_index()
    barrier_summary.columns = ["Barrier", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Page 2: Barrier Distribution")
        st.plotly_chart(px.bar(barrier_summary, x="Barrier", y="Count"), use_container_width=True)

    with col2:
        st.markdown("### 📄 Page 3: Barrier Table")
        st.dataframe(barrier_summary)

    # =====================================================
    # 📖 CHAPTER 3: RISK STORY
    # =====================================================
    st.markdown("## 📖 Chapter 3: Attendance Risk Narrative")

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
        st.markdown("### 📄 Page 4: Risk Distribution")
        st.plotly_chart(px.histogram(student_risk, x="risk_score"), use_container_width=True)

    with col2:
        st.markdown("### 📄 Page 5: Risk Table")
        st.dataframe(student_risk)

    # =====================================================
    # 📖 CHAPTER 4: STUDENT STORY VIEW
    # =====================================================
    st.markdown("## 📖 Chapter 4: Student Story")

    students = sorted(student_risk["student_id"].unique())
    student = st.selectbox("Select Student", students)

    student_name = student_risk[
        student_risk["student_id"] == student
    ]["student_name"].iloc[0]

    st.markdown(f"### 📖 Story: {student_name} ({student})")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Page 6: Student Risk Profile")
        st.dataframe(student_risk[student_risk["student_id"] == student])

    with col2:
        st.markdown("### 📄 Page 7: Student Notes Narrative")
        st.dataframe(
            safe_df(notes[notes["student_id"] == student])
        )

else:
    st.info("Upload both Attendance and Notes files to begin your Data Story.")
