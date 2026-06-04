import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Attendance Intelligence AI System", layout="wide")
st.title("📊 Attendance Intelligence AI System (NLP Upgrade)")


# =========================================================
# SAFE UI CLEANER (prevents Streamlit JSON crashes)
# =========================================================
def safe_ui(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna("")
    return df.astype(str)


# =========================================================
# HEADER FIX
# =========================================================
def fix_headers(df):
    for i in range(min(15, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        if any("student" in x for x in row) or any("note" in x for x in row):
            df.columns = df.iloc[i]
            df = df[i+1:]
            return df.reset_index(drop=True)
    return df


# =========================================================
# CLEAN COLUMNS
# =========================================================
def clean_columns(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.astype(str).str.contains("nan", na=False)]
    df = df.loc[:, ~df.columns.isna()]
    return df


# =========================================================
# EXTRACT ID
# =========================================================
def extract_id(x):
    if pd.isna(x):
        return ""
    match = re.search(r"\d+", str(x))
    return match.group(0) if match else str(x).strip()


# =========================================================
# FILES
# =========================================================
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"])
notes_file = st.file_uploader("Upload Notes Excel", type=["xlsx"])


# =========================================================
# MAIN
# =========================================================
if attendance_file and notes_file:

    attendance = pd.read_excel(attendance_file, header=None)
    notes = pd.read_excel(notes_file, header=None)

    attendance = clean_columns(fix_headers(attendance))
    notes = clean_columns(fix_headers(notes))

    # =====================================================
    # DETECT STUDENT COLUMN
    # =====================================================
    att_id_col = [c for c in attendance.columns if "student" in str(c) or "name" in str(c) or "id" in str(c)][0]
    note_id_col = [c for c in notes.columns if "student" in str(c) or "name" in str(c) or "id" in str(c)][0]

    attendance["student_id"] = attendance[att_id_col].apply(extract_id)
    notes["student_id"] = notes[note_id_col].apply(extract_id)

    # =====================================================
    # NOTES COLUMN DETECTION
    # =====================================================
    note_cols = [c for c in notes.columns if "note" in str(c).lower()]
    if not note_cols:
        st.error("No notes column found.")
        st.stop()

    notes["notes_text"] = notes[note_cols[0]].astype(str)

    # =====================================================
    # ATTENDANCE CLEAN
    # =====================================================
    pct_cols = [c for c in attendance.columns if "att" in str(c) or "%" in str(c)]

    attendance["attendance_pct"] = pd.to_numeric(attendance[pct_cols[0]], errors="coerce") if pct_cols else 0
    attendance["week"] = 1

    # =====================================================
    # SAFE CLEAN
    # =====================================================
    attendance = attendance.replace([np.inf, -np.inf], np.nan).fillna("")
    notes = notes.replace([np.inf, -np.inf], np.nan).fillna("")

    # =====================================================
    # 📊 PREVIEW
    # =====================================================
    st.subheader("Cleaned Data Preview")
    st.dataframe(safe_ui(attendance.head()))
    st.dataframe(safe_ui(notes.head()))

    # =====================================================
    # 📈 ATTENDANCE TREND
    # =====================================================
    st.header("📈 Attendance Trend")

    trend = attendance.copy()
    trend["attendance_pct"] = pd.to_numeric(trend["attendance_pct"], errors="coerce")

    trend = trend.groupby("week")["attendance_pct"].mean().reset_index()

    if len(trend) > 0:
        st.plotly_chart(px.line(trend, x="week", y="attendance_pct"))

    # =====================================================
    # 🔮 SIMPLE FORECAST
    # =====================================================
    if len(trend) > 1:
        trend["week_num"] = np.arange(len(trend))

        model = LinearRegression()
        model.fit(trend[["week_num"]], trend["attendance_pct"])

        future = np.arange(len(trend) + 4).reshape(-1, 1)
        forecast = model.predict(future)

        st.subheader("Forecast")
        st.plotly_chart(px.line(x=list(range(len(forecast))), y=forecast))

    # =====================================================
    # 🧠 NLP CLUSTERING ENGINE (THIS FIXES "OTHER")
    # =====================================================
    st.header("🧠 Smart Barrier Intelligence (NLP)")

    texts = notes["notes_text"].astype(str).tolist()

    if len(texts) > 5:

        vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
        X = vectorizer.fit_transform(texts)

        k = min(6, len(texts))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = model.fit_predict(X)

        notes["cluster"] = clusters

        # cluster labels (top words per cluster)
        terms = vectorizer.get_feature_names_out()

        cluster_labels = {}

        for i in range(k):
            center = model.cluster_centers_[i]
            top_words = [terms[j] for j in center.argsort()[-3:]]
            cluster_labels[i] = " / ".join(top_words)

        notes["cluster_label"] = notes["cluster"].map(cluster_labels)

        # =================================================
        # CLUSTER VIEW
        # =================================================
        cluster_counts = notes["cluster_label"].value_counts().reset_index()
        cluster_counts.columns = ["Barrier Pattern", "Count"]

        st.subheader("Emerging Barrier Patterns (AI Discovered)")
        st.plotly_chart(px.bar(cluster_counts, x="Barrier Pattern", y="Count"))

        st.dataframe(safe_ui(cluster_counts))

        # =================================================
        # SHOW WHAT WAS PREVIOUSLY 'OTHER'
        # =================================================
        st.subheader("Cluster Breakdown (Replaces 'Other')")

        st.dataframe(
            safe_ui(notes[["notes_text", "cluster_label"]].head(50))
        )

    else:
        st.warning("Not enough notes for clustering")

    # =====================================================
    # 🧍 STUDENT VIEW
    # =====================================================
    st.header("🧍 Student View")

    students = attendance["student_id"].astype(str)
    students = students[students != ""].unique()

    if len(students) > 0:
        student = st.selectbox("Select Student", students)

        st.subheader("Attendance")
        st.dataframe(safe_ui(attendance[attendance["student_id"] == student]))

        st.subheader("Notes")
        st.dataframe(safe_ui(notes[notes["student_id"] == student]))

else:
    st.info("Upload both files to begin.")
