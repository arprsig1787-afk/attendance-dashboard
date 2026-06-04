import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Attendance Intelligence System", layout="wide")
st.title("📊 Attendance Intelligence System (Stable + Scoped NLP)")


# =========================================================
# SAFE UI RENDER
# =========================================================
def safe_ui(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna("")
    return df.astype(str)


# =========================================================
# FORCE UNIQUE COLUMNS
# =========================================================
def force_unique_columns(df):
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
# CLEAN COLUMNS
# =========================================================
def clean_columns(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.astype(str).str.contains("nan", na=False)]
    df = df.loc[:, ~df.columns.isna()]
    return df


# =========================================================
# HEADER FIX
# =========================================================
def fix_headers(df):
    for i in range(min(15, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        if any("student" in x for x in row) or any("note" in x for x in row):
            df.columns = df.iloc[i]
            df = df[i + 1:]
            return df.reset_index(drop=True)
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
# UPLOAD FILES
# =========================================================
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"])
notes_file = st.file_uploader("Upload Notes Excel", type=["xlsx"])


# =========================================================
# MAIN PIPELINE
# =========================================================
if attendance_file and notes_file:

    # -------------------------
    # LOAD
    # -------------------------
    attendance = pd.read_excel(attendance_file, header=None)
    notes = pd.read_excel(notes_file, header=None)

    # -------------------------
    # CLEAN STRUCTURE
    # -------------------------
    attendance = clean_columns(fix_headers(attendance))
    notes = clean_columns(fix_headers(notes))

    attendance = force_unique_columns(attendance)
    notes = force_unique_columns(notes)

    # -------------------------
    # DETECT ID COLUMNS
    # -------------------------
    att_id_col = [c for c in attendance.columns if "student" in c or "name" in c or "id" in c]
    note_id_col = [c for c in notes.columns if "student" in c or "name" in c or "id" in c]

    if not att_id_col or not note_id_col:
        st.error("Could not detect student ID columns.")
        st.stop()

    att_id_col = att_id_col[0]
    note_id_col = note_id_col[0]

    attendance["student_id"] = attendance[att_id_col].apply(extract_id)
    notes["student_id"] = notes[note_id_col].apply(extract_id)

    # -------------------------
    # NOTES COLUMN
    # -------------------------
    note_cols = [c for c in notes.columns if "note" in c]

    if not note_cols:
        st.error(f"No Notes column found. Columns: {notes.columns.tolist()}")
        st.stop()

    notes["notes_text"] = notes[note_cols[0]].astype(str)

    # -------------------------
    # ATTENDANCE COLUMN
    # -------------------------
    pct_cols = [c for c in attendance.columns if "att" in c or "%" in c]

    if pct_cols:
        attendance["attendance_pct"] = pd.to_numeric(attendance[pct_cols[0]], errors="coerce")
    else:
        attendance["attendance_pct"] = 0

    attendance["week"] = 1

    # -------------------------
    # CLEAN FINAL
    # -------------------------
    attendance = attendance.replace([np.inf, -np.inf], np.nan).fillna("")
    notes = notes.replace([np.inf, -np.inf], np.nan).fillna("")

    # -------------------------
    # PREVIEW
    # -------------------------
    st.subheader("Cleaned Data Preview")
    st.dataframe(safe_ui(force_unique_columns(attendance.head())))
    st.dataframe(safe_ui(force_unique_columns(notes.head())))

    # =========================================================
    # 📈 ATTENDANCE TREND
    # =========================================================
    st.header("📈 Attendance Trend")

    trend = attendance.copy()
    trend["attendance_pct"] = pd.to_numeric(trend["attendance_pct"], errors="coerce")

    trend = trend.groupby("week")["attendance_pct"].mean().reset_index()

    if len(trend) > 0:
        st.plotly_chart(px.line(trend, x="week", y="attendance_pct"))

    # =========================================================
    # 🔮 FORECAST
    # =========================================================
    if len(trend) > 1:
        trend["week_num"] = np.arange(len(trend))

        model = LinearRegression()
        model.fit(trend[["week_num"]], trend["attendance_pct"])

        future = np.arange(len(trend) + 4).reshape(-1, 1)
        forecast = model.predict(future)

        st.subheader("Forecast")
        st.plotly_chart(px.line(x=list(range(len(forecast))), y=forecast))

    # =========================================================
    # 📞 ATTENDANCE VISIT FILTER
    # =========================================================
    st.header("📞 Attendance Visit Intelligence")

    notes_clean = notes.copy()
    notes_clean.columns = [str(c).strip().lower() for c in notes_clean.columns]

    visit_col = [c for c in notes_clean.columns if "visit" in c]
    note_col = [c for c in notes_clean.columns if "note" in c]

    if visit_col and note_col:

        visit_col = visit_col[0]
        note_col = note_col[0]

        attendance_visits = notes_clean[
            notes_clean[visit_col].astype(str).str.lower() == "attendance"
        ].copy()

        def classify(note):
            note = str(note).lower()

            if any(x in note for x in ["voicemail", "voice mail", "no answer"]):
                return "Voicemail / No Answer"

            if any(x in note for x in ["called", "contacted", "spoke with"]):
                return "Parent Contact"

            if any(x in note for x in ["intervention", "mtss", "meeting"]):
                return "Intervention"

            if any(x in note for x in ["sick", "ill", "doctor"]):
                return "Health"

            if any(x in note for x in ["family", "care"]):
                return "Family"

            return "Other"

        attendance_visits["category"] = attendance_visits[note_col].apply(classify)

        counts = attendance_visits["category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]

        st.plotly_chart(px.bar(counts, x="Category", y="Count"))
        st.dataframe(counts)

    else:
        st.warning("Visit Description or Notes columns not found.")

    # =========================================================
    # 🧠 NLP (FIXED SCOPE VERSION YOU REQUESTED)
    # =========================================================
    st.header("🧠 Barrier Pattern Discovery (Attendance Only NLP)")

    if "attendance_visits" in locals():

        texts = attendance_visits[note_col].astype(str).tolist()
        texts = [t for t in texts if len(str(t).strip()) > 2]

        if len(texts) > 5:

            vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
            X = vectorizer.fit_transform(texts)

            k = min(5, len(texts))
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = model.fit_predict(X)

            attendance_visits["cluster"] = clusters

            terms = vectorizer.get_feature_names_out()

            cluster_labels = {}

            for i in range(k):
                center = model.cluster_centers_[i]
                top_words = [terms[j] for j in center.argsort()[-3:]]
                cluster_labels[i] = " / ".join(top_words)

            attendance_visits["pattern"] = attendance_visits["cluster"].map(cluster_labels)

            summary = attendance_visits["pattern"].value_counts().reset_index()
            summary.columns = ["Barrier Pattern", "Count"]

            st.plotly_chart(px.bar(summary, x="Barrier Pattern", y="Count"))
            st.dataframe(summary)

        else:
            st.warning("Not enough attendance-only notes for NLP.")

    # =========================================================
    # 🧍 STUDENT VIEW
    # =========================================================
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
