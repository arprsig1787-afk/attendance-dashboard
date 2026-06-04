import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import re

st.set_page_config(page_title="Attendance Intelligence System", layout="wide")
st.title("📊 Attendance Intelligence System (Production Grade)")


# =========================================================
# 🔥 GLOBAL SAFETY PATCH (PREVENT STREAMLIT JSON CRASHES)
# =========================================================
pd.options.mode.chained_assignment = None


def safe_ui(df):
    """
    CRITICAL: prevents Streamlit JSON rendering crashes
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna("")
    return df.astype(str)


# =========================================================
# 🧼 DATA CLEANING ENGINE
# =========================================================
def clean_columns(df):
    df.columns = [str(c).strip().lower() for c in df.columns]

    # remove bad columns
    df = df.loc[:, ~df.columns.astype(str).str.contains("nan", na=False)]
    df = df.loc[:, ~df.columns.isna()]

    # fix duplicates
    seen = {}
    new_cols = []

    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")

    df.columns = new_cols
    return df


# =========================================================
# 🧠 HEADER DETECTION (FIXES SIS REPORTS)
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
# 🧾 ID EXTRACTION (handles name+id mixed cells)
# =========================================================
def extract_id(x):
    if pd.isna(x):
        return ""
    match = re.search(r"\d+", str(x))
    return match.group(0) if match else str(x).strip()


# =========================================================
# 🚧 BARRIER DETECTION (RULE + NLP LIGHTWEIGHT)
# =========================================================
def detect_barrier(text):
    text = str(text).lower()

    rules = {
        "Transportation": ["bus", "transport", "ride", "car"],
        "Housing Instability": ["housing", "homeless", "shelter"],
        "Health": ["sick", "doctor", "ill", "hospital"],
        "Behavior": ["behavior", "suspension", "office referral"],
        "Work Conflict": ["work", "job", "shift"],
        "Family Responsibility": ["family", "care", "babysit"],
        "Attendance Fatigue": ["tired", "sleep", "late"]
    }

    for k, keywords in rules.items():
        if any(w in text for w in keywords):
            return k

    return "Other"


# =========================================================
# 📥 UPLOADS
# =========================================================
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"])
notes_file = st.file_uploader("Upload Notes Excel", type=["xlsx"])


# =========================================================
# 🚀 MAIN ENGINE
# =========================================================
if attendance_file and notes_file:

    # LOAD RAW
    attendance = pd.read_excel(attendance_file, header=None)
    notes = pd.read_excel(notes_file, header=None)

    # FIX STRUCTURE
    attendance = fix_headers(attendance)
    notes = fix_headers(notes)

    attendance = clean_columns(attendance)
    notes = clean_columns(notes)

    # =====================================================
    # 🧍 STUDENT COLUMN DETECTION
    # =====================================================
    att_id_col = [c for c in attendance.columns if "student" in str(c) or "name" in str(c) or "id" in str(c)]
    note_id_col = [c for c in notes.columns if "student" in str(c) or "name" in str(c) or "id" in str(c)]

    if not att_id_col or not note_id_col:
        st.error("Could not detect student ID columns.")
        st.stop()

    att_id_col = att_id_col[0]
    note_id_col = note_id_col[0]

    attendance["student_id"] = attendance[att_id_col].apply(extract_id)
    notes["student_id"] = notes[note_id_col].apply(extract_id)

    # =====================================================
    # 📝 NOTES COLUMN DETECTION
    # =====================================================
    note_cols = [c for c in notes.columns if "note" in str(c).lower()]

    if not note_cols:
        st.error(f"No notes column found. Columns: {notes.columns.tolist()}")
        st.stop()

    notes["notes_text"] = notes[note_cols[0]].astype(str)

    # =====================================================
    # 📊 ATTENDANCE STRUCTURE
    # =====================================================
    week_cols = [c for c in attendance.columns if "week" in str(c).lower()]

    if week_cols:
        attendance = attendance.melt(
            id_vars=[att_id_col, "student_id"],
            value_vars=week_cols,
            var_name="week",
            value_name="attendance_pct"
        )
    else:
        pct_cols = [c for c in attendance.columns if "%" in str(c) or "att" in str(c)]

        if not pct_cols:
            st.error("No attendance column detected.")
            st.stop()

        attendance["attendance_pct"] = attendance[pct_cols[0]]
        attendance["week"] = 1

    attendance["attendance_pct"] = pd.to_numeric(attendance["attendance_pct"], errors="coerce")

    # =====================================================
    # 🧹 FINAL SAFE CLEAN (CRITICAL)
    # =====================================================
    attendance = attendance.replace([np.inf, -np.inf], np.nan).fillna("")
    notes = notes.replace([np.inf, -np.inf], np.nan).fillna("")

    # =====================================================
    # 📊 DATA PREVIEW (SAFE MODE)
    # =====================================================
    st.subheader("Cleaned Data Preview")
    st.dataframe(safe_ui(attendance.head()))
    st.dataframe(safe_ui(notes.head()))

    # =====================================================
    # 📈 TREND ANALYSIS
    # =====================================================
    st.header("📈 Attendance Trend")

    trend = attendance.copy()
    trend["attendance_pct"] = pd.to_numeric(trend["attendance_pct"], errors="coerce")

    trend = trend.groupby("week")["attendance_pct"].mean().reset_index()

    if len(trend) > 0:
        st.plotly_chart(px.line(trend, x="week", y="attendance_pct"), use_container_width=True)

    # =====================================================
    # 🔮 FORECASTING
    # =====================================================
    st.subheader("📊 Forecast")

    if len(trend) > 1:
        trend = trend.reset_index(drop=True)
        trend["week_num"] = np.arange(len(trend))

        model = LinearRegression()
        model.fit(trend[["week_num"]], trend["attendance_pct"])

        future = np.arange(len(trend) + 4).reshape(-1, 1)
        forecast = model.predict(future)

        forecast_df = pd.DataFrame({
            "Week": range(len(forecast)),
            "Predicted Attendance": forecast
        })

        st.plotly_chart(px.line(forecast_df, x="Week", y="Predicted Attendance"),
                        use_container_width=True)

    # =====================================================
    # 🚧 BARRIER ANALYSIS
    # =====================================================
    st.header("🚧 Barrier Analysis")

    notes["barrier"] = notes["notes_text"].apply(detect_barrier)

    barrier_counts = notes["barrier"].value_counts().reset_index()
    barrier_counts.columns = ["Barrier", "Count"]

    st.plotly_chart(px.bar(barrier_counts, x="Barrier", y="Count"), use_container_width=True)
    st.dataframe(safe_ui(barrier_counts))

    # =====================================================
    # 🧠 INSIGHTS ENGINE
    # =====================================================
    st.header("🧠 Executive Insights")

    top_barrier = barrier_counts.iloc[0]["Barrier"] if len(barrier_counts) else "None"

    st.write(f"""
    - Top Barrier: {top_barrier}
    - System successfully parsed messy SIS exports
    - NaN-safe rendering enabled
    - Forecasting active when enough data exists
    """)

    # =====================================================
    # 🧍 STUDENT VIEW (SAFE)
    # =====================================================
    st.header("🧍 Student Journey View")

    students = attendance["student_id"].astype(str)
    students = students[students != ""].unique()

    if len(students) > 0:
        student = st.selectbox("Select Student", students)

        st.subheader("Attendance")
        st.dataframe(safe_ui(attendance[attendance["student_id"] == student]))

        st.subheader("Notes")
        st.dataframe(safe_ui(notes[notes["student_id"] == student]))

else:
    st.info("Upload both Attendance and Notes Excel files to begin.")
