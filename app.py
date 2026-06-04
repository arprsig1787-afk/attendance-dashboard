import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import re

st.set_page_config(page_title="Attendance Intelligence Dashboard", layout="wide")
st.title("📊 Attendance Intelligence Dashboard (Production Stable Version)")


# ----------------------------
# FILE UPLOAD
# ----------------------------
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"])
notes_file = st.file_uploader("Upload Notes Excel", type=["xlsx"])


# ----------------------------
# GLOBAL CLEANER (CRITICAL FIX FOR NaN JSON CRASH)
# ----------------------------
def clean_for_streamlit(df):
    df = df.copy()

    # remove broken numeric values
    df = df.replace([np.inf, -np.inf], np.nan)

    # drop empty columns
    df = df.dropna(axis=1, how="all")

    # fill ALL NaN for UI safety
    df = df.fillna("")

    return df


# ----------------------------
# CLEAN COLUMNS (SAFE + UNIQUE)
# ----------------------------
def clean_columns(df):
    df.columns = [str(c).strip().lower() for c in df.columns]

    new_cols = []
    seen = {}

    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")

    df.columns = new_cols

    # remove junk columns
    df = df.loc[:, ~df.columns.astype(str).str.contains("nan", na=False)]

    return df


# ----------------------------
# HEADER FIX (SIS REPORT DETECTION)
# ----------------------------
def fix_headers(df):
    for i in range(min(15, len(df))):
        row = df.iloc[i].astype(str).str.lower()

        if any("student" in x for x in row) or any("note" in x for x in row):
            df.columns = df.iloc[i]
            df = df[i + 1:]
            return df.reset_index(drop=True)

    return df


# ----------------------------
# EXTRACT ID FROM MESSY STRINGS
# ----------------------------
def extract_id(text):
    if pd.isna(text):
        return ""
    match = re.search(r"\d+", str(text))
    return match.group(0) if match else str(text).strip()


# ----------------------------
# BARRIER DETECTION
# ----------------------------
def detect_barrier(text):
    text = str(text).lower()

    if "bus" in text or "transport" in text:
        return "Transportation"
    if "housing" in text or "homeless" in text or "shelter" in text:
        return "Housing Instability"
    if "sick" in text or "doctor" in text or "ill" in text:
        return "Health"
    if "behavior" in text or "suspension" in text:
        return "Behavior"
    if "work" in text or "job" in text:
        return "Work Conflict"
    if "family" in text or "care" in text:
        return "Family Responsibility"
    if "sleep" in text or "tired" in text:
        return "Attendance Fatigue"

    return "Other / Unclear"


# ----------------------------
# MAIN APP
# ----------------------------
if attendance_file and notes_file:

    # LOAD RAW
    attendance = pd.read_excel(attendance_file, header=None)
    notes = pd.read_excel(notes_file, header=None)

    # FIX HEADERS
    attendance = fix_headers(attendance)
    notes = fix_headers(notes)

    # CLEAN STRUCTURE
    attendance = clean_columns(attendance)
    notes = clean_columns(notes)

    st.subheader("Cleaned Preview")
    st.dataframe(clean_for_streamlit(attendance.head()))
    st.dataframe(clean_for_streamlit(notes.head()))

    # ----------------------------
    # STUDENT ID DETECTION (ATTENDANCE)
    # ----------------------------
    id_cols_att = [c for c in attendance.columns if "student" in str(c) or "name" in str(c) or "id" in str(c)]

    if len(id_cols_att) == 0:
        st.error("No student column found in attendance.")
        st.stop()

    att_id_col = id_cols_att[0]
    attendance["student_id"] = attendance[att_id_col].apply(extract_id)

    # ----------------------------
    # STUDENT ID DETECTION (NOTES)
    # ----------------------------
    id_cols_notes = [c for c in notes.columns if "student" in str(c) or "name" in str(c) or "id" in str(c)]

    if len(id_cols_notes) == 0:
        st.error("No student column found in notes.")
        st.stop()

    notes["student_id"] = notes[id_cols_notes[0]].apply(extract_id)

    # ----------------------------
    # NOTES COLUMN FIX
    # ----------------------------
    notes_cols = [c for c in notes.columns if "note" in str(c)]

    if len(notes_cols) == 0:
        st.error(f"No Notes column found. Columns: {notes.columns.tolist()}")
        st.stop()

    notes_col = notes_cols[0]
    notes["notes_text"] = notes[notes_col].astype(str)

    # ----------------------------
    # ATTENDANCE STRUCTURE
    # ----------------------------
    week_cols = [c for c in attendance.columns if "wk" in str(c) or "week" in str(c)]

    if len(week_cols) > 0:

        attendance = attendance.melt(
            id_vars=[att_id_col, "student_id"],
            value_vars=week_cols,
            var_name="week",
            value_name="attendance_pct"
        )

    else:
        pct_cols = [c for c in attendance.columns if "%" in str(c) or "att" in str(c)]

        if len(pct_cols) == 0:
            st.error("Could not detect attendance percentage column.")
            st.stop()

        attendance["attendance_pct"] = attendance[pct_cols[0]]

        if "week" not in attendance.columns:
            attendance["week"] = 1

    # CLEAN NUMERIC
    attendance["attendance_pct"] = pd.to_numeric(attendance["attendance_pct"], errors="coerce")

    # FINAL SAFE CLEAN
    attendance = clean_for_streamlit(attendance)
    notes = clean_for_streamlit(notes)

    # ----------------------------
    # TREND
    # ----------------------------
    st.header("📈 Attendance Trend")

    trend = attendance.groupby("week")["attendance_pct"].mean().reset_index()
    trend = trend.sort_values("week")

    st.plotly_chart(px.line(trend, x="week", y="attendance_pct"), use_container_width=True)

    # ----------------------------
    # FORECAST
    # ----------------------------
    st.subheader("📊 Forecast")

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

    st.plotly_chart(px.line(forecast_df, x="Week", y="Predicted Attendance"), use_container_width=True)

    # ----------------------------
    # BARRELS
    # ----------------------------
    st.header("🚧 Barrier Analysis")

    notes["barrier_type"] = notes["notes_text"].apply(detect_barrier)

    barrier_counts = notes["barrier_type"].value_counts().reset_index()
    barrier_counts.columns = ["Barrier", "Count"]

    barrier_counts = clean_for_streamlit(barrier_counts)

    st.dataframe(barrier_counts)

    st.plotly_chart(px.bar(barrier_counts, x="Barrier", y="Count"), use_container_width=True)

    # ----------------------------
    # INSIGHTS
    # ----------------------------
    st.header("🧠 Insights")

    top_barrier = barrier_counts.iloc[0]["Barrier"] if len(barrier_counts) > 0 else "None"

    st.write(f"""
    - Top Barrier: {top_barrier}
    - System is stable with messy SIS exports
    - NaN-safe rendering enabled
    """)

    # ----------------------------
    # STUDENT VIEW (SAFE)
    # ----------------------------
    st.header("🧍 Student View")

    students = attendance["student_id"].dropna().astype(str).unique()

    if len(students) > 0:
        student = st.selectbox("Select Student", students)

        st.subheader("Attendance")
        st.dataframe(clean_for_streamlit(attendance[attendance["student_id"] == student]))

        st.subheader("Notes")
        st.dataframe(clean_for_streamlit(notes[notes["student_id"] == student]))

else:
    st.info("Upload both files to begin.")
