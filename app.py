import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import re

st.set_page_config(page_title="Attendance Intelligence Dashboard", layout="wide")
st.title("📊 Attendance Intelligence Dashboard (Self-Healing SIS Version)")

# ----------------------------
# FILE UPLOAD
# ----------------------------
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"])
notes_file = st.file_uploader("Upload Notes Excel", type=["xlsx"])


# ----------------------------
# CLEAN COLUMNS (FIX DUPLICATES + SPACES)
# ----------------------------
def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip().str.lower()

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
    return df


# ----------------------------
# EXTRACT ID FROM MESSY STRINGS
# ----------------------------
def extract_id(text):
    if pd.isna(text):
        return None
    match = re.search(r"\d+", str(text))
    return match.group(0) if match else str(text).strip()


# ----------------------------
# SIMPLE NLP BARRIER DETECTION
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

    # LOAD
    attendance = pd.read_excel(attendance_file)
    notes = pd.read_excel(notes_file)

    # CLEAN
    attendance = clean_columns(attendance)
    notes = clean_columns(notes)

    st.subheader("Raw Data Preview")
    st.dataframe(attendance.head())
    st.dataframe(notes.head())

    # ----------------------------
    # AUTO-DETECT STUDENT ID (ATTENDANCE)
    # ----------------------------
    id_cols_att = [c for c in attendance.columns if "name" in c or "student" in c or "id" in c]

    if len(id_cols_att) == 0:
        st.error("No student identifier column found in attendance file.")
        st.stop()

    att_id_col = id_cols_att[0]
    attendance["student_id"] = attendance[att_id_col].apply(extract_id)

    # ----------------------------
    # AUTO-DETECT STUDENT ID (NOTES)
    # ----------------------------
    id_cols_notes = [c for c in notes.columns if "name" in c or "student" in c or "id" in c]

    if len(id_cols_notes) == 0:
        st.error("No student identifier column found in notes file.")
        st.stop()

    notes["student_id"] = notes[id_cols_notes[0]].apply(extract_id)

    # ----------------------------
    # FIND NOTES COLUMN FLEXIBLY (FIX FOR "Y1 Notes")
    # ----------------------------
    notes_candidates = [c for c in notes.columns if "note" in c]

    if len(notes_candidates) == 0:
        st.error(f"No Notes column found. Available columns: {notes.columns.tolist()}")
        st.stop()

    notes_col = notes_candidates[0]
    notes["notes_text"] = notes[notes_col].astype(str)

    # ----------------------------
    # DETECT ATTENDANCE STRUCTURE (WIDE OR LONG)
    # ----------------------------
    week_cols = [c for c in attendance.columns if "wk" in c or "week" in c]

    if len(week_cols) > 0:
        attendance = attendance.melt(
            id_vars=[att_id_col, "student_id"],
            value_vars=week_cols,
            var_name="week",
            value_name="attendance_pct"
        )
    else:
        pct_cols = [c for c in attendance.columns if "%" in c or "att" in c]

        if len(pct_cols) == 0:
            st.error("Could not detect attendance percentage column.")
            st.stop()

        attendance["attendance_pct"] = pd.to_numeric(attendance[pct_cols[0]], errors="coerce")

        if "week" not in attendance.columns:
            attendance["week"] = 1

    attendance["attendance_pct"] = pd.to_numeric(attendance["attendance_pct"], errors="coerce")

    # ----------------------------
    # TREND
    # ----------------------------
    st.header("📈 Attendance Trend")

    trend = attendance.groupby("week")["attendance_pct"].mean().reset_index()
    trend = trend.sort_values("week")

    fig = px.line(trend, x="week", y="attendance_pct",
                  title="Attendance Trend Over Time")
    st.plotly_chart(fig, use_container_width=True)

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

    fig2 = px.line(forecast_df, x="Week", y="Predicted Attendance",
                   title="Forecasted Attendance Trend")
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # NOTES → BARRIERS
    # ----------------------------
    st.header("🚧 Barrier Extraction from Notes")

    notes["barrier_type"] = notes["notes_text"].apply(detect_barrier)

    barrier_counts = notes["barrier_type"].value_counts().reset_index()
    barrier_counts.columns = ["Barrier", "Count"]

    st.dataframe(barrier_counts)

    fig3 = px.bar(barrier_counts, x="Barrier", y="Count",
                  title="Barriers Extracted from Notes")
    st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # INSIGHTS
    # ----------------------------
    st.header("🧠 Executive Insights")

    if len(barrier_counts) > 0:
        top_barrier = barrier_counts.iloc[0]["Barrier"]
    else:
        top_barrier = "No data"

    start = trend["attendance_pct"].iloc[0]
    end = trend["attendance_pct"].iloc[-1]
    change = end - start

    st.write(f"""
    - Primary inferred barrier: **{top_barrier}**
    - Attendance change: **{change:.2f}%**
    - System adapts to messy SIS exports (Y1 Notes supported)
    - Notes are automatically categorized into intervention groups
    """)

    # ----------------------------
    # STUDENT VIEW
    # ----------------------------
    st.header("🧍 Student Journey View")

    student = st.selectbox(
        "Select Student ID",
        attendance["student_id"].dropna().unique()
    )

    st.subheader("Attendance History")
    st.dataframe(attendance[attendance["student_id"] == student])

    st.subheader("Notes History")
    st.dataframe(notes[notes["student_id"] == student])

else:
    st.info("Upload BOTH Attendance and Notes Excel files to begin.")
