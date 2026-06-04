import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import re

st.set_page_config(page_title="Attendance Intelligence Dashboard", layout="wide")

st.title("📊 Attendance Intelligence Dashboard (SIS-Resilient)")

# ----------------------------
# FILE UPLOAD
# ----------------------------
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"])
notes_file = st.file_uploader("Upload Notes Excel", type=["xlsx"])

# ----------------------------
# ID EXTRACTION FUNCTION
# ----------------------------
def extract_id(text):
    """Extract numeric ID from messy strings"""
    if pd.isna(text):
        return None
    match = re.search(r"\d+", str(text))
    return match.group(0) if match else str(text).strip()

# ----------------------------
# BARRIER CLASSIFICATION (RULE-BASED NLP)
# ----------------------------
def detect_barrier(text):
    text = str(text).lower()

    if "bus" in text or "transport" in text:
        return "Transportation"
    if "housing" in text or "homeless" in text:
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

    attendance = pd.read_excel(attendance_file)
    notes = pd.read_excel(notes_file)

    attendance.columns = attendance.columns.astype(str).str.strip().str.lower()
    notes.columns = notes.columns.astype(str).str.strip().str.lower()

    st.subheader("Raw Data Preview")
    st.write(attendance.head())
    st.write(notes.head())

    # ----------------------------
    # AUTO-DETECT STUDENT ID (CRITICAL FIX)
    # ----------------------------
    if "student_id" not in attendance.columns:

        # Try to find any column containing name/id
        possible_cols = [c for c in attendance.columns if "name" in c or "student" in c]

        if len(possible_cols) == 0:
            st.error("No student identifier column found in attendance file.")
            st.stop()

        col = possible_cols[0]

        attendance["student_id"] = attendance[col].apply(extract_id)

    # SAME FIX FOR NOTES FILE
    if "student_id" not in notes.columns:

        possible_cols = [c for c in notes.columns if "name" in c or "student" in c]

        if len(possible_cols) == 0:
            st.error("No student identifier column found in notes file.")
            st.stop()

        col = possible_cols[0]

        notes["student_id"] = notes[col].apply(extract_id)

    # ----------------------------
    # FIND WEEK + ATTENDANCE
    # ----------------------------
    week_col = [c for c in attendance.columns if "week" in c]
    att_col = [c for c in attendance.columns if "att" in c or "%" in c]

    if len(week_col) == 0 or len(att_col) == 0:
        st.error("Could not detect week or attendance percentage columns.")
        st.stop()

    attendance["week"] = attendance[week_col[0]]
    attendance["attendance_pct"] = pd.to_numeric(attendance[att_col[0]], errors="coerce")

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

    notes["notes"] = notes["notes"].astype(str)
    notes["barrier_type"] = notes["notes"].apply(detect_barrier)

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
    - Primary barrier: **{top_barrier}**
    - Attendance change: **{change:.2f}%**
    - System is extracting insights from unstructured SIS notes
    """)

    # ----------------------------
    # STUDENT VIEW
    # ----------------------------
    st.header("🧍 Student Journey")

    student = st.selectbox("Select Student ID", attendance["student_id"].dropna().unique())

    st.subheader("Attendance History")
    st.dataframe(attendance[attendance["student_id"] == student])

    st.subheader("Notes History")
    st.dataframe(notes[notes["student_id"] == student])

else:
    st.info("Upload BOTH files to begin analysis.")
