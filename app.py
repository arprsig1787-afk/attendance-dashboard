import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Attendance Intelligence Dashboard", layout="wide")

st.title("📊 Attendance Intelligence Dashboard (Journey + Notes AI)")

# ----------------------------
# UPLOAD FILES
# ----------------------------
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"])
barrier_file = st.file_uploader("Upload Notes / Barrier Excel", type=["xlsx"])

# ----------------------------
# BAR CODE: SIMPLE NLP MAPPING
# ----------------------------
def detect_barrier(text):
    text = str(text).lower()

    if "transport" in text or "bus" in text:
        return "Transportation"

    if "housing" in text or "homeless" in text or "shelter" in text:
        return "Housing Instability"

    if "sick" in text or "ill" in text or "doctor" in text:
        return "Health"

    if "behavior" in text or "suspension" in text:
        return "Behavior"

    if "work" in text or "job" in text:
        return "Work Conflict"

    if "family" in text or "care" in text or "guardian" in text:
        return "Family Responsibility"

    if "sleep" in text or "tired" in text:
        return "Attendance Fatigue"

    return "Other / Unclear"


# ----------------------------
# MAIN APP
# ----------------------------
if attendance_file and barrier_file:

    # LOAD FILES
    attendance = pd.read_excel(attendance_file)
    barriers = pd.read_excel(barrier_file)

    # CLEAN COLUMNS
    attendance.columns = attendance.columns.astype(str).str.strip().str.lower()
    barriers.columns = barriers.columns.astype(str).str.strip().str.lower()

    # REMOVE DUPLICATES
    attendance = attendance.loc[:, ~attendance.columns.duplicated()]
    barriers = barriers.loc[:, ~barriers.columns.duplicated()]

    st.subheader("Raw Data Preview")
    st.write(attendance.head())
    st.write(barriers.head())

    # ----------------------------
    # CHECK REQUIRED COLUMNS
    # ----------------------------
    if "student_id" not in attendance.columns:
        st.error("Attendance file must contain 'student_id'")
        st.stop()

    if "week" not in attendance.columns:
        st.error("Attendance file must contain 'week'")
        st.stop()

    if "attendance_pct" not in attendance.columns:
        st.error("Attendance file must contain 'attendance_pct'")
        st.stop()

    if "notes" not in barriers.columns:
        st.error("Barrier file must contain 'notes'")
        st.stop()

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
    # FORECASTING
    # ----------------------------
    st.subheader("📊 Forecasting Model")

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
                   title="Forecasted Attendance (Next 4 Weeks)")
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # NOTES → BARRIER EXTRACTION
    # ----------------------------
    st.header("🚧 Barrier Extraction from Notes")

    barriers["barrier_type"] = barriers["notes"].apply(detect_barrier)

    barrier_counts = barriers["barrier_type"].value_counts().reset_index()
    barrier_counts.columns = ["Barrier", "Count"]

    st.dataframe(barrier_counts)

    fig3 = px.bar(barrier_counts, x="Barrier", y="Count",
                  title="AI-Detected Barriers from Notes")
    st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # INSIGHTS
    # ----------------------------
    st.header("🧠 Executive Insights")

    top_barrier = barrier_counts.iloc[0]["Barrier"]

    start = attendance.groupby("week")["attendance_pct"].mean().iloc[0]
    end = attendance.groupby("week")["attendance_pct"].mean().iloc[-1]
    change = end - start

    st.write(f"""
    - Primary inferred barrier: **{top_barrier}**
    - Attendance change: **{change:.2f}%**
    - System is analyzing unstructured notes into structured categories
    - Data is now usable for intervention planning and funding decisions
    """)

    # ----------------------------
    # STUDENT VIEW (JOURNEY LIGHT)
    # ----------------------------
    st.header("🧍 Student View")

    student = st.selectbox("Select Student ID", attendance["student_id"].unique())

    st.subheader("Attendance Journey")
    st.dataframe(attendance[attendance["student_id"] == student])

    st.subheader("Barrier Notes")
    st.dataframe(barriers[barriers["student_id"] == student])

else:
    st.info("Upload BOTH Attendance and Notes Excel files to begin analysis.")
