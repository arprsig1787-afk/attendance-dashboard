import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Attendance Intelligence Dashboard", layout="wide")

st.title("📊 Attendance Intelligence Dashboard")

# ----------------------------
# Upload Files
# ----------------------------
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"])
barrier_file = st.file_uploader("Upload Barrier/Intervention Excel", type=["xlsx"])

if attendance_file and barrier_file:

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    attendance = pd.read_excel(attendance_file)
    barriers = pd.read_excel(barrier_file)

    # ----------------------------
    # CLEAN COLUMN NAMES (IMPORTANT FIX)
    # ----------------------------
    attendance.columns = attendance.columns.astype(str).str.strip()
    barriers.columns = barriers.columns.astype(str).str.strip()

    # ----------------------------
    # REMOVE DUPLICATE COLUMNS (CRASH FIX)
    # ----------------------------
    attendance = attendance.loc[:, ~attendance.columns.duplicated()]
    barriers = barriers.loc[:, ~barriers.columns.duplicated()]

    # ----------------------------
    # SAFE PREVIEW
    # ----------------------------
    st.subheader("Raw Data Preview")
    st.dataframe(attendance.head())
    st.dataframe(barriers.head())

    # ----------------------------
    # CHECK REQUIRED COLUMNS
    # ----------------------------
    required_attendance = {"student_id", "week", "attendance_pct"}
    required_barriers = {"student_id", "barrier_type"}

    if not required_attendance.issubset(attendance.columns):
        st.error(f"Attendance file must include: {required_attendance}")
        st.stop()

    if not required_barriers.issubset(barriers.columns):
        st.error(f"Barrier file must include: {required_barriers}")
        st.stop()

    # ----------------------------
    # BUILD TREND
    # ----------------------------
    st.header("📈 Attendance Trend")

    trend = attendance.groupby("week")["attendance_pct"].mean().reset_index()
    trend = trend.sort_values("week")

    fig = px.line(trend, x="week", y="attendance_pct",
                  title="Building Attendance Trend Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # FORECASTING
    # ----------------------------
    st.subheader("📊 Forecast Model (Simple Linear Trend)")

    trend = trend.reset_index(drop=True)
    trend["week_num"] = np.arange(len(trend))

    X = trend[["week_num"]]
    y = trend["attendance_pct"]

    model = LinearRegression()
    model.fit(X, y)

    future_weeks = np.arange(len(trend) + 4).reshape(-1, 1)
    forecast = model.predict(future_weeks)

    forecast_df = pd.DataFrame({
        "Week": list(range(len(forecast))),
        "Predicted Attendance": forecast
    })

    fig2 = px.line(forecast_df, x="Week", y="Predicted Attendance",
                   title="Forecasted Attendance (Next 4 Weeks)")
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # BARRIER ANALYSIS
    # ----------------------------
    st.header("🚧 Barrier Analysis")

    barrier_counts = barriers["barrier_type"].value_counts().reset_index()
    barrier_counts.columns = ["Barrier", "Count"]

    fig3 = px.bar(barrier_counts, x="Barrier", y="Count",
                  title="Barrier Frequency Distribution")
    st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # EXECUTIVE INSIGHTS
    # ----------------------------
    st.header("🧠 Executive Insights")

    top_barrier = barrier_counts.iloc[0]["Barrier"]

    start = attendance.groupby("week")["attendance_pct"].mean().iloc[0]
    end = attendance.groupby("week")["attendance_pct"].mean().iloc[-1]
    change = end - start

    st.write(f"""
    - Primary barrier: {top_barrier}  
    - Attendance change over time: {change:.2f}%  
    - Overall trend: {"Improving" if change > 0 else "Declining or Flat"}  
    """)

else:
    st.info("Upload BOTH Attendance and Barrier Excel files to begin analysis.")
