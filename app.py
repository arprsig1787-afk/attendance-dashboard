import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import re

st.set_page_config(page_title="Attendance Intelligence Dashboard", layout="wide")

st.title("📊 Attendance Intelligence Dashboard")

# ----------------------------
# UPLOAD FILES
# ----------------------------
attendance_file = st.file_uploader("Upload Attendance Excel", type=["xlsx"])
barrier_file = st.file_uploader("Upload Barrier Excel", type=["xlsx"])

# ----------------------------
# HELPER: EXTRACT STUDENT ID
# ----------------------------
def extract_id(value):
    if pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else value

# ----------------------------
# MAIN APP
# ----------------------------
if attendance_file and barrier_file:

    # ----------------------------
    # READ ALL SHEETS
    # ----------------------------
    xls = pd.ExcelFile(attendance_file)

    st.write("Available Sheets:", xls.sheet_names)

    # Try to find correct sheet automatically
    sheet_name = None
    for sheet in xls.sheet_names:
        if "week" in sheet.lower() or "attendance" in sheet.lower() or "chronic" in sheet.lower():
            sheet_name = sheet
            break

    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    attendance = pd.read_excel(xls, sheet_name=sheet_name)

    barriers = pd.read_excel(barrier_file)

    # ----------------------------
    # CLEAN COLUMN NAMES
    # ----------------------------
    attendance.columns = attendance.columns.astype(str).str.strip().str.lower()
    barriers.columns = barriers.columns.astype(str).str.strip().str.lower()

    # ----------------------------
    # FIX MISSING STRUCTURE (AUTO-DETECT)
    # ----------------------------

    # Try to find columns even if mislabeled
    possible_id_cols = [c for c in attendance.columns if "id" in c or "student" in c]
    possible_week_cols = [c for c in attendance.columns if "week" in c or "date" in c]
    possible_att_cols = [c for c in attendance.columns if "att" in c or "percent" in c or "%" in c]

    if possible_id_cols:
        attendance["student_id"] = attendance[possible_id_cols[0]].apply(extract_id)

    if possible_week_cols:
        attendance["week"] = attendance[possible_week_cols[0]]

    if possible_att_cols:
        attendance["attendance_pct"] = pd.to_numeric(attendance[possible_att_cols[0]], errors="coerce")

    # ----------------------------
    # CLEAN DUPLICATES
    # ----------------------------
    attendance = attendance.loc[:, ~attendance.columns.duplicated()]
    barriers = barriers.loc[:, ~barriers.columns.duplicated()]

    # ----------------------------
    # PREVIEW
    # ----------------------------
    st.subheader("Raw Data Preview")
    st.dataframe(attendance.head())
    st.dataframe(barriers.head())

    # ----------------------------
    # VALIDATION (SO IT DOESN'T CRASH)
    # ----------------------------
    required = ["student_id", "week", "attendance_pct"]

    missing = [c for c in required if c not in attendance.columns]

    if missing:
        st.error(f"Missing required fields after processing: {missing}")
        st.stop()

    # ----------------------------
    # TREND
    # ----------------------------
    st.header("📈 Attendance Trend")

    trend = attendance.groupby("week")["attendance_pct"].mean().reset_index()

    fig = px.line(trend, x="week", y="attendance_pct",
                  title="Building Attendance Trend Over Time")
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
        "Forecast": forecast
    })

    fig2 = px.line(forecast_df, x="Week", y="Forecast",
                   title="Forecasted Attendance Trend")
    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # BARRIERS
    # ----------------------------
    st.header("🚧 Barriers")

    if "barrier_type" in barriers.columns:
        barrier_counts = barriers["barrier_type"].value_counts().reset_index()
        barrier_counts.columns = ["Barrier", "Count"]

        fig3 = px.bar(barrier_counts, x="Barrier", y="Count")
        st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # INSIGHTS
    # ----------------------------
    st.header("🧠 Insights")

    st.write("System is processing non-standard SIS export format successfully.")

else:
    st.info("Upload both Excel files to begin analysis.")
