# =========================================================
# 🔥 PREDICTIVE ATTENDANCE RISK ENGINE
# =========================================================

st.header("🔥 Predictive Attendance Risk Engine")

if "attendance_visits" in locals() and len(attendance_visits) > 0:

    # -----------------------------
    # 1. AGGREGATE STUDENT DATA
    # -----------------------------
    student_barriers = attendance_visits.groupby("student_id").agg(
        barrier_count=("barrier", "count"),
        avg_impact=("impact_score", "mean")
    ).reset_index()

    # Fill missing values
    student_barriers["barrier_count"] = student_barriers["barrier_count"].fillna(0)
    student_barriers["avg_impact"] = student_barriers["avg_impact"].fillna(0)

    # -----------------------------
    # 2. ATTENDANCE SIGNAL
    # -----------------------------
    if "attendance_pct" in attendance.columns:
        att_signal = attendance.groupby("student_id")["attendance_pct"].mean().reset_index()
    else:
        att_signal = pd.DataFrame({
            "student_id": student_barriers["student_id"],
            "attendance_pct": 0
        })

    # -----------------------------
    # 3. MERGE DATA
    # -----------------------------
    risk_df = pd.merge(student_barriers, att_signal, on="student_id", how="left")

    risk_df["attendance_pct"] = risk_df["attendance_pct"].fillna(0)

    # -----------------------------
    # 4. NORMALIZE COMPONENTS
    # -----------------------------
    # lower attendance = higher risk
    risk_df["attendance_risk"] = 1 - (risk_df["attendance_pct"] / 100)

    # normalize barrier count
    if risk_df["barrier_count"].max() > 0:
        risk_df["barrier_risk"] = risk_df["barrier_count"] / risk_df["barrier_count"].max()
    else:
        risk_df["barrier_risk"] = 0

    # impact already 0–1
    risk_df["impact_risk"] = risk_df["avg_impact"].fillna(0)

    # -----------------------------
    # 5. FINAL RISK SCORE
    # -----------------------------
    risk_df["risk_score"] = (
        (risk_df["attendance_risk"] * 0.5) +
        (risk_df["barrier_risk"] * 0.3) +
        (risk_df["impact_risk"] * 0.2)
    ) * 100

    # -----------------------------
    # 6. RISK LABELING
    # -----------------------------
    def risk_label(score):
        if score >= 75:
            return "Critical 🔴"
        elif score >= 50:
            return "High 🟠"
        elif score >= 25:
            return "Medium 🟡"
        else:
            return "Low 🟢"

    risk_df["risk_level"] = risk_df["risk_score"].apply(risk_label)

    # -----------------------------
    # 7. PRIMARY DRIVER (EXPLANATION)
    # -----------------------------
    def driver(row):
        if row["barrier_count"] == 0:
            return "Low data / No barriers logged"

        if row["avg_impact"] >= 0.8:
            return "High-impact barrier (transportation/family)"

        if row["attendance_pct"] < 80:
            return "Chronic attendance decline"

        return "Mixed attendance risk"

    risk_df["primary_driver"] = risk_df.apply(driver, axis=1)

    # -----------------------------
    # 8. SORT & DISPLAY
    # -----------------------------
    risk_df = risk_df.sort_values("risk_score", ascending=False)

    st.subheader("Student Risk Table")

    st.dataframe(
        risk_df[[
            "student_id",
            "attendance_pct",
            "barrier_count",
            "avg_impact",
            "risk_score",
            "risk_level",
            "primary_driver"
        ]]
    )

    # -----------------------------
    # 9. VISUALIZATION
    # -----------------------------
    st.subheader("Risk Distribution")

    st.plotly_chart(
        px.histogram(risk_df, x="risk_score", nbins=20)
    )

    st.subheader("Top At-Risk Students")

    st.dataframe(
        risk_df[risk_df["risk_score"] >= 60].head(25)
    )

else:
    st.warning("Risk engine requires attendance_visits data.")
