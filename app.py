# =========================================================
# FILTER: ONLY ATTENDANCE VISITS
# =========================================================

# normalize column names just in case
df = notes.copy()
df.columns = [str(c).strip().lower() for c in df.columns]

# try to find columns flexibly
visit_col = None
notes_col = None

for c in df.columns:
    if "visit" in c:
        visit_col = c
    if "note" in c:
        notes_col = c

if visit_col is None or notes_col is None:
    st.error("Could not find Visit Description or Notes columns.")
    st.stop()

# filter attendance-related visits ONLY
attendance_visits = df[df[visit_col].astype(str).str.lower() == "attendance"].copy()

st.subheader("Filtered Attendance Visits Only")
st.dataframe(attendance_visits.head())


# =========================================================
# 🧠 KEYWORD INTELLIGENCE FROM NOTES
# =========================================================

def classify_contact(note):
    note = str(note).lower()

    # CONTACT ACTIONS
    if any(x in note for x in ["voicemail", "voice mail", "left message", "no answer"]):
        return "Voicemail / No Answer"

    if any(x in note for x in ["called", "phone call", "contacted parent", "spoke with parent"]):
        return "Parent Contact Made"

    if any(x in note for x in ["attendance meeting", "intervention", "mtss", "check-in"]):
        return "Attendance Intervention"

    # BARRIERS (NOT TRANSPORTATION FOCUSED)
    if any(x in note for x in ["sick", "ill", "doctor", "hospital"]):
        return "Health Barrier"

    if any(x in note for x in ["family", "care", "babysit", "home situation"]):
        return "Family Barrier"

    if any(x in note for x in ["unknown", "no reason", "not provided"]):
        return "Unknown Reason"

    return "Other / General Note"


attendance_visits["contact_category"] = attendance_visits[notes_col].apply(classify_contact)


# =========================================================
# 📊 CATEGORY BREAKDOWN
# =========================================================
st.subheader("Attendance Contact Pattern Breakdown")

category_counts = attendance_visits["contact_category"].value_counts().reset_index()
category_counts.columns = ["Category", "Count"]

st.plotly_chart(px.bar(category_counts, x="Category", y="Count"))
st.dataframe(category_counts)


# =========================================================
# 🧠 INSIGHT ENGINE
# =========================================================
st.subheader("Executive Insight Summary")

total = len(attendance_visits)
voicemail = len(attendance_visits[attendance_visits["contact_category"] == "Voicemail / No Answer"])
contacted = len(attendance_visits[attendance_visits["contact_category"] == "Parent Contact Made"])

st.write(f"""
- Total Attendance Interventions: {total}
- Successful Parent Contacts: {contacted}
- Unsuccessful Contacts (Voicemail/No Answer): {voicemail}

**Interpretation:**
- High voicemail rate = engagement barrier
- High contact rate = strong outreach effectiveness
- Notes show intervention patterns beyond transportation
""")
