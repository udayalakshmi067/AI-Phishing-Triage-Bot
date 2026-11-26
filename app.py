from pathlib import Path
import re
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "phishing_model.pkl"

st.set_page_config(page_title="AI Phishing Triage Bot", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Phishing Triage Bot")
st.caption("Bulk email triage • Explainable indicators • CSV investigation report")

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.I)
SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly")
URGENT_TERMS = (
    "urgent", "immediately", "action required", "suspended", "locked",
    "verify your account", "password", "confirm your credentials",
    "payment failed", "claim your prize", "gift card", "bank details",
    "enable macros", "wire transfer"
)

def analyse_email(row):
    sender = str(row.get("sender", ""))
    subject = str(row.get("subject", ""))
    body = str(row.get("body", ""))
    text = f"{subject} {body}".strip()
    combined = text.lower()

    indicators = []
    urls = URL_RE.findall(text)
    if urls:
        indicators.append(f"{len(urls)} URL(s) found")
    if any(short in url.lower() for url in urls for short in SHORTENERS):
        indicators.append("URL shortener detected")
    if any(term in combined for term in URGENT_TERMS):
        indicators.append("Urgency / credential / payment language")
    if re.search(r"\.(zip|exe|scr|js|vbs|iso)(\s|$)", combined):
        indicators.append("Potentially risky file extension mentioned")
    if sender and ("@" not in sender or sender.count("@") != 1):
        indicators.append("Sender address format looks unusual")
    if re.search(r"@(gmail|yahoo|outlook)\.com$", sender, re.I) and any(
        word in combined for word in ("ceo", "payroll", "invoice", "wire transfer")
    ):
        indicators.append("Free-mail sender with business/payment request")

    # Model probability is one signal, not a verdict.
    try:
        ml_probability = float(model.predict_proba([text])[0][1])
    except Exception:
        ml_probability = 0.0

    # Transparent heuristic contribution; cap the combined score at 100.
    heuristic = 0
    if urls: heuristic += 15
    if any(short in url.lower() for url in urls for short in SHORTENERS): heuristic += 25
    if any(term in combined for term in URGENT_TERMS): heuristic += 25
    if re.search(r"\.(zip|exe|scr|js|vbs|iso)(\s|$)", combined): heuristic += 20
    if "Free-mail sender" in " ".join(indicators): heuristic += 15

    score = round(min(100, max(0, 0.60 * ml_probability * 100 + 0.40 * heuristic)))
    if score >= 70:
        label = "HIGH - Phishing"
    elif score >= 40:
        label = "MEDIUM - Suspicious"
    else:
        label = "LOW - Likely safe"

    return score, label, ml_probability, "; ".join(indicators) if indicators else "No basic indicators detected", " | ".join(urls)

st.markdown("""
**How to use:** Upload a CSV containing `sender`, `subject`, and `body` columns.
Use synthetic or authorised test data only. Do not upload confidential mail.
""")

uploaded = st.file_uploader("Upload email dataset (.csv)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded).fillna("")
        required = {"sender", "subject", "body"}
        if not required.issubset(set(df.columns.str.lower())):
            st.error("CSV must contain these columns: sender, subject, body")
            st.stop()
        df.columns = [c.lower().strip() for c in df.columns]
        if len(df) == 0:
            st.warning("The uploaded CSV has no email rows.")
            st.stop()
        if len(df) > 5000:
            st.error("For this demo, upload at most 5,000 rows at a time.")
            st.stop()

        if st.button("🚀 Analyse Emails", type="primary"):
            results = []
            progress = st.progress(0)
            for i, (_, row) in enumerate(df.iterrows(), start=1):
                score, label, probability, indicators, urls = analyse_email(row)
                results.append({
                    "sender": row["sender"],
                    "subject": row["subject"],
                    "risk_score": score,
                    "classification": label,
                    "model_phishing_probability": round(probability, 3),
                    "indicators": indicators,
                    "urls_found": urls,
                    "body": row["body"]
                })
                progress.progress(i / len(df))
            result_df = pd.DataFrame(results)
            st.session_state["triage_results"] = result_df

    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")

if "triage_results" in st.session_state:
    result_df = st.session_state["triage_results"]
    total = len(result_df)
    high = int((result_df["classification"] == "HIGH - Phishing").sum())
    medium = int((result_df["classification"] == "MEDIUM - Suspicious").sum())
    low = int((result_df["classification"] == "LOW - Likely safe").sum())

    st.divider()
    st.subheader("📊 Triage Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Emails analysed", total)
    c2.metric("High risk", high)
    c3.metric("Medium risk", medium)
    c4.metric("Low risk", low)

    left, right = st.columns(2)
    with left:
        chart_data = pd.DataFrame({"Classification": ["High", "Medium", "Low"],
                                   "Emails": [high, medium, low]})
        st.plotly_chart(px.pie(chart_data, names="Classification", values="Emails",
                               title="Risk distribution"), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(result_df, x="risk_score", nbins=10,
                                     title="Email risk score distribution"),
                        use_container_width=True)

    st.subheader("🔎 Investigation Results")
    risk_filter = st.multiselect(
        "Filter classifications",
        options=result_df["classification"].unique().tolist(),
        default=result_df["classification"].unique().tolist()
    )
    filtered = result_df[result_df["classification"].isin(risk_filter)]
    st.dataframe(filtered.drop(columns=["body"]), use_container_width=True)

    st.subheader("Email detail")
    if not filtered.empty:
        selected = st.selectbox(
            "Choose an email",
            filtered.index,
            format_func=lambda idx: f"{filtered.loc[idx, 'classification']} | {filtered.loc[idx, 'subject'][:70]}"
        )
        item = result_df.loc[selected]
        st.write("**Sender:**", item["sender"])
        st.write("**Subject:**", item["subject"])
        st.write("**Risk score:**", item["risk_score"], "/ 100")
        st.write("**Indicators:**", item["indicators"])
        st.write("**URLs:**", item["urls_found"] or "None detected")
        st.text_area("Email body", item["body"], height=140)

    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download triage report (CSV)", csv_data,
                       file_name="phishing_triage_report.csv", mime="text/csv")
    st.info("A low score does not prove an email is safe. Verify sender identity, links, headers, and attachments manually.")
else:
    st.info("Upload a CSV and click Analyse Emails to view the dashboard.")
