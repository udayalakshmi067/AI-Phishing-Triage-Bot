# AI Phishing Triage Bot

A beginner-friendly defensive cybersecurity project that triages a CSV batch of emails.
It combines a small TF-IDF + Logistic Regression demonstration model with transparent
heuristic indicators for URL patterns, urgency language, and sender format.

## Requirements
- Ubuntu 24.04 or another Python 3 environment
- Python 3, venv, pip

## Setup
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-full
cd AI-Phishing-Triage-Bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Open the local URL printed in the terminal (normally http://localhost:8501).

## CSV format
Required headers: sender,subject,body

See dataset/sample_emails.csv.

## Important limitations
This is an educational prototype. Its small synthetic training set is not suitable for
production, and the displayed risk score is not a calibrated probability. Do not treat
"LOW - Likely safe" as proof of safety. Improve the model with a large, legally obtained,
labelled dataset, evaluate precision/recall and false negatives, and keep human review
in the loop. Do not upload private or confidential emails. Do not automatically click
links, open attachments, or send messages.
