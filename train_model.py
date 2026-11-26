from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Small demonstration dataset only. Replace with a properly labelled,
# representative dataset before using this model for real security decisions.
examples = [
    ("Your account is locked. Verify your password immediately at http://secure-login-example.test", 1),
    ("Payroll update: open the attachment and enter your bank details to receive salary", 1),
    ("Urgent: your mailbox will be deleted. Confirm your credentials now", 1),
    ("You won a gift card. Claim your prize using this shortened link bit.ly/example", 1),
    ("Invoice overdue. Enable macros in the attached document to view payment details", 1),
    ("Unusual sign-in detected. Reset your password using this link", 1),
    ("Your package is on hold. Pay a small fee through this unfamiliar website", 1),
    ("CEO request: buy gift cards and send the codes immediately", 1),
    ("Verify your account to avoid suspension. Login at http://account-check.example.test", 1),
    ("Your tax refund is ready. Submit your bank information today", 1),
    ("Please review the attached meeting agenda before tomorrow's meeting", 0),
    ("The project status meeting is scheduled for 10 AM in room 204", 0),
    ("Your library book is due next Friday. Please return it on time", 0),
    ("Here are the lecture notes from today's class", 0),
    ("Thanks for your report. I have added comments to the shared document", 0),
    ("The office will be closed on the public holiday", 0),
    ("Your appointment is confirmed for Monday at 3 PM", 0),
    ("Monthly newsletter: product updates and upcoming events", 0),
    ("Please find the approved purchase order attached for your records", 0),
    ("The team lunch has been moved to Thursday", 0),
]
texts = [x[0] for x in examples]
labels = [x[1] for x in examples]

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
model.fit(texts, labels)

model_dir = Path(__file__).parent / "models"
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "phishing_model.pkl")
print(f"Demo model saved to {model_dir / 'phishing_model.pkl'}")
print("Reminder: this small synthetic model is for learning and demonstration, not production detection.")
