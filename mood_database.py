import sqlite3
from datetime import datetime
import smtplib
import os

conn = sqlite3.connect("mood_tracking.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS mood_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        emotion TEXT NOT NULL
    )
""")
conn.commit()

print("SQLite Database connected and table created successfully!")

def save_mood_to_db(emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO mood_log (timestamp, emotion) VALUES (?, ?)", (timestamp, emotion))
        conn.commit()
    print(f"Saved: {emotion} at {timestamp}")
    check_stress_alert()


def fetch_last_moods():
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT emotion FROM mood_log ORDER BY timestamp DESC LIMIT 5")
        return [row[0] for row in cursor.fetchall()]

def check_stress_alert():
    stress_emotions = {"sad", "angry", "fear"}
    last_moods = fetch_last_moods()

    if sum(1 for mood in last_moods if mood in stress_emotions) >= 4:
        send_stress_alert()


def send_stress_alert():
    sender_email = os.getenv("EMAIL_USER")  
    receiver_email = os.getenv("HR_EMAIL")  
    password = os.getenv("EMAIL_PASS")  
    if not sender_email or not password:
        print("Email credentials not set. Alert not sent.")
        return

    subject = "Employee Stress Alert!"
    body = "An employee has shown signs of prolonged stress (sad, angry, or fear). Please check in and provide support."

    email_message = f"Subject: {subject}\n\n{body}"

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, email_message)
        print("HR/Manager notified about stress alert!")
    except Exception as e:
        print(f"Error sending email: {e}")


def close_connection():
    conn.close()
    print("Database connection closed!")
