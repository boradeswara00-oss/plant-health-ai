
import sqlite3
from pathlib import Path
from datetime import datetime

# Store the database in the same folder as this Python file
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "plant_history.db"


def get_connection():
    """Create a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def initialize_database():
    """Create the prediction history table if it does not exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time TEXT NOT NULL,
                image_name TEXT NOT NULL,
                predicted_disease TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )


def save_prediction(image_name, predicted_disease, confidence):
    """Save one plant disease prediction permanently."""
    initialize_database()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history
            (prediction_time, image_name, predicted_disease, confidence)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_name,
                predicted_disease,
                float(confidence),
            ),
        )


def get_prediction_history():
    """Return all saved predictions, newest first."""
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT prediction_time, image_name, predicted_disease, confidence
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()

    return [
        {
            "Time": row[0],
            "Image": row[1],
            "Prediction": row[2],
            "Confidence": row[3],
        }
        for row in rows
    ]


def clear_prediction_history():
    """Delete saved prediction records."""
    initialize_database()

    with get_connection() as connection:
        connection.execute("DELETE FROM prediction_history")
