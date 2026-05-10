"""
==========================================================
DATABASE MODULE (SQLite) - PRODUCTION READY
==========================================================
Tables:
  users       - user accounts
  predictions - all prediction records with input/output
  chat_history - chatbot conversation logs
==========================================================
"""
import os
import sqlite3
import json
from datetime import datetime

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(DB_DIR), "data", "churn_app.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_db():
    """Get a database connection with Row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            input_data TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL,
            explanation TEXT,
            recommendations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
    print("[DB] Database initialized:", DB_PATH)


# ══════════════════════════════════════════
# USER OPERATIONS
# ══════════════════════════════════════════
def create_user(username, email, password_hash):
    """Create a new user account. Raises ValueError if username/email taken."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        user = conn.execute(
            "SELECT id, username, email FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        return dict(user)
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            raise ValueError("Username already exists")
        raise ValueError("Email already exists")
    finally:
        conn.close()


def get_user_by_username(username):
    """Fetch a user by username. Returns dict or None."""
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return dict(user) if user else None


# ══════════════════════════════════════════
# PREDICTION OPERATIONS
# ══════════════════════════════════════════
def save_prediction(user_id, input_data, prediction, probability,
                    explanation=None, recommendations=None):
    """Save a prediction record to the database with accurate local timestamp."""
    conn = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """INSERT INTO predictions
           (user_id, input_data, prediction, probability, explanation, recommendations, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            json.dumps(input_data),
            int(prediction),
            float(probability),
            json.dumps(explanation) if explanation else None,
            json.dumps(recommendations) if recommendations else None,
            now,
        )
    )
    conn.commit()
    conn.close()


def get_prediction_history(user_id=None, limit=50):
    """Get predictions for a specific user."""
    conn = get_db()
    if user_id:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return [_format_prediction(r) for r in rows]


def get_all_prediction_history(limit=100):
    """Get ALL predictions (public, no user filter)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [_format_prediction(r) for r in rows]


def _format_prediction(row):
    """Format a prediction row dict, parsing JSON fields and formatting timestamp."""
    d = dict(row)
    # Parse JSON fields safely
    for field in ("input_data", "explanation", "recommendations"):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    # Format timestamp as readable local time string
    if d.get("created_at"):
        try:
            dt = datetime.strptime(str(d["created_at"]), "%Y-%m-%d %H:%M:%S")
            d["display_time"] = dt.strftime("%d %b %Y, %I:%M:%S %p")
        except (ValueError, TypeError):
            d["display_time"] = str(d["created_at"])
    else:
        d["display_time"] = "-"
    return d


def clear_all_history():
    """Delete ALL prediction records. Returns count deleted."""
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return count


def clear_user_history(user_id):
    """Delete predictions for a specific user. Returns count deleted."""
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) as c FROM predictions WHERE user_id = ?", (user_id,)
    ).fetchone()["c"]
    conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return count


# ══════════════════════════════════════════
# DASHBOARD STATS
# ══════════════════════════════════════════
def get_dashboard_stats():
    """Get aggregated statistics for the dashboard."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
    churn = conn.execute(
        "SELECT COUNT(*) as c FROM predictions WHERE prediction = 1"
    ).fetchone()["c"]
    users_count = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]

    # Monthly trend
    monthly = conn.execute("""
        SELECT strftime('%Y-%m', created_at) as month,
               COUNT(*) as total,
               SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as churned
        FROM predictions
        GROUP BY month ORDER BY month DESC LIMIT 12
    """).fetchall()

    conn.close()
    return {
        "total_predictions": total,
        "total_churn": churn,
        "total_no_churn": total - churn,
        "churn_rate": round(churn / total * 100, 1) if total > 0 else 0,
        "total_users": users_count,
        "monthly_trend": [dict(m) for m in monthly],
    }


# ══════════════════════════════════════════
# CHAT OPERATIONS
# ══════════════════════════════════════════
def save_chat(user_id, message, response):
    """Save a chatbot message/response pair with accurate local timestamp."""
    conn = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO chat_history (user_id, message, response, created_at) VALUES (?, ?, ?, ?)",
        (user_id, message, response, now)
    )
    conn.commit()
    conn.close()


# Initialize on import
init_db()
