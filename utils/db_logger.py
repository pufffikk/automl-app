import sqlite3
from datetime import datetime

DB_PATH = 'runs_history.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            accuracy REAL,
            roc_auc REAL,
            additional_info TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_run(model_name, accuracy, roc_auc, additional_info=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO runs (timestamp, model_name, accuracy, roc_auc, additional_info)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.utcnow().isoformat(), model_name, accuracy, roc_auc, additional_info))
    conn.commit()
    conn.close()

def get_all_runs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM runs ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return rows
