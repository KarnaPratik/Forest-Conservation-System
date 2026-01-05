import sqlite3

def init_db():
    conn = sqlite3.connect('vanarakshya.db')
    c = conn.cursor()
    
    # 1. Alerts Table (Central Hub for all detections)
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,          -- 'Satellite', 'Vision', 'Audio'
        incident_type TEXT,   -- 'Fire', 'Logging', 'Poaching'
        confidence REAL,
        latitude REAL,
        longitude REAL,
        region TEXT,
        status TEXT DEFAULT 'New', -- 'New', 'Dispatched', 'Resolved'
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    # 2. Deployments Table (Tracks active missions)
    c.execute('''CREATE TABLE IF NOT EXISTS deployments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id INTEGER,
        latitude REAL,        -- Location of the incident
        longitude REAL,       -- Location of the incident
        ranger_name TEXT,
        status TEXT DEFAULT 'In Transit'
    )''')
    
    conn.commit()
    conn.close()
    print("✅ Database Initialized!")

if __name__ == "__main__":
    init_db()