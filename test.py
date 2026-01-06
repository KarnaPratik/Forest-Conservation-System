# ==============================
# VanaRakshya – Refactored UI
# ==============================

from keras.applications.efficientnet import preprocess_input
from ultralytics import YOLO
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta
import tensorflow as tf
import sqlite3
import tempfile
import requests
import cv2
import gc
import os
from PIL import Image
from audio_to_img import for_single_audio

# ==============================
# CONFIG
# ==============================

st.set_page_config(
    page_title="VanaRakshya",
    page_icon="🌲",
    layout="wide"
)

def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==============================
# DATABASE
# ==============================

def db():
    return sqlite3.connect("vanarakshya.db", check_same_thread=False)

def fetch_df(table):
    try:
        return pd.read_sql(f"SELECT * FROM {table}", db())
    except:
        return pd.DataFrame()

def log_alert(src, typ, conf, lat, lon, region):
    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO alerts (source, incident_type, confidence, latitude, longitude, region)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (src, typ, conf, lat, lon, region))
    con.commit()
    con.close()

def update_status(alert_id, status):
    con = db()
    cur = con.cursor()
    cur.execute("UPDATE alerts SET status=? WHERE id=?", (status, alert_id))
    con.commit()
    con.close()

def deploy(alert_id, lat, lon):
    con = db()
    cur = con.cursor()
    cur.execute("UPDATE alerts SET status='Dispatched' WHERE id=?", (alert_id,))
    cur.execute("""
        INSERT INTO deployments (alert_id, latitude, longitude, ranger_name)
        VALUES (?, ?, ?, ?)
    """, (alert_id, lat, lon, "Nepal Ranger Unit"))
    con.commit()
    con.close()

# ==============================
# MODELS
# ==============================

@st.cache_resource
def load_models():
    v1 = YOLO("models/best_v1.pt")
    v2 = YOLO("models/best_v2.pt")
    a1 = tf.keras.models.load_model("models/audio_forest_69.keras",
                                    custom_objects={'preprocess_input': preprocess_input})
    a2 = tf.keras.models.load_model("models/audio_forest_69420.keras",
                                    custom_objects={'preprocess_input': preprocess_input})
    return v1, v2, a1, a2

vision1, vision2, audio1, audio2 = load_models()

# ==============================
# INFERENCE
# ==============================

def vision_infer_image(file):
    img = np.array(Image.open(file))
    r1 = vision1.predict(img, conf=0.25)[0]
    r2 = vision2.predict(img, conf=0.25)[0]

    det = len(r1.boxes) and len(r2.boxes)
    conf = max(
        float(r1.boxes.conf.max()) if len(r1.boxes) else 0,
        float(r2.boxes.conf.max()) if len(r2.boxes) else 0
    )

    label = "fire" if det or conf > 0.5 else "No detection"
    out = cv2.cvtColor(r1.plot(), cv2.COLOR_BGR2RGB)
    return conf, label, out

def audio_infer(file):
    img, _, _ = for_single_audio(file)
    img = preprocess_input(img)
    p1 = audio1.predict(img)[0][0]
    if p1 < 0.5:
        return 1 - p1, "natural sound"
    p2 = audio2.predict(img)[0]
    idx = np.argmax(p2)
    return float(p2[idx]), ["fire", "logging", "poaching"][idx]

# ==============================
# MOCK DATA
# ==============================

def mock_fires(days=7):
    locs = [
        (27.5, 84.4, "Chitwan"),
        (28.6, 81.3, "Bardiya"),
        (28.9, 80.2, "Shuklaphanta"),
    ]
    rows = []
    for d in range(days):
        for lat, lon, reg in locs:
            if np.random.rand() > 0.3:
                rows.append({
                    "latitude": lat + np.random.uniform(-0.1, 0.1),
                    "longitude": lon + np.random.uniform(-0.1, 0.1),
                    "confidence": np.random.randint(70, 99),
                    "brightness": np.random.randint(335, 370),
                    "region": reg,
                    "date": (datetime.now() - timedelta(days=d)).date()
                })
    return pd.DataFrame(rows)

# ==============================
# SIDEBAR NAV
# ==============================

st.sidebar.title("🌲 VanaRakshya")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Live Map",
        "Alerts & Dispatch",
        "Analytics",
        "Model Testing",
        "Live Surveillance",
        "Audio Sensors"
    ]
)

# ==============================
# DASHBOARD
# ==============================

if page == "Dashboard":
    df = mock_fires()
    st.markdown("## 🌍 System Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Hotspots", len(df))
    c2.metric("High Severity", len(df[df.brightness > 360]))
    c3.metric("Avg Confidence", f"{int(df.confidence.mean())}%")

# ==============================
# LIVE MAP
# ==============================

elif page == "Live Map":
    st.markdown("## 🗺️ Live Hotspot Map")
    df = mock_fires()
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[longitude, latitude]',
        get_radius=12000,
        get_color='[220,38,38,160]'
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=28, longitude=84, zoom=6),
        map_style="dark"
    ))

# ==============================
# ALERTS
# ==============================

elif page == "Alerts & Dispatch":
    st.markdown("## 🚨 Active Alerts")
    alerts = fetch_df("alerts")
    for _, r in alerts.iterrows():
        st.markdown(f"""
        **🔥 {r.incident_type}**  
        {r.region} • {r.status} • {r.confidence:.1%}
        """)
        c1, c2 = st.columns(2)
        if c1.button("Deploy", key=f"d{r.id}"):
            deploy(r.id, r.latitude, r.longitude)
            st.rerun()
        if c2.button("Resolve", key=f"r{r.id}"):
            update_status(r.id, "Resolved")
            st.rerun()

# ==============================
# MODEL TESTING
# ==============================

elif page == "Model Testing":
    st.markdown("## 🧪 Test Models")
    img = st.file_uploader("Image", type=["jpg", "png"])
    if img:
        conf, lbl, out = vision_infer_image(img)
        st.image(out)
        st.metric("Confidence", f"{conf:.1%}")
        if conf > 0.5:
            log_alert("Vision AI", lbl, conf, 27.7, 84.3, "Test Zone")

# ==============================
# LIVE SURVEILLANCE
# ==============================

elif page == "Live Surveillance":
    st.markdown("## 📹 Live Camera Feed")
    ip = st.text_input("Camera IP")
    if st.button("Connect"):
        requests.post("http://localhost:1234/setting_camera", json={"ip": ip})
    st.image("http://localhost:1234/video_feed")

# ==============================
# AUDIO
# ==============================

elif page == "Audio Sensors":
    st.markdown("## 🔊 Audio Monitoring")
    ip = st.text_input("Sensor IP")
    if st.button("Connect"):
        requests.post("http://localhost:1235/setting_audio", json={"ip": ip})
    st.audio("http://localhost:1235/audio_feed")
