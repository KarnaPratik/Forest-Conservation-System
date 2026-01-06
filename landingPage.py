from keras.applications.efficientnet import preprocess_input
import keras
from ultralytics import YOLO
import os
#dependency error to be resolved
#Fixed local changes.
# Define the custom objects mapping
custom_dict = {
    'preprocess_input': preprocess_input
}
import time
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta
import tensorflow as tf
import torch
import io
from PIL import Image
import tempfile  # Added this
import gc        # Added this
#import librosa
import cv2
import sqlite3
import requests
from audio_to_img import for_single_audio

#Database Helper Functions
def get_db_connection():
    # check_same_thread=False allows Streamlit's threads to access the DB safely
    return sqlite3.connect('vanarakshya.db', check_same_thread=False)

def get_table_df(table_name):
    """Fetches any table from the DB and returns it as a Pandas DataFrame"""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception as e:
        # Returns an empty dataframe if the table doesn't exist yet
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def update_alert_status(alert_id, new_status):
    """Updates the status of an alert (New -> Dispatched -> Resolved)"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE alerts SET status = ? WHERE id = ?", (new_status, alert_id))
    conn.commit()
    conn.close()

def log_detection(source, inc_type, conf, lat, lon, region):
    """Saves a new AI or Satellite detection into the database"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO alerts (source, incident_type, confidence, latitude, longitude, region)
                 VALUES (?, ?, ?, ?, ?, ?)''', (source, inc_type, conf, lat, lon, region))
    conn.commit()
    conn.close()

def dispatch_ranger(alert_id, lat, lon, name="Ranger Arjun"):
    """Marks alert as Dispatched and creates a record in the deployments table"""
    conn = get_db_connection()
    c = conn.cursor()
    # 1. Update Alert Status
    c.execute("UPDATE alerts SET status = 'Dispatched' WHERE id = ?", (alert_id,))
    # 2. Create Deployment record for the map icon
    c.execute("INSERT INTO deployments (alert_id, latitude, longitude, ranger_name) VALUES (?, ?, ?, ?)",
              (alert_id, lat, lon, name))
    conn.commit()
    conn.close()

def reset_database():
    """Wipes all alerts and deployments from the database"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM alerts")
    c.execute("DELETE FROM deployments")
    conn.commit()
    conn.close()

# --- MODEL LOADING ---
@st.cache_resource
def load_vision_models():
    base_path = os.path.dirname(os.path.abspath(__file__))

    model_path_1 = os.path.join(base_path, 'models', 'best_v1.pt')
    model_path_2 = os.path.join(base_path, 'models', 'best_v2.pt')
    model1 = YOLO(model_path_1)
    model2 = YOLO(model_path_2)
    return model1, model2

@st.cache_resource
def load_audio_model():
    audio_model1 = tf.keras.models.load_model('./models/audio_forest_69.keras', custom_objects=custom_dict)
    audio_model2 = tf.keras.models.load_model('./models/audio_forest_69420.keras', custom_objects=custom_dict)
    return (audio_model1, audio_model2)



vision_model1, vision_model2 = load_vision_models()
audio_model1, audio_model2 = load_audio_model()

# --- HELPER FUNCTIONS FOR INFERENCE ---
def run_vision_inference(file_buffer=None, is_video=False, FRAME_ARRAY=None):
    # Helper to check agreement between two result objects
    def get_consensus(res1, res2):
        # res1 is usually a list from .predict(), res2 is a result object
        det1 = len(res1[0].boxes) > 0
        det2 = len(res2.boxes) > 0
        
        # Get max confidence if available
        conf1 = float(res1[0].boxes[0].conf[0]) if det1 else 0.0
        conf2 = float(res2.boxes[0].conf[0]) if det2 else 0.0
        max_conf = max(conf1, conf2)

        # CASE 1: BOTH MODELS AGREE
        if det1 and det2:
            return max_conf, "fire"
        
        # CASE 2: ONLY ONE MODEL DETECTS
        elif det1 or det2:
            if max_conf > 0.50:
                return max_conf, "fire"
            return max_conf, "Unconfirmed fire"
        
        return 0.0, "No detection"

    # --- 1. LIVE FRAME LOGIC ---
    if FRAME_ARRAY is not None:
        results1 = vision_model1.predict(source=FRAME_ARRAY, imgsz=640, conf=0.35, verbose=False)
        results2 = vision_model2.predict(source=FRAME_ARRAY, imgsz=640, conf=0.35, verbose=False)
        
        conf, label = get_consensus(results1, results2[0])
        annotated_img = results1[0].plot() 
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return conf, label, annotated_rgb

    # --- 2. IMAGE UPLOAD LOGIC ---
    elif not is_video and file_buffer is not None:
        img = Image.open(file_buffer)
        img_array = np.array(img) # Now correctly uses global np
        
        results1 = vision_model1.predict(source=img_array, imgsz=800, conf=0.25)
        results2 = vision_model2.predict(source=img_array, imgsz=800, conf=0.25)
        
        # results2[0] ensures we pass the result object, not the list
        conf, label = get_consensus(results1, results2[0])
        annotated_img = results1[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return conf, label, annotated_rgb

    # --- 3. VIDEO FILE LOGIC ---
    elif is_video and file_buffer is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(file_buffer.read())
            temp_path = tfile.name
        
        try:
            st_frame = st.empty() 
            highest_conf = 0.0
            top_label = "Scanning..."
            frame_count = 0

            stream1 = vision_model1.predict(source=temp_path, stream=True, conf=0.25)
            
            for res1 in stream1:
                frame_count += 1
                frame = res1.plot()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Image Enhancement
                lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                enhanced_frame = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)
                st_frame.image(enhanced_frame, channels="RGB", use_container_width=True)

                if len(res1.boxes) > 0:
                    res2_list = vision_model2.predict(source=enhanced_frame, conf=0.25, verbose=False)
                    if res2_list and len(res2_list[0].boxes) > 0: 
                        conf, label = get_consensus([res1], res2_list[0])
                        if conf > highest_conf:
                            highest_conf = conf
                            top_label = label
                
                elif frame_count % 10 == 0:
                    h, w, _ = enhanced_frame.shape
                    y1, y2 = int(h * 0.25), int(h * 0.75)
                    x1, x2 = int(w * 0.25), int(w * 0.75)
                    zoom_frame = enhanced_frame[y1:y2, x1:x2]

                    res2_list = vision_model2.predict(source=zoom_frame, conf=0.25, verbose=False)
                    if res2_list and len(res2_list[0].boxes) > 0:
                        current_res2_conf = res2_list[0].boxes.conf.max().item()
                        if current_res2_conf > highest_conf:
                            highest_conf = current_res2_conf
                            top_label = "Fire (Low-light environment)"
            
            gc.collect()
            return highest_conf, top_label, None

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Cleanup error: {e}")
    
    return 0.0, "Invalid Input", None

def run_audio_inference(file_buffer):
    class_names = ['natural sound', 'unnatural']
    class_names2 = ['fire', 'logging', 'poaching']

    # 1. Get image in 0-255 range
    img_ready, y_raw, sr = for_single_audio(file_buffer)

# EfficientNet specific preprocessing (Handles mean/std subtraction)
    img_for_model = preprocess_input(img_ready)

    # 3. Predict
    y_pred1 = audio_model1.predict(img_for_model)

    confidence = float(y_pred1[0][0])
    class_index = int(confidence > 0.5)
    label = class_names[class_index]

    confidence = confidence if class_index == 1 else (1 - confidence)

    if confidence > 0.5:
        # 1. Get the full probability distribution
        y_pred2 = audio_model2.predict(img_for_model)
        
        # 2. Find the index of the highest probability
        # y_pred2[0] is used assuming batch size of 1
        class_index2 = np.argmax(y_pred2[0])
        
        # 3. Extract the actual probability (confidence) for that class
        confidence2 = float(y_pred2[0][class_index2])
        
        # 4. Map to label and update the confidence variable
        label = class_names2[class_index2]
        confidence = confidence2
    

    return confidence, label

#Separate CSS file definition
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page config - MUST BE FIRST
st.set_page_config(
    page_title="VanaRakshya",
    page_icon="🌳🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Load
local_css("styles.css")

#Custom fragment code for only the map to rerun everytime along with the database.
@st.fragment(run_every=10)
def live_map_fragment(satellite_df):
    # Fetch real data from DB
    db_alerts = get_table_df("alerts")
    deployments = get_table_df("deployments")
    
    layers = []

    # LAYER 1: The Mock Satellite Fires (Red)
    if not satellite_df.empty:
        layers.append(pdk.Layer(
            'ScatterplotLayer',
            data=satellite_df,
            get_position='[longitude, latitude]',
            get_color=[220, 38, 38, 120], # Red with transparency
            get_radius=5000,
        ))

    # LAYER 2: AI Detections from DB (Purple/Orange)
    if not db_alerts.empty:
        layers.append(pdk.Layer(
            'ScatterplotLayer',
            data=db_alerts[db_alerts['source'] != 'Satellite'],
            get_position='[longitude, latitude]',
            get_color=[180, 0, 255, 160], # Purple for AI
            get_radius=5000,
        ))

    # LAYER 3: THE BLUE DOTS (Active Missions)
    # This will now cover BOTH AI and Satellite hotspots if they are in 'deployments'
    if not deployments.empty:
        layers.append(pdk.Layer(
            'ScatterplotLayer',
            data=deployments,
            get_position='[longitude, latitude]',
            get_color=[0, 150, 255, 255], # Bright Solid Blue
            get_radius=5000,             # Larger to 'contain' the red dot
            pickable=True
        ))

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=28.1, longitude=84.2, zoom=6.5, pitch=0),
        map_style='dark'
    ))

# Safety check to make sure db is ready before app launch
if 'db_initialized' not in st.session_state:
    import sqlite3
    conn = sqlite3.connect('vanarakshya.db')
    c = conn.cursor()
    # Create tables if they are missing
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, incident_type TEXT, confidence REAL, latitude REAL, longitude REAL, region TEXT, status TEXT DEFAULT 'New', timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS deployments (id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER, latitude REAL, longitude REAL, ranger_name TEXT, status TEXT DEFAULT 'In Transit')''')
    conn.commit()
    conn.close()
    st.session_state['db_initialized'] = True

# Generate mock data
def generate_mock_fire_data(days_back=7):
    """Generate realistic mock fire data"""
    base_date = datetime.now()
    data = []
    
    # Real fire-prone locations with realistic coordinates
    locations = [
        {"lat": 27.5250, "lon": 84.4500, "region": "Chitwan National Park (Core)", "base_brightness": 355},
        {"lat": 27.5800, "lon": 84.3200, "region": "Chitwan Buffer Zone", "base_brightness": 342},
        {"lat": 28.6000, "lon": 81.3000, "region": "Bardiya National Park", "base_brightness": 348},
        {"lat": 28.9500, "lon": 80.2000, "region": "Shuklaphanta Forest", "base_brightness": 340},
        {"lat": 27.4500, "lon": 84.9000, "region": "Parsa National Park", "base_brightness": 352},
        {"lat": 28.3000, "lon": 82.2000, "region": "Banke Forest Sector", "base_brightness": 345},
        {"lat": 26.6500, "lon": 87.2500, "region": "Koshi Tappu Region", "base_brightness": 339},
    ]
    
    for day in range(days_back):
        date = base_date - timedelta(days=day)
        date_str = date.strftime("%Y-%m-%d")
        
        for loc in locations:
            # Simulate some variability - not all fires every day
            if np.random.random() > 0.3:  # 70% chance of detection
                # Add slight coordinate variation
                lat_offset = np.random.uniform(-0.1, 0.1)
                lon_offset = np.random.uniform(-0.1, 0.1)
                
                # Brightness variation
                brightness = loc["base_brightness"] + np.random.uniform(-10, 15)
                
                data.append({
                    "latitude": loc["lat"] + lat_offset,
                    "longitude": loc["lon"] + lon_offset,
                    "brightness": round(brightness, 1),
                    "confidence": np.random.randint(75, 99),
                    "acq_date": date_str,
                    "acq_time": f"{np.random.randint(0, 23):02d}:{np.random.randint(0, 59):02d}",
                    "region": loc["region"]
                })
    
    return pd.DataFrame(data)

def get_severity(brightness):
    """Determine fire severity based on brightness temperature"""
    if brightness >= 360:
        return "HIGH", "#dc2626"
    elif brightness >= 340:
        return "MEDIUM", "#f59e0b"
    else:
        return "LOW", "#528b3a"

# Sidebar
with st.sidebar:
    st.markdown("# 🌲 VanaRakshya")
    st.markdown("---")
    
    st.markdown("### 📅 Time Range")
    days_back = st.slider("Days of historical data", 1, 14, 7)
    
    st.markdown("### 🔍 Filters")
    min_confidence = st.slider("Minimum Confidence %", 0, 100, 70)
    
    st.markdown("### 🎯 Focus Regions")
    all_regions = ["All Regions", "California", "Amazon", "Australia"]
    selected_regions = st.multiselect(
        "Filter by region",
        all_regions,
        default=["All Regions"]
    )
    
    st.markdown("---")
    with st.sidebar.expander("🛠️ Admin Tools"):
        st.warning("This will clear all live data.")
        if st.button("Clear All Alerts & Missions", use_container_width=True):
            reset_database()
            st.success("Database cleared!")
            st.rerun()
    st.markdown("### 📊 Data Stats")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

# Generate data
try:
    df = generate_mock_fire_data(days_back)
    
    # Apply filters
    df_filtered = df[df['confidence'] >= min_confidence].copy()
    
    # Region filter
    if "All Regions" not in selected_regions and len(selected_regions) > 0:
        df_filtered = df_filtered[df_filtered['region'].str.contains('|'.join(selected_regions), case=False, na=False)]
    
    # Main header
    st.markdown('<div class="main-header">🤖VanaRakshya</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(df_filtered)}</div>
            <div class="stat-label">Active Hotspots</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_severity = len(df_filtered[df_filtered['brightness'] >= 360])
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{high_severity}</div>
            <div class="stat-label">High Severity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_conf = int(df_filtered['confidence'].mean()) if len(df_filtered) > 0 else 0
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{avg_conf}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_regions = df_filtered['region'].nunique()
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{unique_regions}</div>
            <div class="stat-label">Affected Regions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    
    # Tabs
    tab1, tab2, tab3, tab4,tab5,tab6 = st.tabs(["🗺️ Live Map", "⚠️ Active Alerts", "📊 Timeline Analytics", "🔬 Model Testing","🛰️ Live Monitoring","🔉 Streaming Audio Sensors"])
    
    with tab1:
        st.markdown("### 🌍 Global Hotspot Detection Map")
        st.caption("Interactive map showing detected fire hotspots and active deployments.")
        
        # CALL THE CORRECT NAME HERE
        if len(df_filtered) > 0:
            live_map_fragment(df_filtered)
        else:
            st.warning("⚠️ No hotspots detected matching current filters.")

        # Legend (Static, stays below the map)
        st.markdown("---")
        l1, l2, l3, l4 = st.columns(4)
        l1.markdown("🔴 **High Risk**")
        l2.markdown("🟡 **Medium Risk**")
        l3.markdown("🟢 **Low Risk**")
        l4.markdown("🔵 **Ranger Deployed**")
        
    with tab2:
        st.markdown("### 🚨 Critical Incident Control")
        st.caption("Manage AI detections and high-severity satellite hotspots.")

        # 1. Fetch AI Alerts from SQLite
        conn = get_db_connection()
        # We fetch everything that isn't 'Resolved'
        db_alerts = pd.read_sql_query("SELECT * FROM alerts WHERE status != 'Resolved' ORDER BY timestamp DESC", conn)
        conn.close()

        # 2. Combine with Satellite Alerts (High Severity Only)
        # We filter your 'df_filtered' for brightness >= 340 to keep Tab 2 focused on action
        sat_alerts = df_filtered[df_filtered['brightness'] >= 340].copy()
        
        # 3. Display AI Alerts First (Priority)
        if not db_alerts.empty:
            st.markdown("#### 🤖 AI Detections (Vision/Audio)")
            for idx, row in db_alerts.iterrows():
                # Color coding based on status
                status_color = "#3b82f6" if row['status'] == 'Dispatched' else "#ef4444"
                
                with st.container():
                    st.markdown(f"""
                    <div style='background: #f8fafc; padding: 1.2rem; border-radius: 10px; margin: 0.8rem 0; 
                                border-left: 5px solid {status_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                        <div style='display: flex; justify-content: space-between;'>
                            <div>
                                <h4 style='margin:0; color: #1e293b;'>🔥 {row['incident_type']} in {row['region']}</h4>
                                <p style='margin:0; color: #64748b; font-size: 0.85rem;'>Detected via {row['source']} • {row['timestamp']}</p>
                            </div>
                            <span style='background: {status_color}; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem;'>
                                {row['status'].upper()}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        if row['status'] == 'New':
                            if st.button("🚁 Deploy", key=f"sat_dep_{idx}"):
                            # 1. Create a 'Real' alert in the DB from this Mock Hotspot
                                conn = get_db_connection()
                                c = conn.cursor()
                                c.execute('''INSERT INTO alerts (source, incident_type, confidence, latitude, longitude, region, status)
                                            VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                                        ("Satellite", "Wildfire", row['confidence'], row['latitude'], row['longitude'], row['region'], "Dispatched"))
                                
                                alert_id = c.lastrowid # Get the ID of the fire we just saved
                                
                                # 2. Create the Deployment entry
                                c.execute("INSERT INTO deployments (alert_id, latitude, longitude, ranger_name) VALUES (?, ?, ?, ?)",
                                        (alert_id, row['latitude'], row['longitude'], "Nepal Ranger Unit 1"))
                                
                                conn.commit()
                                conn.close()
                                
                                st.success(f"Units Dispatched to {row['region']}!")
                                st.rerun()
        
                    with c2:
                        if st.button(f"✅ Resolve", key=f"ai_res_{row['id']}"):
                            update_alert_status(row['id'], 'Resolved')
                            st.rerun()

        st.markdown("---")
                       
        
    with tab3:
        st.markdown("### 📈 Temporal Analysis")
        
        if len(df_filtered) > 0:
            # Timeline
            st.markdown("#### Hotspot Activity Over Time")
            timeline = df_filtered.groupby('acq_date').size().reset_index(name='count')
            timeline = timeline.sort_values('acq_date')
            
            st.line_chart(timeline.set_index('acq_date')['count'], use_container_width=True, color="#528b3a")
            
            # Regional distribution
            st.markdown("#### Regional Distribution")
            region_counts = df_filtered.groupby('region').size().reset_index(name='count')
            region_counts = region_counts.sort_values('count', ascending=False)
            
            st.bar_chart(region_counts.set_index('region')['count'], use_container_width=True, color="#6a9955")
            
            # Daily breakdown
            st.markdown("#### 📅 Daily Breakdown")
            dates = sorted(df['acq_date'].unique(), reverse=True)
            selected_date = st.select_slider("Select date", options=dates, value=dates[0])
            
            daily = df_filtered[df_filtered['acq_date'] == selected_date]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Hotspots", len(daily))
            with col2:
                if len(daily) > 0:
                    st.metric("Avg Brightness", f"{daily['brightness'].mean():.1f}K")
            with col3:
                if len(daily) > 0:
                    st.metric("Max Confidence", f"{daily['confidence'].max()}%")
            
            if len(daily) > 0:
                st.dataframe(
                    daily[['region', 'acq_time', 'brightness', 'confidence', 'latitude', 'longitude']].sort_values('brightness', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No detections on this date.")
        else:
            st.warning("⚠️ No data available for analysis.")

    with tab4:
        st.markdown("### 🔬 Custom Data Prediction")

        col_img, col_vid, col_aud = st.columns(3)

        with col_img:
            st.subheader("🖼️ Image Input")
            img_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="img_test")
            if img_file:
                st.image(img_file, use_container_width=True)
                if st.button("Run Vision Model (Img)", use_container_width=True):
                    with st.spinner("Analyzing pixels..."):
                        score, label, result_img = run_vision_inference(img_file, is_video=False)
                        st.image(result_img, caption=f"Detection Result: {label}", use_container_width=True)
                        
                        if score > 0.40: # Only log if it's a confident detection
                            # 📍 Log to Database
                            log_detection(
                                source="Vision AI", 
                                inc_type=label, 
                                conf=score, 
                                lat=27.98, # In a real app, these come from EXIF data or camera GPS
                                lon=86.92, 
                                region="Test"
                            )
                            if label == "fire":
                                #triggered because both agreed OR one was moderately confident
                                st.success(f"🚨 Fire confirmed and logged to map! (Confidence: {score:.2f})")
                            elif label == "Unconfirmed fire":
                                #triggered because only one saw it with moderate confidence
                                st.warning(f"⚠️ Unconfirmed fire detected by single model. (Confidence: {score:.2f})")
                        else:
                            st.info("No threats detected above threshold.")

        with col_vid:
            st.subheader("📽️ Video Input")
            vid_file = st.file_uploader("Upload Video", type=['mp4', 'mov'], key="vid_test")
            if vid_file:
                st.video(vid_file)
                if st.button("Run Vision Model (Vid)", use_container_width=True):
                    with st.spinner("Scanning frames..."):
                        #triggering the 'is_video' block in run_vision_inference above
                        score, label, _ = run_vision_inference(vid_file, is_video=True)
                        #showing final best result after loop is done
                        st.metric("Top Confidence", f"{score:.1%}")
                        if score > 0.5:
                            log_detection("Vision AI (Vid)", label, score, 27.58, 84.30, "Western Buffer Zone")
                            st.success(f"Detected {label}! Video event logged to database.!")

        with col_aud:
            st.subheader("🔊 Audio Input")
            aud_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'], key="aud_test")
            if aud_file:
                st.audio(aud_file)
                if st.button("Run Audio Model", use_container_width=True):
                    with st.spinner("Analyzing frequencies..."):
                        score, label = run_audio_inference(aud_file)
                        st.metric("Confidence", f"{score:.1%}")
                        
                        if score > 0.5 and label != 'natural sound':
                            # 📍 Log to Database
                            log_detection("Audio AI", label, score, 27.60, 84.55, "Narayani Riverbank")
                            st.error(f"⚠️ {label.upper()} DETECTED! Alerting Rangers.")
                        else:
                            st.write(f"**Result:** {label}")
    with tab5:
        st.header("Live AI Surveillance")
        
        # Connection Interface
        st.info("Enter the IP address of your camera to start the AI stream.")
        
        col_ip, col_port, col_btn = st.columns([3, 1, 1])
        
        with col_ip:
            ip_addr = st.text_input("IP Address", placeholder="192.168.1.10",key="cam_ip")
        with col_port:
            port_num = st.text_input("Port", value="8080",key="cam_port")
        with col_btn:
            st.write("##") # Align button with inputs
            if st.button("Connect", use_container_width=True,key="cam_btn"):
                full_ip = f"{ip_addr}:{port_num}"
                try:
                    # Tell Flask to switch to this new camera IP
                    resp = requests.post("http://localhost:1234/setting_camera", json={"ip": full_ip})
                    if resp.status_code == 200:
                        st.success("Camera Link Established!")
                    else:
                        st.error("Flask rejected the IP format.")
                except:
                    st.error("Error: Flask server is not running on port 1234.")

        # Video Feed Display
        st.divider()
        
        # Flask serves the MJPEG stream at this URL
        flask_url = "http://localhost:1234/video_feed"
        
        # SHOW VIDEO FIRST
        st.image(flask_url, caption="Real-time Detection Feed", use_container_width=True)
        
        # THEN SHOW THE THREAT STATUS BELOW THE VIDEO
        st.divider()
        st.subheader("🎯 Threat Status")

        # Initialize session state for caching
        if "last_combined_label" not in st.session_state:
            st.session_state.last_combined_label = None
        
        status_placeholder = st.empty()
        
        @st.fragment(run_every=2)  # Update every 2 seconds
        def combined_status_fragment():
            try:
                # Fetch combined prediction from Flask
                res = requests.get("http://localhost:1234/combined_pred", timeout=1).json()
                combined_label = res.get("Combined_label", "Analysing...")
                
                # Initialize cache if first run
                if st.session_state.last_combined_label is None:
                    st.session_state.last_combined_label = combined_label
                
                # Only update if values changed
                if combined_label == st.session_state.last_combined_label:
                    return
                
                # Update cache
                st.session_state.last_combined_label = combined_label
                
                # Convert label to short, actionable message
                if combined_label == "Natural":
                    color = "#22c55e"  # Green
                    icon = "✅"
                    message = "All Clear"
                elif "fire" in combined_label.lower():
                    color = "#ef4444"  # Red
                    icon = "🔥"
                    message = "FIRE DETECTED"
                elif "logging" in combined_label.lower():
                    color = "#ef4444"  # Red
                    icon = "🪓"
                    message = "ILLEGAL LOGGING"
                elif "poaching" in combined_label.lower():
                    color = "#ef4444"  # Red
                    icon = "🎯"
                    message = "POACHING ACTIVITY"
                elif "Sounds and seems Suspicious" in combined_label:
                    color = "#dc2626"  # Dark Red
                    icon = "🚨"
                    # Extract the specific threat from the label
                    if "fire" in combined_label.lower():
                        message = "FIRE ALERT"
                    elif "logging" in combined_label.lower():
                        message = "LOGGING DETECTED"
                    elif "poaching" in combined_label.lower():
                        message = "POACHING DETECTED"
                    else:
                        message = "THREAT DETECTED"
                elif "Sounds Suspicious but cannot see" in combined_label:
                    color = "#f59e0b"  # Orange
                    icon = "👂"
                    # Extract audio threat
                    if "fire" in combined_label.lower():
                        message = "Fire Sounds Detected"
                    elif "logging" in combined_label.lower():
                        message = "Logging Sounds Detected"
                    elif "poaching" in combined_label.lower():
                        message = "Poaching Sounds Detected"
                    else:
                        message = "Suspicious Sounds"
                elif "Seems suspicious but doesn't sound" in combined_label:
                    color = "#f59e0b"  # Orange
                    icon = "👁️"
                    message = "Visual Anomaly Detected"
                elif "Analysing" in combined_label:
                    color = "#6b7280"  # Gray
                    icon = "🔍"
                    message = "Analyzing Stream..."
                else:
                    color = "#f59e0b"  # Orange
                    icon = "⚠️"
                    message = "Unknown Threat"
                
                # Display status card - SIMPLE AND BOLD
                status_placeholder.markdown(f"""
                    <div style="
                        padding:30px;
                        border-radius:16px;
                        background-color:{color};
                        color:white;
                        text-align:center;
                        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                        transition: all 0.3s ease;
                    ">
                        <div style="font-size:4rem; margin-bottom:15px;">{icon}</div>
                        <h2 style="margin:0; font-size:2rem; font-weight:700; letter-spacing:1px;">
                            {message}
                        </h2>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                status_placeholder.warning(f"⏳ Waiting for threat assessment...")
        
        # Render the fragment
        combined_status_fragment()

        st.markdown("---")
        st.info("💡 **Note:** Connect your camera to start real-time forest monitoring with AI-powered threat detection.")

    
        st.markdown("---")
        st.info("💡 **Note:** The `type` parameter in the uploader strictly prevents users from uploading incorrect formats (e.g., an MP3 will not be accepted in the Image section).")
    with tab6:
        st.header("Monitor Audio Sensors")
        # init state
        if "audio_connected" not in st.session_state:
            st.session_state.audio_connected = False

        # Connection Interface
        st.info("Enter the IP address of your audio sensor to start the AI stream.")
        col_ip, col_port, col_btn = st.columns([3, 1, 1])

        with col_ip:
            ip_addr = st.text_input("IP Address", placeholder="192.168.1.10", key="audio_ip")
        with col_port:
            port_num = st.text_input("Port", value="8080", key="audio_port")
        with col_btn:
            st.write("##")  # align
            if st.button("Connect", use_container_width=True, key="audio_btn"):
                full_ip = f"{ip_addr}:{port_num}"
                try:
                    resp = requests.post("http://localhost:5000/setting_audio", json={"ip": full_ip}, timeout=3)
                    if resp.status_code == 200:
                        st.session_state.audio_connected = True
                        st.success("Audio Sensor Link Established!")
                    else:
                        # show server message if available
                        try:
                            msg = resp.json()
                        except:
                            msg = {"error": f"code {resp.status_code}"}
                        st.error(f"Flask rejected the IP format: {msg}")
                except Exception as e:
                    st.error(f"Error: Could not reach Flask on port 1235 ({e})")

        st.divider()

        # audio feed URL (used by browser player)
        flask_url = "http://localhost:5000/audio_feed"
        status_box = st.empty()
        if "last_audio_label" not in st.session_state:
            st.session_state.last_audio_label = None
        if "last_audio_pred" not in st.session_state:
            st.session_state.last_audio_pred = None

        @st.fragment(run_every=2)
        def audio_status_fragment():
            if not st.session_state.get("audio_connected", False):
                status_box.info("🔌 Connect an audio sensor to start monitoring.")
                return

            # 🔊 ALWAYS render audio player
            st.markdown(
                f'<audio controls src="{flask_url}" style="width:100%; margin-bottom:10px;"></audio>',
                unsafe_allow_html=True
            )

            try:
                res = requests.get("http://localhost:5000/status", timeout=1).json()
                label = res.get("Audio_Label", "Analysing...")
                pred = float(res.get("Audio_Prediction", 0.0))

                # Initialize state for logging cooldown
                if "last_log_time" not in st.session_state:
                    st.session_state.last_log_time = datetime.now() - timedelta(minutes=5)

                # --- DATABASE LOGGING LOGIC ---
                # Check if it's a threat and confidence is high (> 50%)
                is_threat = str(label).lower() in ["fire", "logging", "poaching"]
                
                if is_threat and pred > 0.5:
                    # Cooldown: Only log to DB if 1 minute has passed since the last log of this type
                    time_since_last_log = datetime.now() - st.session_state.last_log_time
                    
                    if time_since_last_log > timedelta(minutes=1):
                        log_detection(
                            source="Acoustic Sensor",
                            inc_type=label.title(),
                            conf=pred,
                            lat=27.9881, # Sagarmatha Region or dynamic sensor lat
                            lon=86.9250, # Sagarmatha Region or dynamic sensor lon
                            region="Sagarmatha Buffer Zone"
                        )
                        st.session_state.last_log_time = datetime.now()
                        st.toast(f"🚨 ALERT: {label} detected and logged to Command Center!")

                # --- UI UPDATE LOGIC ---
                if (label == st.session_state.get("last_audio_label") and 
                    abs(pred - st.session_state.get("last_audio_pred", 0)) < 0.02):
                    return

                st.session_state.last_audio_label = label
                st.session_state.last_audio_pred = pred

                color = "#ef4444" if is_threat else "#22c55e"

                status_box.markdown(f"""
                    <div style="
                        min-height:110px;
                        padding:14px;
                        border-radius:12px;
                        background-color:{color};
                        color:white;
                        text-align:center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
                        transition: all 0.25s ease;
                    ">
                        <h3 style="margin:0;">{str(label).upper()}</h3>
                        <div style="opacity:0.9; font-weight:bold;">Confidence: {pred:.1%}</div>
                        <div style="font-size:0.75rem; margin-top:5px;">{'LOGGED TO DATABASE' if is_threat else 'ENVIRONMENT SECURE'}</div>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                status_box.warning(f"Waiting for audio sensor status...")

        # render / schedule the fragment
        audio_status_fragment()
 
        
        st.markdown("---")
    
    # comment
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #5a7a52; font-size: 0.85rem; padding: 1rem 0;'>
        <p><strong>VanaRakshya</strong> • Data refreshed every 10 seconds</p>
        <p>🌍 Protecting forests through intelligent monitoring</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ Error loading data: {str(e)}")
    st.info("Please refresh the page or check your connection.")
    st.exception(e)