from keras.applications.efficientnet import preprocess_input
import keras
from ultralytics import YOLO
import os
#dependency error to be resolved
# Define the custom objects mapping
custom_dict = {
    'preprocess_input': preprocess_input
}

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta
import tensorflow as tf
import torch
import io
from PIL import Image
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
def load_vision_model():
    base_path = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_path, 'models', 'best.pt')
    model = YOLO(model_path)
    return model

@st.cache_resource
def load_audio_model():
    audio_model1 = tf.keras.models.load_model('./models/audio_forest_69.keras', custom_objects=custom_dict)
    audio_model2 = tf.keras.models.load_model('./models/audio_forest_69420.keras', custom_objects=custom_dict)
    return (audio_model1, audio_model2)



vision_model = load_vision_model()
audio_model1, audio_model2 = load_audio_model()

# --- HELPER FUNCTIONS FOR INFERENCE ---

def run_vision_inference(file_buffer=None, is_video=False, FRAME_ARRAY=None):
    # --- 1. LIVE FRAME LOGIC (Direct from OpenCV) ---
    if FRAME_ARRAY is not None:
        results = vision_model.predict(source=FRAME_ARRAY, imgsz=640, conf=0.35, verbose=False)
        annotated_img = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        if len(results[0].boxes) > 0:
            conf = float(results[0].boxes[0].conf[0])
            label = results[0].names[int(results[0].boxes[0].cls[0])]
            return conf, label, annotated_rgb
        return 0.0, "No detection", annotated_rgb

    # --- 2. IMAGE UPLOAD LOGIC ---
    elif not is_video and file_buffer is not None:
        img = Image.open(file_buffer)
        img_array = np.array(img)
        results = vision_model.predict(source=img_array, imgsz=800, conf=0.35)
        annotated_img = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        if len(results[0].boxes) > 0:
            conf = float(results[0].boxes[0].conf[0])
            label = results[0].names[int(results[0].boxes[0].cls[0])]
            return conf, label, annotated_rgb
        return 0.0, "No detection", annotated_rgb

    # --- 3. VIDEO FILE LOGIC (Corrected Structure) ---
    elif is_video and file_buffer is not None:
        import tempfile
        import os 
        import gc

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(file_buffer.read())
            temp_path = tfile.name
        
        try:
            st_frame = st.empty() 
            highest_conf = 0.0
            top_label = "Scanning..."

            # stream=True prevents memory overflow on large files
            results = vision_model.predict(source=temp_path, stream=True, conf=0.25)

            for result in results: 
                frame = result.plot()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

                if len(result.boxes) > 0:
                    current_conf = float(result.boxes[0].conf[0])
                    if current_conf > highest_conf:
                        highest_conf = current_conf
                        top_label = result.names[int(result.boxes[0].cls[0])]
            
            # Releasing memory locks
            del results
            gc.collect()
            return highest_conf, top_label, None

        finally:
            # THIS IS THE CLEANUP YOU NEEDED
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
            get_radius=10000,
        ))

    # LAYER 2: AI Detections from DB (Purple/Orange)
    if not db_alerts.empty:
        layers.append(pdk.Layer(
            'ScatterplotLayer',
            data=db_alerts[db_alerts['source'] != 'Satellite'],
            get_position='[longitude, latitude]',
            get_color=[180, 0, 255, 160], # Purple for AI
            get_radius=12000,
        ))

    # LAYER 3: THE BLUE DOTS (Active Missions)
    # This will now cover BOTH AI and Satellite hotspots if they are in 'deployments'
    if not deployments.empty:
        layers.append(pdk.Layer(
            'ScatterplotLayer',
            data=deployments,
            get_position='[longitude, latitude]',
            get_color=[0, 150, 255, 255], # Bright Solid Blue
            get_radius=15000,             # Larger to 'contain' the red dot
            pickable=True
        ))

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=28.1, longitude=84.2, zoom=6.5, pitch=40),
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
    st.markdown("### 🛰️ Data Source")
    st.info("NASA FIRMS (VIIRS)\nNear Real-Time Detection")
    
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
    st.markdown('<div class="sub-header">Real-time monitoring powered by NASA satellite data</div>', unsafe_allow_html=True)
    
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
    tab1, tab2, tab3, tab4,tab5 = st.tabs(["🗺️ Live Map", "⚠️ Active Alerts", "📊 Timeline Analytics", "🔬 Model Testing","🛰️ Live Monitoring"])
    
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
                            if st.button(f"🚀 Deploy", key=f"ai_dep_{row['id']}"):
                                dispatch_ranger(row['id'], row['latitude'], row['longitude'])
                                st.rerun()
                    with c2:
                        if st.button(f"✅ Resolve", key=f"ai_res_{row['id']}"):
                            update_alert_status(row['id'], 'Resolved')
                            st.rerun()

        st.markdown("---")

        # 4. Display Satellite Alerts (Your existing UI logic, but with functional buttons)
        if not sat_alerts.empty:
            st.markdown("#### 🛰️ Satellite Hotspots")
            # Sort by brightness
            alerts = sat_alerts.sort_values('brightness', ascending=False).head(10)
            
            for idx, row in alerts.iterrows():
                severity, color = get_severity(row['brightness'])
                
                with st.container():
                    st.markdown(f"""
                    <div style='background: #f0f7ed; padding: 1.2rem; border-radius: 10px; margin: 0.8rem 0; 
                                border-left: 5px solid {color}; box-shadow: 0 2px 4px rgba(45,80,22,0.15);'>
                        <div style='display: flex; justify-content: space-between; align-items: start;'>
                            <div style='flex: 1;'>
                                <h4 style='margin: 0 0 0.5rem 0; color: #2d5016; font-size: 1.1rem;'>📍 {row['region']}</h4>
                                <p style='margin: 0; color: #5a7a52; font-size: 0.9rem;'>🕐 Detected: {row['acq_date']} at {row['acq_time']} UTC</p>
                                <div style='margin-top: 1rem; display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                                    <div><div style='color: #6b8e5f; font-size: 0.8rem;'>Brightness</div><div style='font-weight: 600;'>{row['brightness']:.1f}K</div></div>
                                    <div><div style='color: #6b8e5f; font-size: 0.8rem;'>Confidence</div><div style='font-weight: 600;'>{row['confidence']}%</div></div>
                                    <div><div style='color: #6b8e5f; font-size: 0.8rem;'>Lat/Lon</div><div style='font-weight: 600;'>{row['latitude']:.2f}, {row['longitude']:.2f}</div></div>
                                </div>
                            </div>
                            <span style='background: {color}; color: white; padding: 0.4rem 0.8rem; border-radius: 16px; font-size: 0.85rem;'>{severity} RISK</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
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
        st.caption("Upload specific media types to test our specialized AI models and log detections to the live map.")

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
                        
                        if score > 0.4: # Only log if it's a confident detection
                            # 📍 Log to Database
                            log_detection(
                                source="Vision AI", 
                                inc_type=label, 
                                conf=score, 
                                lat=27.52, # In a real app, these come from EXIF data or camera GPS
                                lon=84.45, 
                                region="Chitwan Sector A"
                            )
                            st.success(f"🚨 {label} logged to map!")
                            st.balloons()
                        else:
                            st.warning("No threats detected above threshold.")

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
            ip_addr = st.text_input("IP Address", placeholder="192.168.1.10")
        with col_port:
            port_num = st.text_input("Port", value="8080")
        with col_btn:
            st.write("##") # Align button with inputs
            if st.button("Connect", use_container_width=True):
                full_ip = f"{ip_addr}:{port_num}"
                try:
                    # Tell Flask to switch to this new camera IP
                    resp = requests.post("http://localhost:1234/setting_camera", json={"ip": full_ip})
                    if resp.status_code == 200:
                        st.success("Camera Link Established!")
                    else:
                        st.error("Flask rejected the IP format.")
                except:
                    st.error("Error: Flask server is not running on port 5000.")

        # Video Feed Display
        st.divider()
        # Flask serves the MJPEG stream at this URL
        flask_url = "http://localhost:1234/video_feed"
        
        # st.image handles the MJPEG stream automatically
        st.image(flask_url, caption="Real-time Detection Feed", use_container_width=True)
    
        st.markdown("---")
        st.info("💡 **Note:** The `type` parameter in the uploader strictly prevents users from uploading incorrect formats (e.g., an MP3 will not be accepted in the Image section).")
    # comment
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #5a7a52; font-size: 0.85rem; padding: 1rem 0;'>
        <p><strong>ForestGuard</strong> • Powered by NASA FIRMS and El Quarters (VIIRS/MODIS) • Data refreshed every 3 hours, hola ig, herya xaina aaile samma</p>
        <p>🌍 Protecting forests through intelligent monitoring</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ Error loading data: {str(e)}")
    st.info("Please refresh the page or check your connection.")
    st.exception(e)