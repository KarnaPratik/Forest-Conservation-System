from tensorflow.keras.applications.efficientnet import preprocess_input
import keras

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
# import torch
import io
from PIL import Image
#import librosa
#import cv2

from audio_to_img import for_single_audio

# --- MODEL LOADING (Placeholders) ---
@st.cache_resource
def load_vision_model():
    # model = torch.load('vision_model.pth') or tf.keras.models.load_model('model.h5')
    return "Vision Model Loaded"

@st.cache_resource
def load_audio_model():
    audio_model1 = tf.keras.models.load_model('./models/audio_forest_69.keras', custom_objects=custom_dict)
    # audio_model2 = tf.keras.models.load_model('./models/audio_multi_classification.keras', custom_objects=custom_dict)
    return (audio_model1, 'audio_model2')



vision_model = load_vision_model()
audio_model1, audio_model2 = load_audio_model()

# --- HELPER FUNCTIONS FOR INFERENCE ---
def run_vision_inference(file_buffer, is_video=False):
    """Passes image or video to the Vision Model"""
    if not is_video:
        img = Image.open(file_buffer)
        # Process: img_array = np.array(img.resize((224, 224)))
        # prediction = vision_model.predict(img_array)
        return np.random.uniform(0.6, 0.98), "Smoke Detected"
    else:
        # For video, you'd typically sample frames using cv2
        return np.random.uniform(0.5, 0.85), "Active Fire Front"

import numpy as np

def run_audio_inference(file_buffer):
    class_names = ['natural sound', 'unnatural']

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

    return confidence, label

# Page config - MUST BE FIRST
st.set_page_config(
    page_title="ForestGuard - Wildfire Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #ef4444;
        text-align: center;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #dc2626;
        margin: 0;
    }
    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Generate mock data
def generate_mock_fire_data(days_back=7):
    """Generate realistic mock fire data"""
    base_date = datetime.now()
    data = []
    
    # Real fire-prone locations with realistic coordinates
    locations = [
        {"lat": 38.5816, "lon": -121.4944, "region": "Northern California", "base_brightness": 345},
        {"lat": 37.2744, "lon": -119.2696, "region": "Sierra Nevada", "base_brightness": 355},
        {"lat": 34.0522, "lon": -118.2437, "region": "Southern California", "base_brightness": 340},
        {"lat": -33.8688, "lon": 151.2093, "region": "New South Wales, AU", "base_brightness": 360},
        {"lat": -3.4653, "lon": -62.2159, "region": "Amazon Rainforest", "base_brightness": 338},
        {"lat": 40.7128, "lon": -122.4194, "region": "Shasta County, CA", "base_brightness": 350},
        {"lat": 39.3293, "lon": -120.1833, "region": "Tahoe National Forest", "base_brightness": 342},
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
        return "LOW", "#10b981"

# Sidebar
with st.sidebar:
    st.markdown("# 🌲 ForestGuard")
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
    st.markdown('<div class="main-header">🔥 ForestGuard Wildfire Detection System</div>', unsafe_allow_html=True)
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
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Live Map", "⚠️ Active Alerts", "📊 Timeline Analytics", "🔬 Model Testing"])
    
    with tab1:
        st.markdown("### 🌍 Global Hotspot Detection Map")
        st.caption("Interactive map showing detected fire hotspots. Hover over markers for details.")
        
        if len(df_filtered) > 0:
            # Add severity info
            df_filtered['severity'], df_filtered['color'] = zip(*df_filtered['brightness'].apply(get_severity))
            
            # Convert hex to RGB for pydeck
            df_filtered['color_rgb'] = df_filtered['color'].apply(
                lambda x: [int(x[1:3], 16), int(x[3:5], 16), int(x[5:7], 16), 200]
            )
            
            # Calculate center
            center_lat = df_filtered['latitude'].mean()
            center_lon = df_filtered['longitude'].mean()
            
            # Create PyDeck map
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=3,
                pitch=0,
            )
            
            layer = pdk.Layer(
                'ScatterplotLayer',
                data=df_filtered,
                get_position='[longitude, latitude]',
                get_color='color_rgb',
                get_radius='brightness * 100',
                pickable=True,
                auto_highlight=True,
            )
            
            tooltip = {
                "html": "<b>🔥 Fire Detected</b><br/>"
                        "<b>Location:</b> {region}<br/>"
                        "<b>Brightness:</b> {brightness}K<br/>"
                        "<b>Confidence:</b> {confidence}%<br/>"
                        "<b>Severity:</b> {severity}<br/>"
                        "<b>Time:</b> {acq_date} {acq_time}",
                "style": {"backgroundColor": "#1f2937", "color": "white", "padding": "10px"}
            }
            
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style=None,
            )
            
            st.pydeck_chart(deck, use_container_width=True)
            
            # Legend
            st.markdown("#### 🎨 Severity Legend")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🔴 **HIGH** - Brightness ≥ 360K")
            with col2:
                st.markdown("🟡 **MEDIUM** - Brightness 340-360K")
            with col3:
                st.markdown("🟢 **LOW** - Brightness < 340K")
        else:
            st.warning("⚠️ No hotspots detected matching current filters.")
            st.info("Try adjusting the confidence threshold or time range in the sidebar.")
    
    with tab2:
        st.markdown("### 🚨 Critical Fire Alerts")
        st.caption("Most severe fires detected, sorted by brightness temperature")
        
        if len(df_filtered) > 0:
            # Sort by brightness
            alerts = df_filtered.sort_values('brightness', ascending=False).head(15)
            
            for idx, row in alerts.iterrows():
                severity, color = get_severity(row['brightness'])
                
                with st.container():
                    st.markdown(f"""
                    <div style='background: white; padding: 1.2rem; border-radius: 10px; margin: 0.8rem 0; 
                                border-left: 5px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <div style='display: flex; justify-content: space-between; align-items: start;'>
                            <div style='flex: 1;'>
                                <h4 style='margin: 0 0 0.5rem 0; color: #1f2937; font-size: 1.1rem;'>
                                    📍 {row['region']}
                                </h4>
                                <p style='margin: 0; color: #6b7280; font-size: 0.9rem;'>
                                    🕐 Detected: {row['acq_date']} at {row['acq_time']} UTC
                                </p>
                                <div style='margin-top: 1rem; display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                                    <div>
                                        <div style='color: #9ca3af; font-size: 0.8rem; margin-bottom: 0.25rem;'>Brightness</div>
                                        <div style='font-weight: 600; color: #1f2937; font-size: 1.1rem;'>{row['brightness']:.1f}K</div>
                                    </div>
                                    <div>
                                        <div style='color: #9ca3af; font-size: 0.8rem; margin-bottom: 0.25rem;'>Confidence</div>
                                        <div style='font-weight: 600; color: #1f2937; font-size: 1.1rem;'>{row['confidence']}%</div>
                                    </div>
                                    <div>
                                        <div style='color: #9ca3af; font-size: 0.8rem; margin-bottom: 0.25rem;'>Coordinates</div>
                                        <div style='font-weight: 600; color: #1f2937; font-size: 0.9rem;'>{row['latitude']:.3f}°, {row['longitude']:.3f}°</div>
                                    </div>
                                </div>
                            </div>
                            <div style='margin-left: 1rem;'>
                                <span style='background: {color}; color: white; padding: 0.4rem 0.8rem; 
                                             border-radius: 16px; font-size: 0.85rem; font-weight: 600; white-space: nowrap;'>
                                    {severity} RISK
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("📊 Details", key=f"detail_{idx}"):
                            st.info(f"Showing detailed analysis for {row['region']}")
                    with col2:
                        if st.button("🚁 Deploy", key=f"deploy_{idx}"):
                            st.success(f"Alert sent to nearest ranger station!")
        else:
            st.warning("⚠️ No alerts matching current filters.")
    
    with tab3:
        st.markdown("### 📈 Temporal Analysis")
        
        if len(df_filtered) > 0:
            # Timeline
            st.markdown("#### Hotspot Activity Over Time")
            timeline = df_filtered.groupby('acq_date').size().reset_index(name='count')
            timeline = timeline.sort_values('acq_date')
            
            st.line_chart(timeline.set_index('acq_date')['count'], use_container_width=True, color="#dc2626")
            
            # Regional distribution
            st.markdown("#### Regional Distribution")
            region_counts = df_filtered.groupby('region').size().reset_index(name='count')
            region_counts = region_counts.sort_values('count', ascending=False)
            
            st.bar_chart(region_counts.set_index('region')['count'], use_container_width=True, color="#ef4444")
            
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
        st.caption("Upload specific media types to test our specialized AI models.")

        # Create three columns for the three data types
        col_img, col_vid, col_aud = st.columns(3)

        with col_img:
            st.subheader("🖼️ Image Input")
            img_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="img_test")
            if img_file:
                st.image(img_file, use_container_width=True)
                if st.button("Run Vision Model (Img)", use_container_width=True):
                    with st.spinner("Analyzing pixels..."):
                        score, label = run_vision_inference(img_file, is_video=False)
                        st.metric("Confidence", f"{score:.1%}")
                        st.write(f"**Result:** {label}")

        with col_vid:
            st.subheader("📽️ Video Input")
            vid_file = st.file_uploader("Upload Video", type=['mp4', 'mov'], key="vid_test")
            if vid_file:
                st.video(vid_file)
                if st.button("Run Vision Model (Vid)", use_container_width=True):
                    with st.spinner("Scanning frames..."):
                        score, label = run_vision_inference(vid_file, is_video=True)
                        st.metric("Confidence", f"{score:.1%}")
                        st.write(f"**Result:** {label}")

        with col_aud:
            st.subheader("🔊 Audio Input")
            aud_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'], key="aud_test")
            if aud_file:
                st.audio(aud_file)
                if st.button("Run Audio Model", use_container_width=True):
                    with st.spinner("Analyzing frequencies..."):
                        score, label = run_audio_inference(aud_file)
                        st.metric("Confidence", f"{score:.1%}")
                        st.write(f"**Result:** {label}")

        st.markdown("---")
        st.info("💡 **Note:** The `type` parameter in the uploader strictly prevents users from uploading incorrect formats (e.g., an MP3 will not be accepted in the Image section).")
    # comment
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #9ca3af; font-size: 0.85rem; padding: 1rem 0;'>
        <p><strong>ForestGuard</strong> • Powered by NASA FIRMS and El Quarters (VIIRS/MODIS) • Data refreshed every 3 hours, hola ig, herya xaina aaile samma</p>
        <p>🌍 Protecting forests through intelligent monitoring</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ Error loading data: {str(e)}")
    st.info("Please refresh the page or check your connection.")
    st.exception(e)