import os
import cv2
import gc
import io
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
from keras.applications.efficientnet import preprocess_input
from audio_to_img import for_single_audio
import time
import sqlite3

# --- 0. ESTABLISH DATABASE CONNECTION ---
def log_detection_to_db(source, inc_type, conf, lat, lon, region):
    try:
        # Use a local connection inside the function for thread safety in Flask
        conn = sqlite3.connect('vanarakshya.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''INSERT INTO alerts (source, incident_type, confidence, latitude, longitude, region)
                     VALUES (?, ?, ?, ?, ?, ?)''', (source, inc_type, conf, lat, lon, region))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database Error: {e}")

# Global tracker to prevent duplicate logs (e.g., log once every 30 seconds per type)
last_log_times = {}

# --- 1. CONFIGURATION & MODELS ---
custom_dict = {'preprocess_input': preprocess_input}

def load_vision_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models', 'best_v2.pt')
    return YOLO(model_path)

def load_audio_model():
    m1 = tf.keras.models.load_model('./models/audio_forest_69.keras', custom_objects=custom_dict)
    m2 = tf.keras.models.load_model('./models/audio_forest_69420.keras', custom_objects=custom_dict)
    return m1, m2

# Load models globally once upon import
vision_model = load_vision_model()
audio_model1, audio_model2 = load_audio_model()

# --- 2. VISION INFERENCE ---

def run_vision_inference(file_buffer=None, is_video=False, FRAME_ARRAY=None):
    """
    Handles Live frames (OpenCV), Image uploads, and Video file processing.
    Returns: (confidence, label, annotated_image_rgb)
    """
    
    # CASE 1: Live Frame (Used by Flask/OpenCV)
    if FRAME_ARRAY is not None:
        results = vision_model.predict(source=FRAME_ARRAY, imgsz=640, conf=0.35, verbose=False)
        annotated_img = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        if len(results[0].boxes) > 0:
            conf = float(results[0].boxes[0].conf[0])
            label = results[0].names[int(results[0].boxes[0].cls[0])]
    
    #To log the detected issue in the firebase
            if conf > 0.5:
                    current_time = time.time()
                    # Only log if we haven't logged this type of incident in the last 30 seconds
                    if label not in last_log_times or (current_time - last_log_times[label] > 30):
                        log_detection_to_db(
                            source="Drone Live Feed",
                            inc_type=label.capitalize(),
                            conf=conf,
                            lat=29.11,  # You can pass real GPS here later
                            lon=82.94, 
                            region="Test Region"
                        )
                        last_log_times[label] = current_time
            
            return conf, label, annotated_rgb
        return 0.0, "No detection", annotated_rgb

    # CASE 2: Image Upload
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

    # CASE 3: Video File Processing (Background Analysis)
    elif is_video and file_buffer is not None:
        import tempfile
        highest_conf = 0.0
        top_label = "Scanning..."

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(file_buffer.read())
            temp_path = tfile.name
        
        try:
            # Process video frames silently
            results = vision_model.predict(source=temp_path, stream=True, conf=0.25)
            for result in results: 
                if len(result.boxes) > 0:
                    current_conf = float(result.boxes[0].conf[0])
                    if current_conf > highest_conf:
                        highest_conf = current_conf
                        top_label = result.names[int(result.boxes[0].cls[0])]
            
            del results
            gc.collect()
            return highest_conf, top_label, None

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return 0.0, "Invalid Input", None

# --- 3. AUDIO INFERENCE ---

def run_audio_inference(file_buffer=None,FRAME_ARRAY=None):
    """
    Processes audio buffer and returns: (confidence, label)
    """
    class_names = ['natural sound', 'unnatural']
    class_names2 = ['fire', 'logging', 'poaching']
    if not file_buffer is None:
        img_ready, _, _ = for_single_audio(file_buffer)
    img_for_model = preprocess_input(img_ready)

    # Level 1: Natural vs Unnatural
    y_pred1 = audio_model1.predict(img_for_model, verbose=0)
    confidence = float(y_pred1[0][0])
    class_index = int(confidence > 0.5)
    
    label = class_names[class_index]
    final_conf = confidence if class_index == 1 else (1 - confidence)

    # Level 2: Specific Threat Detection
    if class_index == 1: # If 'unnatural'
        y_pred2 = audio_model2.predict(img_for_model, verbose=0)
        class_index2 = np.argmax(y_pred2[0])
        label = class_names2[class_index2]
        final_conf = float(y_pred2[0][class_index2])
    
    return final_conf, label

