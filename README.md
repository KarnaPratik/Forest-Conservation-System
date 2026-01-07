# 🔥 VanaRakshya - AI-Powered Forest Conservation System 
**[Developed for LOCUS HACK-A-WEEK-2026 EVENT - Selected for Finals]**

An intelligent, multi-modal AI application designed for forest rangers and officials to monitor and protect biodiversity in real-time. VanaRakshya detects wildfires, illegal logging, and poaching by analyzing live feeds from drones, GoPros, and CCTVs connected via local WLAN using edge-computing machine learning models.

---

## 🙌 Team
Developed by: **Pratik Karna, Mahesh Panta, Abhyudaya Pokhrel, Aadim Sapkota**

## 🌟 Features

### 🗺️ **Interactive AI Detection Map**
- Real-time visualization of AI-detected incidents (Vision and Audio).
- GPS-synced markers showing exactly where a threat was identified.
- Interactive 3D rendering of terrain and hotspots using PyDeck.

### ⚠️ **Smart Alert System**
- **Vision AI:** Real-time fire and smoke detection from aerial and ground cameras.
- **Audio AI:** Detects unnatural sounds like chainsaws (logging) and gunshots (poaching).
- One-click dispatch system to send Ranger Units to specific coordinates.

### 📊 **Incident Analytics**
- Historical tracking of forest threats and AI detections.
- Regional distribution analysis (e.g., Chitwan, Bardiya, Parsa forest sectors).
- Trend visualization to identify high-risk zones over time.

### 🎥 **Live Surveillance Hub**
- Integrated MJPEG streaming for low-latency live monitoring.
- Dual-model consensus logic to ensure high accuracy and reduce false positives.

---

## 🚀 Installation & Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- [Optional] IP Camera or mobile device with a camera app for live testing

### 1. Clone & Install
```bash
# Clone the repository
git clone [https://github.com/KarnaPratik/Forest-Conservation-System](https://github.com/KarnaPratik/Forest-Conservation-System)

# Navigate to the project directory
cd Forest-Conservation-System

pip install -r requirements.txt
streamlit run landing_page.py #to run the landing page

#run in separate terminal
python audio_stream.py #flask app to receive live signal and audio run the inference


#run in separate terminal
python live_stream.py #flask app to recieve live video and audio signal to give bimodal prediction

# Install dependencies
pip install -r requirements.txt
