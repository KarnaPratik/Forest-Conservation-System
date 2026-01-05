# 🔥 VanaRakshya - AI-Powered Forest Conservation System
An intelligent web application for forest rangers and officials to detect and monitor wildfires, illegal logging and poaching in real-time using NASA satellite data, drone imagery, and machine learning.

---

## 🙌 Team
Developed by: 
Pratik Karna 
Mahesh Panta 
Abhyudaya Pokhrel 
Aadim Sapkota

## 🌟 Features

### 🗺️ **Interactive Live Map**
- Real-time visualization of active fire hotspots
- Color-coded severity indicators (High/Medium/Low)
- Interactive markers with detailed fire information
- 3D map rendering with PyDeck

### ⚠️ **Smart Alert System**
- Prioritized list of most severe fires
- Detailed fire metrics (brightness, confidence, coordinates)
- One-click alert deployment to ranger stations
- Comprehensive fire intelligence cards, alerts for poaching and illegal logging

### 📊 **Timeline Analytics**
- Historical fire activity tracking
- Regional distribution analysis
- Daily breakdown with interactive date slider
- Trend visualization with charts and graphs

### 🎛️ **Flexible Filtering**
- Time range selector (1-14 days)
- Confidence threshold adjustment
- Regional focus filters
- Real-time data updates

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/KarnaPratik/Forest-Conservation-System

```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run landingPage.py
```

4. **Open in browser**
The app will automatically open at `http://localhost:8501`

---

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
pydeck>=0.8.0
requests>=2.31.0
```

Install all at once:
```bash
pip install streamlit pandas numpy pydeck requests
```

---

## 🛠️ Project Structure

```
forestguard/
│
├── landingPage.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── data/                  # Data directory (optional)
│   └── sample_fires.csv   # Sample fire data
│
├── models/                # ML models (future)
│   └── fire_detector.h5   # Trained detection model
│
└── assets/                # Images and resources
    └── logo.png           # ForestGuard logo
```

---

## 🔧 Configuration

### Using Real NASA FIRMS Data

To connect to live NASA satellite data:

1. **Get API Key**
   - Visit: [https://firms.modaps.eosdis.nasa.gov/api/](https://firms.modaps.eosdis.nasa.gov/api/)
   - Register for a free MAP_KEY

2. **Update the code** in `landingPage.py`:

```python
import requests
from io import StringIO

def fetch_real_fire_data(days_back=7, map_key="YOUR_KEY_HERE"):
    """Fetch real-time fire data from NASA FIRMS API"""
    
    date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_NRT/world/1/{date}"
    
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text))
    
    # Rename columns to match app
    df = df.rename(columns={
        'latitude': 'latitude',
        'longitude': 'longitude',
        'bright_ti4': 'brightness',
        'confidence': 'confidence',
        'acq_date': 'acq_date',
        'acq_time': 'acq_time'
    })
    
    # Add region (use reverse geocoding API like Nominatim)
    df['region'] = "Unknown Region"
    
    return df
```

3. **Replace mock data call**:
```python
# Replace this:
df = generate_mock_fire_data(days_back)

# With this:
df = fetch_real_fire_data(days_back, map_key="YOUR_KEY_HERE")
```

---

## 🎯 Usage Guide

### For Forest Rangers

1. **Monitor Active Fires**
   - Navigate to the "Live Map" tab
   - Hover over markers for fire details
   - Adjust filters in sidebar to focus on your region

2. **Respond to Alerts**
   - Check "Active Alerts" tab for prioritized fires
   - Click "Deploy" to alert nearest ranger station
   - View detailed metrics for each fire

3. **Analyze Trends**
   - Use "Timeline Analytics" to identify patterns
   - Compare current activity to historical data
   - Export data for reporting

### For Officials

1. **Dashboard Overview**
   - View key metrics at a glance (top stat cards)
   - Monitor high-severity fire count
   - Track affected regions

2. **Strategic Planning**
   - Analyze regional distribution
   - Identify fire-prone areas
   - Allocate resources based on data

3. **Generate Reports**
   - Export daily breakdown data
   - Use charts for presentations
   - Share insights with stakeholders

---


## 🎨 Customization

### Change Color Scheme

Edit the CSS in `landingPage.py`:

```python
st.markdown("""
<style>
    .stat-box {
        border-left: 4px solid #YOUR_COLOR;
    }
    .stat-number {
        color: #YOUR_COLOR;
    }
</style>
""", unsafe_allow_html=True)
```

### Modify Severity Thresholds

Update the `get_severity()` function:

```python
def get_severity(brightness):
    if brightness >= 370:  # Change threshold
        return "CRITICAL", "#8b0000"  # New level
    elif brightness >= 360:
        return "HIGH", "#dc2626"
    # ... add more levels
```

### Add New Regions

Extend the mock data or API filters:

```python
locations = [
    {"lat": YOUR_LAT, "lon": YOUR_LON, "region": "YOUR_REGION", "base_brightness": 345},
    # Add more...
]
```

---


## 🐛 Troubleshooting

### App won't start
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Try running with verbose mode
streamlit run app.py --logger.level=debug
```

### Blank page / No data showing
- Ensure `st.set_page_config()` is the first Streamlit command
- Check browser console for JavaScript errors
- Try a different browser (Chrome recommended)
- Clear Streamlit cache: `streamlit cache clear`

### Map not rendering
```bash
# Install missing dependencies
pip install pydeck

# Check if PyDeck is working

python -c "import pyde



