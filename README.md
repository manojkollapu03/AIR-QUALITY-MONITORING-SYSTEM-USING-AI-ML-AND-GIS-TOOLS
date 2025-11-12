🌍 Air Quality Monitoring System using AI/ML and GIS Tools
🧾 Project Overview

The Air Quality Monitoring System using AI/ML and GIS Tools is a real-time dashboard that collects, analyzes, and visualizes air pollution data from multiple sources. It uses Machine Learning algorithms such as Random Forest, XGBoost, and LSTM to predict future AQI levels and integrates GIS visualization for location-based pollution mapping. The system also features multilingual voice narration using Google Text-to-Speech (gTTS) for accessibility and provides health advisories based on AQI categories.

This project aims to raise public awareness about air pollution and assist government bodies, researchers, and citizens in making data-driven environmental decisions.

🚀 Features

📊 Real-Time AQI Monitoring – Fetches live air quality data via APIs and Kaggle datasets.

🤖 AI/ML Prediction – Uses Random Forest, XGBoost, and LSTM models to forecast AQI trends.

🗺️ GIS-Based Visualization – Displays pollutant levels on an interactive map using Folium and Streamlit.

🔊 Voice Narration (gTTS) – Converts AQI reports into speech in multiple languages.

💡 Health Advisory System – Suggests safety measures based on AQI category.

🧩 Streamlit Dashboard – Clean and interactive web interface for end users.

🧠 System Architecture
Data Sources (Kaggle Dataset, APIs)
        ↓
Data Preprocessing (Cleaning, Validation)
        ↓
AI/ML Prediction Engine (Random Forest, XGBoost, LSTM)
        ↓
Dashboard Application (Streamlit)
        ↓
 ┌───────────────┬──────────────────┬────────────────────┐
 │ Map Integration│ Voice Synthesis  │ Health Advisory    │
 │ (Folium + GIS) │ (gTTS)           │ (AQI-based Alerts) │
 └───────────────┴──────────────────┴────────────────────┘
        ↓
          →  End Users (Visuals, Audio, Insights)

🧩 Tech Stack
Category	Tools / Libraries
Frontend	Streamlit
Backend / Logic	Python
Machine Learning	scikit-learn, XGBoost, TensorFlow/Keras
Visualization	Plotly, Folium
Speech Synthesis	gTTS (Google Text-to-Speech)
Data Handling	Pandas, NumPy
Dataset Source	Kaggle Air Quality Dataset, OpenWeather API

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/yourusername/Air-Quality-Monitoring-System.git
cd Air-Quality-Monitoring-System

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

4. Access the dashboard

Open your browser and go to:
👉 http://localhost:8501

📁 Project Structure
📦 Air-Quality-Monitoring-System
├── app.py                      # Main Streamlit application
├── requirements.txt            # List of dependencies
├── models/                     # (Optional) Pre-trained AI/ML models
├── data/                       # Raw / cleaned datasets
├── assets/                     # Images, diagrams, and icons
└── README.md                   # Project documentation

🧬 Machine Learning Models

The system uses the following algorithms for AQI prediction:

Random Forest Regressor – For fast, tree-based AQI forecasting.

XGBoost Regressor – For high-accuracy, gradient-boosted AQI predictions.

LSTM (Long Short-Term Memory) – For time-series prediction based on historical air quality data.

🔊 Voice Narration (gTTS)

The application includes a voice assistant feature powered by Google Text-to-Speech (gTTS), which narrates the current or predicted AQI values.
This feature supports multilingual output for accessibility and user engagement.

📉 Example Outputs

Real-time AQI charts by city

Predicted AQI values for the next 24 hours

Interactive GIS map showing pollutant levels

Voice-based AQI summary

Health advisory messages based on air quality conditions


📚 References

Kaggle Air Quality Data in India

OpenWeatherMap API

Streamlit Documentation

Google Text-to-Speech (gTTS)
