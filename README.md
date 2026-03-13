# WebWorkutePoseAnalyzer

A full-stack web application for analyzing workout poses using computer vision. Upload exercise videos, capture key poses at specific timestamps, and generate structured exercise configurations with joint angle data -- built for the Workute fitness platform.

## Tech Stack

| Layer    | Technology                                           |
|----------|------------------------------------------------------|
| Frontend | React 19, Bootstrap, JavaScript, HTML5, CSS3         |
| Backend  | Python, Flask, Flask-CORS                            |
| AI/Vision| MediaPipe Pose, OpenCV, NumPy                        |

## Features

- **Video-based pose capture** -- Upload workout videos and mark key timestamps to capture representative poses for each exercise state.
- **Joint angle analysis** -- MediaPipe detects body landmarks and calculates angles for 10 joints: left/right shoulders, elbows, armpits, waist, and knees.
- **Angle range fine-tuning** -- Interactive editor for adjusting min/max angle thresholds across three difficulty levels (strong, normal, weak).
- **Exercise configuration builder** -- Define exercise metadata, repetition step patterns, and body part display settings through a guided interface.
- **Mirror mode** -- Option to average symmetrical left/right joint angles for exercises performed facing the camera.
- **JSON export** -- Generate, preview, copy, and download exercise definition files ready for the Workute platform.

## Application Sections

1. **Exercise Info** -- Set exercise key, title, hints, and descriptions
2. **Exercise Steps** -- Build repetition patterns and step sequences
3. **Body Part Display** -- Configure focus positions and display lines
4. **Motion Analysis** -- Upload videos, playback, and capture poses at timestamps
5. **Angle Fine-tuning** -- Edit angle ranges per status with tab-based navigation
6. **Generate and Export** -- Preview JSON output, copy to clipboard, or download

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+
- pip, npm

### Installation

```bash
git clone https://github.com/MelodyccLo/WebWorkutePoseAnalyzer.git
cd WebWorkutePoseAnalyzer

# Backend
cd backend
pip install flask flask-cors mediapipe opencv-python numpy

# Frontend
cd ../frontend
npm install
```

### Running

```bash
# Start the backend (port 3000)
cd backend
python app.py

# Start the frontend (port 3001)
cd frontend
npm start
```

## API Reference

### POST `/process_video`

Analyze a workout video and extract joint angles at specified timestamps.

**Request** (multipart/form-data):
| Field            | Type   | Description                              |
|------------------|--------|------------------------------------------|
| video            | file   | Video file to analyze                    |
| capture_times    | JSON   | Array of timestamps to capture poses     |
| mirror           | bool   | Whether to average left/right angles     |
| range_width      | number | Degree range for angle suggestions (default: 20) |

**Response**: JSON object with detected angles and suggested ranges for each captured pose.

## Project Structure

```
WebWorkutePoseAnalyzer/
├── backend/
│   └── app.py                # Flask API server with MediaPipe processing
└── frontend/
    ├── package.json
    ├── public/
    │   └── workute.html      # Main application interface
    └── src/                  # React boilerplate
```
