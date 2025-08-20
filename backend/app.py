import os
import cv2
import mediapipe as mp
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# ================================================
#           INITIALIZE FLASK APP
# ================================================
app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.INFO)
# ================================================

# Configuration & Constants
mp_pose = mp.solutions.pose
pose_processor = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

JOINT_PAIRS = {
    "Shoulder": ("L Shoulder", "R Shoulder"), "Elbow": ("L Elbow", "R Elbow"),
    "Armpit": ("L Armpit", "R Armpit"), "Waist": ("L Waist", "R Waist"),
    "Knee": ("L Knee", "R Knee"),
}
ALL_JOINTS = [item for pair in JOINT_PAIRS.values() for item in pair]

# Helper Functions
def calculate_angle(a, b, c):
    if not all([a, b, c]): return 0.0
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_angles_from_landmarks(landmarks):
    lm = landmarks.landmark
    pl = mp_pose.PoseLandmark
    return {
        "L Shoulder": calculate_angle(lm[pl.LEFT_ELBOW], lm[pl.LEFT_SHOULDER], lm[pl.RIGHT_SHOULDER]),
        "R Shoulder": calculate_angle(lm[pl.RIGHT_ELBOW], lm[pl.RIGHT_SHOULDER], lm[pl.LEFT_SHOULDER]),
        "L Elbow": calculate_angle(lm[pl.LEFT_SHOULDER], lm[pl.LEFT_ELBOW], lm[pl.LEFT_WRIST]),
        "R Elbow": calculate_angle(lm[pl.RIGHT_SHOULDER], lm[pl.RIGHT_ELBOW], lm[pl.RIGHT_WRIST]),
        "L Armpit": calculate_angle(lm[pl.LEFT_ELBOW], lm[pl.LEFT_SHOULDER], lm[pl.LEFT_HIP]),
        "R Armpit": calculate_angle(lm[pl.RIGHT_ELBOW], lm[pl.RIGHT_SHOULDER], lm[pl.RIGHT_HIP]),
        "L Waist": calculate_angle(lm[pl.LEFT_SHOULDER], lm[pl.LEFT_HIP], lm[pl.LEFT_KNEE]),
        "R Waist": calculate_angle(lm[pl.RIGHT_SHOULDER], lm[pl.RIGHT_HIP], lm[pl.RIGHT_KNEE]),
        "L Knee": calculate_angle(lm[pl.LEFT_HIP], lm[pl.LEFT_KNEE], lm[pl.LEFT_ANKLE]),
        "R Knee": calculate_angle(lm[pl.RIGHT_HIP], lm[pl.RIGHT_KNEE], lm[pl.RIGHT_ANKLE]),
    }

def calculate_suggested_range(angle, range_width=20, round_to=5):
    delta = range_width / 2
    if (angle - delta) < 0:
        raw_lower = 0
        raw_upper = range_width
    elif (angle + delta) > 180:
        raw_lower = 180 - range_width
        raw_upper = 180
    else:
        raw_lower = angle - delta
        raw_upper = angle + delta
    final_lower = round(raw_lower / round_to) * round_to
    final_upper = round(raw_upper / round_to) * round_to
    return {"min": int(final_lower), "max": int(final_upper)}

# ================================================
#           API ENDPOINT
# ================================================
@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    try:
        # 1. Get data from the front-end
        if 'video' not in request.files: 
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        captures = json.loads(request.form.get('captures', '[]'))
        mirror_settings = json.loads(request.form.get('mirrorSettings', '{}'))
        
        # --- NEW: Get range_width from request, with a default of 20 ---
        range_width = int(request.form.get('rangeWidth', 20))

        if not captures:
             return jsonify({"error": "No capture timestamps provided"}), 400

        app.logger.info(f"Received {len(captures)} captures, range width {range_width}, and mirror settings: {mirror_settings}")

        temp_video_path = f"temp_{video_file.filename}"
        video_file.save(temp_video_path)

        # 2. Get Raw Angles for all captured frames
        raw_angles_by_status = {}
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened(): 
            app.logger.error("Could not open video file.")
            return jsonify({"error": "Could not open video file"}), 500
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        for capture in captures:
            status_name = capture['statusName']
            timestamp = capture['time']
            frame_number = int(timestamp * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_processor.process(image)
                if results.pose_landmarks:
                    angles = get_angles_from_landmarks(results.pose_landmarks) 
                    if status_name not in raw_angles_by_status:
                        raw_angles_by_status[status_name] = []
                    raw_angles_by_status[status_name].append(angles)
                else:
                    app.logger.warning(f"No pose detected in frame {frame_number} for status '{status_name}'.")
        cap.release()
        os.remove(temp_video_path)
        
        if not raw_angles_by_status:
            return jsonify({"error": "No poses could be detected in any of the captured frames."}), 400
        
        # 3. Calculate Averaged & Mirrored Angles
        processed_angles = {}
        for status, captured_frames in raw_angles_by_status.items():
            if not captured_frames: continue
            
            status_averages = {}
            for pair_name, (l_joint, r_joint) in JOINT_PAIRS.items():
                l_angles = [frame[l_joint] for frame in captured_frames]
                r_angles = [frame[r_joint] for frame in captured_frames]
                if mirror_settings.get(pair_name, False):
                    combined_angles = l_angles + r_angles
                    if combined_angles:
                        unified_avg = np.mean(combined_angles)
                        status_averages[l_joint] = unified_avg
                        status_averages[r_joint] = unified_avg
                else:
                    if l_angles: status_averages[l_joint] = np.mean(l_angles)
                    if r_angles: status_averages[r_joint] = np.mean(r_angles)
            
            # 4. Generate suggested min/max ranges from averages
            suggested_ranges = {}
            for joint_name, avg_angle in status_averages.items():
                # --- UPDATED: Pass the range_width to the calculation ---
                suggested_ranges[joint_name] = calculate_suggested_range(avg_angle, range_width=range_width)
                
            processed_angles[status] = {"angles": suggested_ranges}

        app.logger.info("Successfully processed all captures. Returning data.")
        return jsonify(processed_angles)

    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# Run Server
if __name__ == '__main__':
    app.run(debug=True, port=3000)